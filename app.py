"""
Quiet Voice - shows quiet speech as large text on screen.

How it works:
  1. The microphone is read continuously in small 30 ms blocks.
  2. Each block's loudness (RMS) is compared to the room's background
     noise level. Anything clearly louder than the background counts as speech.
  3. When she stops talking for a moment, the speech we collected is
     cleaned up (DC offset removed, low rumble filtered, volume boosted)
     so even very soft speech becomes loud and clear.
  4. That audio is sent to Whisper, a speech recognition model that runs
     entirely on this computer (nothing is uploaded), and the text appears
     on screen in big letters.

Run:  python app.py
      python app.py --model small.en      (more accurate, slower)
      python app.py --list-devices        (find your microphone's number)
      python app.py --device 3            (use a specific microphone)
"""

import argparse
import queue
import threading
import time
import tkinter as tk
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000          # Whisper expects 16 kHz audio
BLOCK_MS = 30
BLOCK_SIZE = SAMPLE_RATE * BLOCK_MS // 1000
PRE_ROLL_MS = 300            # audio kept from just before speech starts
END_SILENCE_MS = 800         # this much quiet means she finished a phrase
MIN_SPEECH_MS = 250          # ignore clicks/bumps shorter than this
MAX_UTTERANCE_S = 15         # force a transcription if she talks for a long time
TARGET_RMS = 0.1             # loudness we boost speech up to before recognition
MAX_GAIN = 60.0              # cap on boost so pure noise isn't amplified forever


# ---------------------------------------------------------------- audio side

def clean_audio(audio: np.ndarray) -> np.ndarray:
    """Make quiet speech loud and clear for the recognizer."""
    audio = audio - np.mean(audio)                     # remove DC offset

    # Simple first-order high-pass filter (~100 Hz) to remove rumble and
    # handling noise from holding the mic.
    alpha = 0.96
    filtered = np.empty_like(audio)
    prev_in = prev_out = 0.0
    for i, x in enumerate(audio):
        prev_out = alpha * (prev_out + x - prev_in)
        prev_in = x
        filtered[i] = prev_out

    rms = np.sqrt(np.mean(filtered ** 2)) + 1e-9
    gain = min(TARGET_RMS / rms, MAX_GAIN)
    return np.clip(filtered * gain, -1.0, 1.0).astype(np.float32)


class SpeechDetector:
    """Splits the live microphone stream into separate phrases."""

    def __init__(self, on_phrase, on_level):
        self.on_phrase = on_phrase       # called with a numpy array per phrase
        self.on_level = on_level         # called with (level, threshold) for the meter
        self.sensitivity = 3.0           # speech must be this many times louder than noise
        self.noise_floor = None
        self.pre_roll = deque(maxlen=PRE_ROLL_MS // BLOCK_MS)
        self.phrase = []
        self.speaking = False
        self.silent_blocks = 0
        self.speech_blocks = 0

    def threshold(self):
        return max(self.noise_floor * self.sensitivity, 0.0008)

    def keep_threshold(self):
        # Once she has started talking, softer sounds still count as speech
        # (halfway between the noise and the start threshold). This stops
        # quiet words in the middle of a sentence from being cut off.
        return (self.noise_floor + self.threshold()) / 2

    def process(self, block: np.ndarray):
        level = float(np.sqrt(np.mean(block ** 2)))

        if self.noise_floor is None:
            self.noise_floor = level
        is_speech = level > self.threshold()

        # Slowly learn the background noise while nobody is talking, so the
        # app adapts to quiet bedrooms and noisy waiting rooms alike.
        if not is_speech and not self.speaking:
            self.noise_floor = 0.95 * self.noise_floor + 0.05 * level

        self.on_level(level, self.threshold())

        if not self.speaking:
            self.pre_roll.append(block)
            if is_speech:
                self.speaking = True
                self.phrase = list(self.pre_roll)
                self.silent_blocks = 0
                self.speech_blocks = 1
            return

        self.phrase.append(block)
        if level > self.keep_threshold():
            self.speech_blocks += 1
            self.silent_blocks = 0
        else:
            self.silent_blocks += 1

        too_long = len(self.phrase) * BLOCK_MS >= MAX_UTTERANCE_S * 1000
        finished = self.silent_blocks * BLOCK_MS >= END_SILENCE_MS
        if finished or too_long:
            if self.speech_blocks * BLOCK_MS >= MIN_SPEECH_MS:
                self.on_phrase(np.concatenate(self.phrase))
            self.speaking = False
            self.phrase = []
            self.pre_roll.clear()


# ---------------------------------------------------------- recognition side

class Transcriber(threading.Thread):
    """Runs Whisper in the background so the window never freezes."""

    def __init__(self, model_name, results: queue.Queue, status: queue.Queue):
        super().__init__(daemon=True)
        self.model_name = model_name
        self.jobs = queue.Queue()
        self.results = results
        self.status = status

    def run(self):
        self.status.put(f"Loading speech model '{self.model_name}' "
                        "(first run downloads it)...")
        from faster_whisper import WhisperModel
        model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
        self.status.put("Listening... hold the mic close and let her speak.")

        while True:
            audio = self.jobs.get()
            self.status.put("Understanding...")
            segments, _ = model.transcribe(
                clean_audio(audio),
                language="en",
                beam_size=5,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            words = []
            for seg in segments:
                # Whisper sometimes "hears" words in pure noise; skip
                # segments it is not confident actually contain speech.
                if seg.no_speech_prob > 0.6 and seg.avg_logprob < -0.8:
                    continue
                words.append(seg.text.strip())
            text = " ".join(w for w in words if w)
            if text:
                self.results.put(text)
            self.status.put("Listening...")


# ------------------------------------------------------------------- window

class App:
    def __init__(self, root, model_name, device):
        self.root = root
        self.font_size = 40
        self.results = queue.Queue()
        self.status = queue.Queue()
        self.levels = queue.Queue()
        self.history = []

        self.transcriber = Transcriber(model_name, self.results, self.status)
        self.detector = SpeechDetector(self.transcriber.jobs.put,
                                       lambda lv, th: self.levels.put((lv, th)))

        self.build_ui()
        self.transcriber.start()

        self.audio_in = queue.Queue()
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=1,
            dtype="float32", device=device,
            callback=lambda data, *_: self.audio_in.put(data[:, 0].copy()),
        )
        self.stream.start()
        threading.Thread(target=self.feed_detector, daemon=True).start()
        self.poll()

    def feed_detector(self):
        while True:
            self.detector.process(self.audio_in.get())

    def build_ui(self):
        r = self.root
        r.title("Quiet Voice")
        r.geometry("1000x650")
        r.configure(bg="black")

        self.latest = tk.Label(r, text="", fg="white", bg="black",
                               wraplength=950, justify="center",
                               font=("Segoe UI", self.font_size, "bold"))
        self.latest.pack(fill="both", expand=True, padx=20, pady=20)

        self.older = tk.Label(r, text="", fg="#888", bg="black",
                              wraplength=950, justify="center",
                              font=("Segoe UI", 16))
        self.older.pack(fill="x", padx=20)

        self.meter = tk.Canvas(r, height=24, bg="#222", highlightthickness=0)
        self.meter.pack(fill="x", padx=20, pady=(10, 0))

        bar = tk.Frame(r, bg="black")
        bar.pack(fill="x", padx=20, pady=10)
        self.status_label = tk.Label(bar, text="Starting...", fg="#6cf",
                                     bg="black", font=("Segoe UI", 12))
        self.status_label.pack(side="left")

        btn = dict(font=("Segoe UI", 12), width=8)
        tk.Button(bar, text="Save", command=self.save, **btn).pack(side="right", padx=4)
        tk.Button(bar, text="Clear", command=self.clear, **btn).pack(side="right", padx=4)
        tk.Button(bar, text="A+", command=lambda: self.zoom(6), **btn).pack(side="right", padx=4)
        tk.Button(bar, text="A-", command=lambda: self.zoom(-6), **btn).pack(side="right", padx=4)

        tk.Label(bar, text="Sensitivity", fg="white", bg="black",
                 font=("Segoe UI", 12)).pack(side="right", padx=(20, 4))
        self.sens = tk.Scale(bar, from_=1, to=10, resolution=0.5,
                             orient="horizontal", showvalue=False,
                             length=160, bg="black",
                             highlightthickness=0, command=self.set_sensitivity)
        self.sens.set(11.5 - self.detector.sensitivity)
        self.sens.pack(side="right")

    def set_sensitivity(self, value):
        # Slider right = more sensitive = lower multiple of the noise floor.
        self.detector.sensitivity = 11.5 - float(value)

    def zoom(self, step):
        self.font_size = max(16, min(96, self.font_size + step))
        self.latest.configure(font=("Segoe UI", self.font_size, "bold"))

    def clear(self):
        self.history.clear()
        self.latest.configure(text="")
        self.older.configure(text="")

    def save(self):
        if not self.history:
            return
        folder = Path(__file__).parent / "transcripts"
        folder.mkdir(exist_ok=True)
        path = folder / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.txt"
        path.write_text("\n".join(f"[{t}] {s}" for t, s in self.history),
                        encoding="utf-8")
        self.status_label.configure(text=f"Saved to {path.name}")

    def draw_meter(self, level, threshold):
        w = self.meter.winfo_width()
        # Use a log scale so quiet sounds still move the bar visibly.
        to_x = lambda v: w * min(1.0, max(0.0, (np.log10(v + 1e-6) + 4) / 3.5))
        color = "#3c3" if level > threshold else "#555"
        self.meter.delete("all")
        self.meter.create_rectangle(0, 0, to_x(level), 24, fill=color, width=0)
        tx = to_x(threshold)
        self.meter.create_line(tx, 0, tx, 24, fill="#fc3", width=2)

    def poll(self):
        while not self.status.empty():
            self.status_label.configure(text=self.status.get())
        while not self.results.empty():
            text = self.results.get()
            self.history.append((datetime.now().strftime("%H:%M:%S"), text))
            self.latest.configure(text=text)
            self.older.configure(text="\n".join(s for _, s in self.history[-4:-1]))
        last = None
        while not self.levels.empty():
            last = self.levels.get()
        if last:
            self.draw_meter(*last)
        self.root.after(50, self.poll)


def main():
    parser = argparse.ArgumentParser(description="Show quiet speech as large text.")
    parser.add_argument("--model", default="base.en",
                        help="Whisper model: tiny.en, base.en, small.en, medium.en")
    parser.add_argument("--device", type=int, default=None,
                        help="microphone number from --list-devices")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    root = tk.Tk()
    App(root, args.model, args.device)
    root.mainloop()


if __name__ == "__main__":
    main()
