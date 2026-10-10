# Quiet Voice

My sister has a medical condition that makes her speak at a very low volume, so
it is often hard to understand her. Quiet Voice listens through a microphone held
near her mouth and shows what she says as large text on screen.

## How it works

```
microphone ─▶ speech detector ─▶ audio cleanup ─▶ Whisper (offline) ─▶ big text on screen
```

1. **Speech detection.** The mic is read in 30 ms blocks. The app keeps learning
   the background noise level of the room, and anything clearly louder than that
   counts as speech. A phrase has to be loud enough to *start*, but once she is
   talking, softer sounds still count (hysteresis), so quiet words in the middle
   of a sentence are not cut off. A pause of 0.8 s ends a phrase.
2. **Audio cleanup.** Each phrase has its DC offset removed, low rumble filtered
   out (from holding the mic), and is boosted up to normal speaking volume.
3. **Recognition.** The cleaned audio goes to OpenAI's Whisper model through
   `faster-whisper`. It runs **completely on this computer**, which matters for
   medical privacy: no audio is ever uploaded.
4. **Display.** The newest phrase is shown in large white text, the previous few
   in smaller grey text underneath.

## Setup (Windows)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

The first run downloads the speech model (about 150 MB for `base.en`).

## Options

| Command | What it does |
|---|---|
| `python app.py --model small.en` | More accurate, a bit slower |
| `python app.py --model tiny.en` | Fastest, for slower laptops |
| `python app.py --list-devices` | Lists microphones and their numbers |
| `python app.py --device 3` | Uses microphone number 3 |

## Using it

- **Level bar:** the bar turns green when the app hears speech. The yellow line
  is the speech threshold.
- **Sensitivity:** if her words don't turn the bar green, move the slider right.
  If background noise keeps triggering it, move it left.
- **A+ / A-:** text size. **Clear:** wipes the screen. **Save:** writes the
  conversation with timestamps to the `transcripts` folder.

## Testing

A recorded sentence was turned down about 40 dB (to roughly whisper volume,
less than 2× louder than the background hiss) and run through the pipeline.
The first version detected only one of the two sentences: the second sentence's
soft sounds were rejected as noise. Adding hysteresis fixed it, and both
sentences were transcribed word for word.

## Ideas for next steps

- Test with real recordings of my sister and measure accuracy (word error rate).
- Show words live while she is still speaking, not only after she pauses.
- A phone or Raspberry Pi version with a clip-on lapel mic, so it is portable.
- Fine-tune Whisper on her voice if accuracy on her speech is low.
