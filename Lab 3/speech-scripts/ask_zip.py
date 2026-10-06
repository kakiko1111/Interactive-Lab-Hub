#!/usr/bin/env python3
"""Ask for a zip code out loud, record the spoken answer, and log what was heard."""

import csv
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from faster_whisper import WhisperModel

HERE = Path(__file__).resolve().parent
VOICES_DIR = HERE.parent / "voices"
LOG_FILE = HERE / "zip_answers.csv"
VOICE = "en_US-lessac-medium"
MODEL_SIZE = "base.en"
RECORD_SECONDS = 5

# Spoken digit words that should count as digits.
WORD_TO_DIGIT = {
    "zero": "0", "oh": "0", "o": "0",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
}


def say(text):
    """Speak text through Piper, streaming straight to the speaker."""
    piper = subprocess.Popen(
        [sys.executable, "-m", "piper",
         "--model", VOICE,
         "--data-dir", str(VOICES_DIR),
         "--output-raw",
         "--", text],
        stdout=subprocess.PIPE,
    )
    subprocess.run(
        ["aplay", "-q", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-"],
        stdin=piper.stdout,
    )
    piper.wait()


def record(path):
    """Record a fixed-length answer from the default microphone."""
    subprocess.run(
        ["arecord", "-q", "-d", str(RECORD_SECONDS),
         "-f", "S16_LE", "-c", "1", "-r", "16000", str(path)],
        check=True,
    )


def transcribe(model, path):
    segments, _ = model.transcribe(str(path), beam_size=1, language="en")
    return " ".join(seg.text.strip() for seg in segments).strip()


def extract_digits(text):
    """Pull digits out of a transcript, e.g. '10,044' or 'one oh oh four four'."""
    digits = []
    for token in re.findall(r"[a-z]+|\d+", text.lower()):
        if token.isdigit():
            digits.append(token)
        elif token in WORD_TO_DIGIT:
            digits.append(WORD_TO_DIGIT[token])
    return "".join(digits)


def log_answer(timestamp, transcript, digits, valid, audio_file):
    new_file = not LOG_FILE.exists()
    with LOG_FILE.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["time", "transcript", "digits", "valid", "audio_file"])
        writer.writerow([timestamp, transcript, digits, valid, audio_file])


def main():
    print(f"Loading {MODEL_SIZE}...")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

    say("Hi! What is your five digit zip code?")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_file = HERE / f"zip_answer_{timestamp}.wav"
    print(f"Listening for {RECORD_SECONDS} seconds... speak now!")
    record(audio_file)

    transcript = transcribe(model, audio_file)
    digits = extract_digits(transcript)
    valid = len(digits) == 5

    print(f"Transcript: {transcript!r}")
    print(f"Digits:     {digits or '(none)'}")
    print(f"Valid zip:  {valid}")

    if valid:
        # Spaces make Piper read the digits one by one instead of as a big number.
        say(f"Got it. Your zip code is {' '.join(digits)}. Thank you!")
    else:
        say(f"Sorry, I heard {len(digits)} digits. A zip code should have five.")

    log_answer(timestamp, transcript, digits, valid, audio_file.name)
    print(f"Saved to {LOG_FILE.name}")


if __name__ == "__main__":
    main()