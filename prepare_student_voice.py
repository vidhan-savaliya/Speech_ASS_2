"""
prepare_student_voice.py
─────────────────────────────────────────────────────────────────────────────
Task 3.1 Pre-step — Student Voice Preparation

Author : Vidhatha (Student — Speech Understanding PA2)
Purpose: Convert my personal voice recording (Myvoice.m4a) into the
         16 kHz / 60-second mono WAV required by the pipeline as the
         speaker-reference for zero-shot cross-lingual voice cloning.

Recording details:
  • File    : Myvoice.m4a          (root of repository)
  • Content : ~60 s of my own voice reading a short Hindi-English passage
  • Purpose : X-Vector (ECAPA-TDNN) speaker embedding extraction (Task 3.1)
             MCD evaluation reference (Task 3.3 / evaluation)

Conversion steps performed here:
  1. Load Myvoice.m4a via torchaudio (ffmpeg backend).
  2. Mix down to mono if stereo.
  3. Resample to 16 000 Hz (wav2vec2 / SpeechBrain standard).
  4. Trim / pad to exactly 60 seconds.
  5. Save to data/student_voice_ref.wav.

Usage:
    python prepare_student_voice.py
"""

import os
import sys
import subprocess

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MY_VOICE_SRC  = os.path.join(BASE_DIR, "Myvoice.m4a")   # MY OWN voice recording
OUTPUT_WAV    = os.path.join(BASE_DIR, "data", "student_voice_ref.wav")

TARGET_SR     = 16_000   # 16 kHz — required by SpeechBrain ECAPA-TDNN
TARGET_DUR_S  = 60       # exactly 60 seconds as per task requirement


def convert_my_voice() -> None:
    """
    Convert Myvoice.m4a → data/student_voice_ref.wav.

    This is MY OWN voice (student reference), recorded personally for
    Task 3.1 (Speaker Embedding Extraction) and used as the MCD
    evaluation reference throughout Part III.
    """
    print("=" * 65)
    print("  Student Voice Preparation — Task 3.1")
    print("  Source : Myvoice.m4a  (my own voice recording)")
    print("  Author : Vidhatha  |  Speech Understanding PA2")
    print("=" * 65)

    if not os.path.exists(MY_VOICE_SRC):
        sys.exit(
            f"[!] 'Myvoice.m4a' not found at:\n    {MY_VOICE_SRC}\n"
            "    Place your 60-second personal voice recording there and retry."
        )

    os.makedirs(os.path.dirname(OUTPUT_WAV), exist_ok=True)

    print(f"[voice] Processing: {MY_VOICE_SRC}")
    print("[voice] Using ffmpeg to resample (16kHz), mono mix-down, and hard trim to 60s...")
    
    # ── FFmpeg Native Conversion ──────────────────────────────────────────────
    # We use ffmpeg directly (like download_segment.py) because torchaudio
    # can sometimes struggle with .m4a containers on Windows native backends.
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", MY_VOICE_SRC,
        "-t", str(TARGET_DUR_S), # Truncate exactly to 60s
        "-ac", "1",              # Mono 
        "-ar", str(TARGET_SR),   # 16000 Hz
        "-sample_fmt", "s16",    # 16-bit
        # audio normalization via af
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", 
        OUTPUT_WAV
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        size_kb = os.path.getsize(OUTPUT_WAV) / 1024
        print(f"[voice] Saved -> {OUTPUT_WAV}  ({size_kb:.1f} KB)")
        print()
        print("  [DONE]  student_voice_ref.wav is ready for:")
        print("       Task 3.1 — X-Vector / ECAPA-TDNN speaker embedding")
        print("       Task 3.3 — MCD evaluation reference (target < 8.0 dB)")
        print()
    except FileNotFoundError:
        sys.exit("[!] 'ffmpeg' not found in PATH. Please install ffmpeg.")
    except subprocess.CalledProcessError as e:
        sys.exit(f"[!] ffmpeg conversion failed. Code: {e.returncode}")


if __name__ == "__main__":
    convert_my_voice()
