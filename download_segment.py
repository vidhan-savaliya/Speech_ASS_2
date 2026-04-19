"""
download_segment.py
───────────────────
Download and trim the required 10-minute lecture segment from YouTube.

Source : https://youtu.be/ZPUtA3W-7_I
Segment: 2 h 20 min → 2 h 30 min  (first 10-minute window in the 2:20–2:54 range)

Usage:
    python download_segment.py
    python download_segment.py --url URL --start 02:20:00 --end 02:30:00

Requirements:
    pip install yt-dlp
    ffmpeg must be on PATH  (https://ffmpeg.org/download.html)
"""

import argparse
import os
import subprocess
import sys

DEFAULT_URL   = "https://www.youtube.com/watch?v=ZPUtA3W-7_I"
DEFAULT_START = "02:20:00"
DEFAULT_END   = "02:30:00"   # 10-minute segment
OUTPUT_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_FILE   = os.path.join(OUTPUT_DIR, "original_segment.wav")


def check_dependency(name: str) -> bool:
    """Return True if a command-line tool exists on PATH."""
    try:
        subprocess.run([name, "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def time_to_seconds(t: str) -> int:
    """Convert HH:MM:SS to total seconds."""
    parts = list(map(int, t.split(":")))
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return int(parts[0])


def download_segment(url: str, start: str, end: str, output: str) -> None:
    os.makedirs(os.path.dirname(output), exist_ok=True)

    if os.path.exists(output):
        print(f"[download_segment] Output already exists: {output}")
        return

    # Verify dependencies — use python -m yt_dlp so PATH doesn't matter
    if not check_dependency("ffmpeg"):
        sys.exit("[!] 'ffmpeg' not found. Download from https://ffmpeg.org and add to PATH.")

    start_sec = time_to_seconds(start)
    duration  = time_to_seconds(end) - start_sec

    print(f"[download_segment] Downloading: {url}")
    print(f"[download_segment] Trim: {start} → {end}  ({duration}s)")

    # Step 1 — download best audio to a temp file
    # Use `python -m yt_dlp` so it works without yt-dlp.exe on PATH
    tmp_audio = os.path.join(OUTPUT_DIR, "_tmp_full_audio.%(ext)s")
    dl_cmd = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist",
        "-f", "bestaudio",
        "-o", tmp_audio,
        "--no-warnings",
        url,
    ]
    print("[download_segment] Downloading audio (this may take a few minutes)…")
    subprocess.run(dl_cmd, check=True)

    # Find the downloaded file (extension varies)
    tmp_files = [
        os.path.join(OUTPUT_DIR, f)
        for f in os.listdir(OUTPUT_DIR)
        if f.startswith("_tmp_full_audio")
    ]
    if not tmp_files:
        sys.exit("[!] yt-dlp download succeeded but output file not found.")
    tmp_file = tmp_files[0]

    # Step 2 — trim and convert to 16 kHz mono WAV with ffmpeg
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", tmp_file,
        "-ss", str(start_sec),
        "-t",  str(duration),
        "-ac", "1",           # mono
        "-ar", "16000",       # 16 kHz
        "-sample_fmt", "s16",
        output,
    ]
    print("[download_segment] Trimming and converting to 16 kHz mono WAV…")
    subprocess.run(ffmpeg_cmd, check=True)

    # Clean up temp file
    os.remove(tmp_file)
    print(f"[download_segment] Done → {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download lecture segment from YouTube")
    parser.add_argument("--url",    default=DEFAULT_URL,   help="YouTube URL")
    parser.add_argument("--start",  default=DEFAULT_START, help="Start time HH:MM:SS")
    parser.add_argument("--end",    default=DEFAULT_END,   help="End   time HH:MM:SS")
    parser.add_argument("--output", default=OUTPUT_FILE,   help="Output WAV path")
    args = parser.parse_args()

    download_segment(args.url, args.start, args.end, args.output)
