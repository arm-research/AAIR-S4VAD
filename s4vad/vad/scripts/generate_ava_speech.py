# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
import csv
import subprocess
from pathlib import Path
from collections import defaultdict

# ----------------------------
# CONFIG
# ----------------------------
LABELS_CSV = "./data/AVA_speech/ava_speech_labels.csv"               # path to the AVA-Speech labels
OUTPUT_ROOT = Path("./data/AVA_speech/AVA_full/ava_speech_audio")    # root folder for all audio clips
VIDEO_CACHE = Path("./data/AVA_speech/AVA_full/ava_videos")          # where full YouTube videos are cached
SAMPLE_RATE = 16000                       # Hz
AUDIO_EXT = ".wav"                        # output audio format
YTDLP_BIN = "yt-dlp"                      # or "youtube-dl"
MAX_VIDEOS = None                         # set an int for debugging, e.g. 3

# ----------------------------
# Helper functions
# ----------------------------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def download_video(video_id: str) -> Path:
    """
    Download a YouTube video for the given video_id if not already cached.
    Returns the local video path.
    """
    ensure_dir(VIDEO_CACHE)
    # We will save as VIDEO_ID.mp4
    out_tmpl = str(VIDEO_CACHE / f"{video_id}.%(ext)s")
    target_mp4 = VIDEO_CACHE / f"{video_id}.mp4"

    if target_mp4.exists():
        print(f"[INFO] Video already downloaded: {target_mp4}")
        return target_mp4

    # Build YouTube URL from AVA video_id
    url = f"https://www.youtube.com/watch?v={video_id}"
    print(f"[INFO] Downloading {url}")

    cmd = [
        YTDLP_BIN,
        "--ignore-errors",
        "--no-warnings",
        "--extractor-args", "youtube:player_client=default",
        "-f", "mp4/bestvideo[ext=mp4]+bestaudio/best",
        "-o", out_tmpl,
        url,
    ]

    subprocess.run(cmd, check=True)

    # After download, make sure the mp4 exists
    if not target_mp4.exists():
        # Sometimes yt-dlp might have saved as .mkv or something else
        # Find any file starting with video_id.
        matches = list(VIDEO_CACHE.glob(f"{video_id}.*"))
        if not matches:
            raise FileNotFoundError(f"Could not find downloaded file for {video_id}")
        # Rename the first one to .mp4 for consistency
        matches[0].rename(target_mp4)

    return target_mp4


def extract_audio_segment(video_path: Path,
                          start_time: float,
                          end_time: float,
                          out_path: Path):
    """
    Use ffmpeg to extract a mono, SAMPLE_RATE audio segment.
    """
    ensure_dir(out_path.parent)
    duration = end_time - start_time
    if duration <= 0:
        print(f"[WARN] Non-positive segment duration, skipping {out_path}")
        return

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", str(start_time),
        "-t", str(duration),
        "-i", str(video_path),
        "-ac", "1",                    # mono
        "-ar", str(SAMPLE_RATE),       # sample rate
        "-vn",                         # no video
        "-y",                          # overwrite without asking
        str(out_path),
    ]

    subprocess.run(cmd, check=True)


# ----------------------------
# Main script logic
# ----------------------------

def main():
    ensure_dir(OUTPUT_ROOT)
    ensure_dir(VIDEO_CACHE)

    # Map: video_id -> list of segments (start, end, label, index)
    segments_by_video = defaultdict(list)

    # Read CSV
    """
    CSV format (from AVA-Speech docs) is:
      video_id, label_start_timestamp_seconds, label_end_timestamp_seconds, label
    where label ∈ {NO_SPEECH, CLEAN_SPEECH, SPEECH_WITH_MUSIC, SPEECH_WITH_NOISE}
    :contentReference[oaicite:1]{index=1}
    """
    with open(LABELS_CSV, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            video_id, start_s, end_s, label = row
            start_s = float(start_s)
            end_s = float(end_s)
            segments_by_video[video_id].append((start_s, end_s, label))

    print(f"[INFO] Found labels for {len(segments_by_video)} videos.")

    # Optionally limit number of videos (for testing)
    video_items = list(segments_by_video.items())
    if MAX_VIDEOS is not None:
        video_items = video_items[:MAX_VIDEOS]

    # Process each video
    for vid_idx, (video_id, segs) in enumerate(video_items, start=1):
        print(f"[INFO] ({vid_idx}/{len(video_items)}) Processing video {video_id}")

        # 1) Download full video
        try:
            video_path = download_video(video_id)
        except Exception as e:
            print(f"[ERROR] Failed to download {video_id}: {e}")
            continue

        # 2) For each labeled segment, cut its audio
        for seg_idx, (start_s, end_s, label) in enumerate(segs):
            # Sanitize label to directory name
            label_dir = label.lower()
            # e.g. "clean_speech", "speech_with_music", "speech_with_noise", "no_speech"
            out_dir = OUTPUT_ROOT / label_dir
            clip_name = f"{video_id}_{int(start_s*1000)}_{int(end_s*1000)}{AUDIO_EXT}"
            out_path = out_dir / clip_name

            if out_path.exists():
                # Already extracted
                continue

            try:
                extract_audio_segment(video_path, start_s, end_s, out_path)
            except Exception as e:
                print(f"[ERROR] Failed segment {video_id} [{start_s}, {end_s}] ({label}): {e}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
