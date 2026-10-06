"""Render the 30-second film: capture every frame of the page's scripted timeline, add the synthesised sound,
and encode an H.264 MP4.

The page's film mode is a pure function of time, so frames are captured one at a time in headless Chromium
instead of screen-recording in real time: no dropped frames, and rerunning gives the same video.

    python scripts/render_film.py --out docs/unmaze-demo.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).parent))
from film_audio import synth, write_wav  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CHROME = next(Path("/opt/pw-browsers").glob("chromium-*/chrome-linux/chrome"), None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "unmaze-demo.mp4")
    ap.add_argument("--maze", type=int, help="which exported maze to film (default: the page picks one)")
    ap.add_argument("--limit", type=int, help="render only the first N frames (for quick checks)")
    ap.add_argument("--quality", type=int, default=94, help="JPEG quality of captured frames")
    ap.add_argument("--crf", type=int, default=25, help="x264 quality: lower is bigger and better (grain is expensive)")
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args()

    data_js = (ROOT / "web" / "data" / "overlook.js").read_text()
    data = json.loads(data_js[len("window.UNMAZE = "):].rstrip().rstrip(";"))

    url = (ROOT / "web" / "index.html").as_uri() + "?film=1" + (f"&maze={args.maze}" if args.maze is not None else "")
    launch = {"args": ["--no-sandbox", "--disable-gpu"]}
    if CHROME:
        launch["executable_path"] = str(CHROME)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, sync_playwright() as p:
        browser = p.chromium.launch(**launch)
        page = browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url)
        page.wait_for_function("window.FILM_READY === true", timeout=30_000)
        info = page.evaluate("window.FILM_INFO")
        fps, seconds = info["fps"], info["duration"]
        frames = int(fps * seconds) if args.limit is None else min(args.limit, int(fps * seconds))
        print(f"filming maze {info['hero']} of {len(data['mazes'])}: {frames} frames at {fps} fps", flush=True)

        silent = Path(tmp) / "silent.mp4"
        ffmpeg = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(fps), "-c:v", "mjpeg", "-i", "-",
             "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf), "-maxrate", "14M", "-bufsize", "28M", "-pix_fmt", "yuv420p", "-vf", "scale=1920:1080",
             "-movflags", "+faststart", str(silent)],
            stdin=subprocess.PIPE,
        )
        start = time.time()
        for i in range(frames):
            page.evaluate(f"window.renderFilmFrame({i / fps}, {i})")
            ffmpeg.stdin.write(page.screenshot(type="jpeg", quality=args.quality))
            if i % 30 == 29:
                rate = (i + 1) / (time.time() - start)
                print(f"  frame {i + 1}/{frames}  {rate:.1f} fps  eta {(frames - i - 1) / rate:.0f}s", flush=True)
        ffmpeg.stdin.close()
        if ffmpeg.wait() != 0:
            raise SystemExit("ffmpeg failed")
        browser.close()
        if errors:
            raise SystemExit("page errors while filming:\n" + "\n".join(errors))

        if args.no_audio:
            silent.replace(args.out)
        else:
            wav = Path(tmp) / "audio.wav"
            write_wav(wav, synth(info, data["sigma"], seconds))
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), "-i", str(wav), "-c:v", "copy",
                 "-c:a", "aac", "-b:a", "192k", "-shortest", str(args.out)],
                check=True,
            )
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
