"""
Batch downloader: paste YouTube links, it downloads every one into ./input (where clipfarm.py watches).

Run:
    venv\\Scripts\\python.exe getvideos.py              (then paste links, empty line to start)
    venv\\Scripts\\python.exe getvideos.py links.txt    (read links from a text file instead)

Needs yt-dlp:  winget install yt-dlp    (and ffmpeg, which you already have)
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------- settings ----------------
BASE = Path(__file__).parent
IN_DIR = BASE / "input"            # finished videos land here -> clipfarm.py picks them up
TMP_DIR = BASE / "tmp_dl"          # partial downloads live here so clipfarm never grabs half a file
ARCHIVE = BASE / "downloaded.txt"  # remembers what was downloaded; delete a line (or the file) to re-download
MAX_HEIGHT = 1080                  # 1080 is plenty for 9:16 shorts
ALLOW_PLAYLISTS = False            # True = a playlist link downloads the whole playlist
# ------------------------------------------

URL_RE = re.compile(r"https?://[^\s<>\"']+")


def find_ytdlp():
    exe = shutil.which("yt-dlp")
    if exe:
        return [exe]
    try:
        import yt_dlp  # noqa: F401
        return [sys.executable, "-m", "yt_dlp"]
    except ImportError:
        return None


def read_links():
    if len(sys.argv) > 1:
        text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
        found = URL_RE.findall(text)
    else:
        print("Paste your YouTube links (one per line, or several at once).")
        print("Press Enter on an empty line when you're done.\n")
        found = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if not line.strip():
                if found:
                    break
                continue
            found += URL_RE.findall(line)
    seen, links = set(), []
    for u in found:                        # remove duplicates, keep order
        if u not in seen:
            seen.add(u)
            links.append(u)
    return links


def download(ytdlp, url):
    cmd = ytdlp + [
        "-P", str(IN_DIR),
        "-P", f"temp:{TMP_DIR}",
        # <=1080p, prefer H.264 (decodes faster and more reliably than AV1), mp4 container
        "-S", f"res:{MAX_HEIGHT},vcodec:h264,ext:mp4:m4a",
        "--merge-output-format", "mp4",
        "--download-archive", str(ARCHIVE),
        "-o", "%(title).80B [%(id)s].%(ext)s",
        "-N", "4",
        "--newline",
        "--yes-playlist" if ALLOW_PLAYLISTS else "--no-playlist",
        url,
    ]
    return subprocess.run(cmd).returncode == 0


def main():
    ytdlp = find_ytdlp()
    if not ytdlp:
        print("yt-dlp not found. Install it with:  winget install yt-dlp")
        print("Then close and reopen PowerShell and run this again.")
        return
    IN_DIR.mkdir(exist_ok=True)
    TMP_DIR.mkdir(exist_ok=True)

    links = read_links()
    if not links:
        print("No links found.")
        return
    print(f"\n{len(links)} link(s) to download.\n")

    ok, failed = 0, []
    for i, url in enumerate(links, 1):
        print(f"=== [{i}/{len(links)}] {url}")
        if download(ytdlp, url):
            ok += 1
        else:
            failed.append(url)
            print("    FAILED (continuing with the next one)")

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print(f"\nDone: {ok} downloaded or already had, {len(failed)} failed.")
    if failed:
        print("Failed links (try again, or update yt-dlp with `yt-dlp -U`):")
        for u in failed:
            print("  " + u)


if __name__ == "__main__":
    main()
