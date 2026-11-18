# Clipfarm

Clipfarm turns long videos into captioned vertical clips. Put videos in `input`; the clips are saved in `output`, and processed videos are moved to `done`.

Output videos use H.264 `yuv420p` with AAC audio for broad media-player and YouTube compatibility.

Audio transcription runs in overlapping five-minute chunks to keep memory use bounded on long videos.

## Requirements

- Python 3.12
- FFmpeg and ffprobe available on your PATH
- An NVIDIA GPU for the default CUDA setup

## Install

In PowerShell, from this folder:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install faster-whisper opencv-python nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
```

## Run

Put video files in `input`, then start Clipfarm:

```powershell
python clipfarm.py
```

To download videos into `input` instead, install yt-dlp and run the downloader:

```powershell
python -m pip install yt-dlp
python getvideos.py
```

## Local clip titles

Clipfarm uses Ollama locally to generate an advertiser-friendly title from each clip's transcript and uses it as the MP4 filename. Titles are instructed to avoid profanity stronger than "damn", slurs, and sensationalized sensitive-topic wording. Titles must be no more than 45 characters and pass the known-profanity check; otherwise, Clipfarm retries once and then uses a safe fallback title.

Candidates must be 25–40 seconds and contain 65–110 transcript words. Ollama rates whether a candidate opens with a strong claim, number, or question and ends with a clear payoff; the output is still based on transcript and duration, not a human review of the video. Captions refresh at least every 2.5 seconds during speech.

Ollama rates transcript candidates from the perspective of an average Shorts viewer on a 1–10 scale. Ratings of 7 or higher count as good; it keeps checking candidates until it finds five or runs out. If fewer than five pass, only the passing clips are made; if none pass, the two highest-rated clips are made. Ratings are based on transcript and duration, not direct video viewing.

Install Ollama, then download the small default model:

In PowerShell, install Ollama with its official installer:

```powershell
irm https://ollama.com/install.ps1 | iex
```

Close and reopen PowerShell so the updated PATH is loaded, then run:

```powershell
ollama --version
ollama pull qwen2.5:3b
```

Keep Ollama running while Clipfarm processes videos. Set `OLLAMA_MODEL` or `OLLAMA_URL` to use a different model or local Ollama endpoint.

## Automatic clip edits

Ollama creates a bold, high-contrast 3–7 word hook for the top third of each clip, shown for the first two seconds. Clipfarm also adds a brief zoom pulse about every 2.8 seconds and blends the ending into the opening frame and audio for a loop.

When Ollama recommends background music, Clipfarm randomly chooses a track from `audio` and mixes it very quietly under the source audio. If Ollama does not recommend music, or the folder has no supported audio tracks, the source audio is left alone. The folder includes three royalty-free tracks from [Mixkit](https://mixkit.co/free-stock-music/), licensed for use in videos under the [Mixkit Free License](https://mixkit.co/license/#musicFree); MP3 files in `audio` are excluded from Git.

For reaction videos, Ollama can select a video-focused layout when the transcript clearly indicates a reaction and the visual analysis finds a facecam. The source video takes the larger panel, with the facecam below it.
