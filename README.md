# Clipfarm

Clipfarm turns long videos into captioned vertical clips. Put videos in `input`; the clips are saved in `output`, and processed videos are moved to `done`.

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

## Optional titles

Clipfarm can use the Anthropic API to rank clips and suggest titles. Set the `ANTHROPIC_API_KEY` environment variable to enable this; otherwise, it runs without automatic titles.
