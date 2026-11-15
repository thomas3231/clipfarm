"""
Local clip farm v4
Drop long videos in ./input -> get vertical 9:16 captioned clips in ./output.

What it does per video:
  1. Transcribes (faster-whisper, word-level timestamps)
  2. Finds candidate moments from the transcript (hooks, questions, energy, chat-talk)
  3. Looks at video frames of the best candidates: finds faces, motion, and figures out
     what is going on (gaming with facecam / hanging out with friends / talking to camera / screen only)
  4. Picks a layout per clip:
        gaming + facecam -> facecam on top, gameplay below
        you or friends   -> 9:16 crop that follows the face(s)
        screen only/wide -> full frame over a blurred background
  5. Burns short 2-line captions that change every ~1-2 seconds
  6. (optional) asks Claude to rank the moments and write titles (needs ANTHROPIC_API_KEY)

Install: pip install faster-whisper opencv-python nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
Needs:   Python 3.12, ffmpeg + ffprobe on PATH
Run:     venv\\Scripts\\python.exe clipfarm.py
"""
import json
import os
import re
import shutil
import site
import subprocess
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ---- make the pip-installed NVIDIA DLLs visible to CTranslate2 (Windows) ----
for _sp in site.getsitepackages():
    for _sub in ("cublas", "cudnn"):
        _d = os.path.join(_sp, "nvidia", _sub, "bin")
        if os.path.isdir(_d):
            os.add_dll_directory(_d)
            os.environ["PATH"] = _d + os.pathsep + os.environ["PATH"]

import cv2
import numpy as np
from faster_whisper import WhisperModel

# ================= settings =================
BASE = Path(__file__).parent
IN_DIR, OUT_DIR, DONE_DIR = BASE / "input", BASE / "output", BASE / "done"
CACHE_DIR, TMP_DIR, MODEL_DIR = BASE / "cache", BASE / "tmp", BASE / "models"

CLIPS_PER_VIDEO = 5
MIN_LEN, MAX_LEN = 25, 55            # clip length in seconds
N_VISUAL = 12                        # how many top transcript moments get a visual check

MODEL_SIZE = "medium"                # tiny / base / small / medium / large-v3
DEVICE = "cuda"                      # "cuda" for the RTX 3070, "cpu" to force CPU
COMPUTE = "float16"                  # float16 on cuda, int8 on cpu
EXTS = {".mp4", ".mkv", ".mov", ".webm"}

# --- captions ---
CAPS = True                          # ALL CAPS captions
CAP_MAX_LINE = 16                    # max characters per line (2 lines max)
CAP_MAX_SEC = 1.8                    # max seconds a caption stays up
CAP_FONT = "Arial"
CAP_FONT_SIZE = 74                   # pixels on the 1080x1920 output
CAP_MARGIN_V = 420                   # distance from the bottom edge in pixels
CAP_OUTLINE = 5                      # black outline thickness in pixels

# --- reframing / analysis ---
SAMPLE_FPS = 2                       # frames per second looked at for faces/motion
ANALYZE_W = 640                      # analysis frame width
MIN_FACE_FRAC = 0.02                 # ignore faces smaller than this fraction of frame width
DEADZONE = 0.04                      # camera only moves if the face drifts > 4% of frame width
FACECAM_MAX_W = 0.14                 # facecam faces are smaller than 14% of frame width

# --- optional LLM ranking (set ANTHROPIC_API_KEY in your terminal to enable) ---
LLM_MODEL = "claude-sonnet-5-5"
USE_LLM = bool(os.environ.get("ANTHROPIC_API_KEY"))

FFMPEG_TRIES = 3                     # retries per ffmpeg call if it crashes

HOOK_WORDS = {
    "secret", "never", "always", "mistake", "truth", "crazy", "insane", "actually",
    "why", "how", "biggest", "worst", "best", "problem", "learned", "money",
    "nobody", "everyone", "important", "shocking", "wrong", "stop", "listen",
    "bro", "dude", "wtf", "bruh", "no", "what", "oh", "yo",
}
CHAT_WORDS = {
    "chat", "chatters", "donation", "donated", "subs", "sub", "gifted", "raid",
    "raiding", "viewers", "mods", "mod", "stream",
}
YUNET_URLS = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
)
# ============================================

for d in (IN_DIR, OUT_DIR, DONE_DIR, CACHE_DIR, MODEL_DIR):
    d.mkdir(exist_ok=True)

model = None
using_device = None
detector = None


# ---------------------------------------------------------------- helpers
def even(v) -> int:
    return int(v) // 2 * 2


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def safe_name(stem: str) -> str:
    """Strip FAILED_ prefix, YouTube IDs, brackets and special characters."""
    stem = re.sub(r"^FAILED_", "", stem)
    stem = re.sub(r"\[[A-Za-z0-9_-]{11}\]", "", stem)
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")
    return stem[:60] or "video"


def run_ffmpeg(cmd, cwd=None, tries=FFMPEG_TRIES):
    """Run ffmpeg with retries (this ffmpeg build crashes randomly sometimes)."""
    err = ""
    for attempt in range(1, tries + 1):
        r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if r.returncode == 0:
            return True, ""
        err = r.stderr.decode(errors="ignore").strip()[-200:]
        print(f"    ffmpeg failed (exit {r.returncode}) attempt {attempt}/{tries} {err}")
        time.sleep(1)
    return False, err


def probe(video: Path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True,
    )
    w, h = r.stdout.strip().split(",")[:2]
    return int(w), int(h)


# ---------------------------------------------------------------- whisper
def load_model(device, compute):
    global model, using_device
    print(f"Loading model '{MODEL_SIZE}' on {device} ...")
    model = WhisperModel(MODEL_SIZE, device=device, compute_type=compute)
    using_device = device


def ensure_model():
    if model is not None:
        return
    try:
        load_model(DEVICE, COMPUTE)
    except Exception as e:
        print(f"Could not start on {DEVICE} ({e}). Falling back to CPU.")
        load_model("cpu", "int8")


def load_audio(path):
    """Decode audio with ffmpeg (avoids the PyAV version problem)."""
    cmd = ["ffmpeg", "-nostdin", "-i", str(path),
           "-f", "f32le", "-ac", "1", "-ar", "16000", "-v", "error", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32)


def _run_whisper(audio):
    seg_iter, _ = model.transcribe(audio, vad_filter=True, word_timestamps=True)
    segs = []
    for s in seg_iter:
        words = [{"start": w.start, "end": w.end, "word": w.word} for w in (s.words or [])]
        segs.append({"start": s.start, "end": s.end, "text": s.text, "words": words})
    return segs


def transcribe(video: Path):
    ensure_model()
    audio = load_audio(video)
    try:
        return _run_whisper(audio)
    except Exception as e:
        msg = str(e).lower()
        if using_device == "cuda" and any(k in msg for k in ("cublas", "cudnn", "cuda", "dll")):
            print(f"GPU failed ({e}). Switching to CPU for the rest of this session.")
            load_model("cpu", "int8")
            return _run_whisper(audio)
        raise


def get_segments(video: Path, base: str):
    cache = CACHE_DIR / f"{base}.v2.json"       # v2 = includes word timestamps
    if cache.exists():
        print("Using saved transcript (skipping transcription).")
        return json.loads(cache.read_text(encoding="utf-8"))
    t0 = time.time()
    print("Transcribing ...")
    segs = transcribe(video)
    print(f"Transcribed in {(time.time() - t0) / 60:.1f} min ({len(segs)} segments)")
    cache.write_text(json.dumps(segs), encoding="utf-8")
    return segs


# ---------------------------------------------------------------- transcript scoring
def score_text(segs) -> float:
    text = " ".join(s["text"] for s in segs).lower()
    words = re.findall(r"[a-zA-Z']+", text)
    if not words:
        return 0.0
    dur = segs[-1]["end"] - segs[0]["start"]
    hooks = sum(w in HOOK_WORDS for w in words)
    chat = sum(w in CHAT_WORDS for w in words)
    punct = text.count("?") + text.count("!")
    density = len(words) / dur                 # fast talking = energy
    return hooks * 1.5 + chat * 1.0 + punct * 1.5 + density


def chat_hits(segs) -> int:
    words = re.findall(r"[a-zA-Z']+", " ".join(s["text"] for s in segs).lower())
    return sum(w in CHAT_WORDS for w in words)


def pick_candidates(segs, n):
    cands = []
    for i in range(len(segs)):
        chunk = []
        for j in range(i, len(segs)):
            chunk.append(segs[j])
            dur = chunk[-1]["end"] - chunk[0]["start"]
            if dur > MAX_LEN:
                break
            if dur >= MIN_LEN:
                cands.append((score_text(chunk), list(chunk)))
                break
    cands.sort(key=lambda c: c[0], reverse=True)
    chosen = []
    for sc, chunk in cands:
        a, b = chunk[0]["start"], chunk[-1]["end"]
        if all(b <= c["chunk"][0]["start"] or a >= c["chunk"][-1]["end"] for c in chosen):
            chosen.append({"chunk": chunk, "text_score": sc})
        if len(chosen) >= n:
            break
    return chosen


# ---------------------------------------------------------------- faces
class FaceDetector:
    """YuNet (accurate, auto-downloads ~230 KB model) with Haar cascade fallback."""

    def __init__(self):
        self.yunet = None
        self.haar = None
        path = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
        if not path.exists():
            print("Downloading face detection model (one time) ...")
            for url in YUNET_URLS:
                try:
                    tmp = str(path) + ".part"
                    urllib.request.urlretrieve(url, tmp)
                    if os.path.getsize(tmp) < 100_000:      # tiny file = not the real model
                        raise ValueError("downloaded file is too small")
                    os.replace(tmp, path)
                    break
                except Exception as e:
                    print(f"  download failed ({e})")
                    if os.path.exists(str(path) + ".part"):
                        os.remove(str(path) + ".part")
        if path.exists():
            try:
                self.yunet = cv2.FaceDetectorYN.create(str(path), "", (320, 320), 0.7, 0.3, 5000)
            except Exception as e:
                print(f"YuNet failed to load ({e}).")
        if self.yunet is None:
            if hasattr(cv2, "CascadeClassifier"):
                self.haar = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            else:
                raise RuntimeError(
                    "No face detector available. Download this file and save it as "
                    f"{path}\n{YUNET_URLS[0]}")

    def detect(self, img):
        h, w = img.shape[:2]
        if self.yunet is not None:
            self.yunet.setInputSize((w, h))
            _, faces = self.yunet.detect(img)
            if faces is None:
                return []
            return [(float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[14])) for f in faces]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found = self.haar.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        return [(float(x), float(y), float(fw), float(fh), 1.0) for (x, y, fw, fh) in found]


def extract_frames(video: Path, start: float, dur: float):
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    TMP_DIR.mkdir(exist_ok=True)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
           "-ss", f"{start:.2f}", "-t", f"{dur:.2f}", "-i", str(video.resolve()),
           "-vf", f"fps={SAMPLE_FPS},scale={ANALYZE_W}:-2", "-q:v", "4",
           str(TMP_DIR / "%04d.jpg")]
    ok, _ = run_ffmpeg(cmd)
    return ok


def analyze_frames(video: Path, sw: int, sh: int, start: float, end: float):
    """Returns (frames, diffs). frames = [{'t':sec,'faces':[(x,y,w,h,score) in source px]}]."""
    frames, diffs, prev = [], [], None
    if not extract_frames(video, start, end - start):
        return frames, diffs
    for i, f in enumerate(sorted(TMP_DIR.glob("*.jpg"))):
        img = cv2.imread(str(f))
        if img is None:
            continue
        scale = sw / img.shape[1]
        faces = []
        for (x, y, w, h, s) in detector.detect(img):
            if w * scale >= MIN_FACE_FRAC * sw:
                faces.append((x * scale, y * scale, w * scale, h * scale, s))
        frames.append({"t": i / SAMPLE_FPS, "faces": faces})
        g = cv2.cvtColor(cv2.resize(img, (160, 90)), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            diffs.append(float(np.mean(cv2.absdiff(g, prev))))
        prev = g
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    return frames, diffs


def decide(frames, diffs, sw, sh):
    """Work out what is happening in the clip and which layout to use."""
    info = {"activity": "unknown", "layout": "center", "face_ratio": 0.0,
            "multi_ratio": 0.0, "motion": float(np.mean(diffs)) if diffs else 0.0, "cam": None}
    n = len(frames)
    if n == 0:
        return info
    with_faces = [f for f in frames if f["faces"]]
    face_ratio = len(with_faces) / n
    multi = sum(1 for f in frames if len(f["faces"]) >= 2) / n
    info["face_ratio"], info["multi_ratio"] = face_ratio, multi
    cw = even(sh * 9 / 16)

    if face_ratio < 0.2:
        info.update(activity="screen only (no face)", layout="fit")
        return info

    bigs = [max(f["faces"], key=lambda b: b[2] * b[3]) for f in with_faces]
    medw = float(np.median([b[2] for b in bigs])) / sw
    cxs = np.array([(b[0] + b[2] / 2) / sw for b in bigs])
    cys = np.array([(b[1] + b[3] / 2) / sh for b in bigs])
    cx, cy = float(np.median(cxs)), float(np.median(cys))
    stable = float(np.std(cxs)) < 0.03 and float(np.std(cys)) < 0.03
    cornerish = cx < 0.3 or cx > 0.7 or cy < 0.25 or cy > 0.75

    if medw < FACECAM_MAX_W and stable and cornerish and multi < 0.4:
        cam_h = float(np.median([b[3] for b in bigs]))
        info.update(activity="gaming (facecam)", layout="stack",
                    cam=(cx * sw, cy * sh, medw * sw, cam_h))
    elif multi >= 0.4:
        spans = []
        for f in frames:
            if len(f["faces"]) >= 2:
                spans.append(max(b[0] + b[2] for b in f["faces"]) - min(b[0] for b in f["faces"]))
        wide = float(np.median(spans)) > cw * 0.85
        info.update(activity="hanging out (2+ people)", layout="fit" if wide else "follow_group")
    elif medw >= FACECAM_MAX_W:
        info.update(activity="talking to camera", layout="follow")
    else:
        info.update(activity="person on screen", layout="follow")
    return info


def visual_bonus(info) -> float:
    bonus = 3.0 * info["face_ratio"] + min(info["motion"], 20.0) / 5.0
    if info["multi_ratio"] >= 0.4:
        bonus += 3.0
    if info["face_ratio"] < 0.2 and info["motion"] < 2.0:
        bonus -= 5.0                      # dead screen / menu / idle
    return bonus


# ---------------------------------------------------------------- reframing
def follow_keys(frames, sw, cw, dur, group):
    """Smoothed camera path as [(time, crop_left_x)] with a deadzone so it stays steady."""
    centers, last = [], None
    for fr in frames:
        fs = fr["faces"]
        if fs:
            if group and len(fs) > 1:
                last = (min(b[0] for b in fs) + max(b[0] + b[2] for b in fs)) / 2
            else:
                big = max(fs, key=lambda b: b[2] * b[3])
                last = big[0] + big[2] / 2
        centers.append(last)
    first = next((c for c in centers if c is not None), sw / 2)
    centers = [first if c is None else c for c in centers]
    if not centers:
        centers = [sw / 2]
    arr = np.array(centers, dtype=float)
    k = 5
    padded = np.pad(arr, (k // 2, k // 2), mode="edge")
    smooth = np.convolve(padded, np.ones(k) / k, mode="valid")
    xs = [clamp(c - cw / 2, 0, sw - cw) for c in smooth]

    keys = [(0.0, xs[0])]
    for i, x in enumerate(xs):
        t = frames[i]["t"] if frames else 0.0
        if abs(x - keys[-1][1]) > DEADZONE * sw:
            if t - 0.6 > keys[-1][0]:
                keys.append((t - 0.6, keys[-1][1]))     # hold, then pan over 0.6s
            keys.append((t, x))
    if dur > keys[-1][0]:
        keys.append((dur, keys[-1][1]))
    return keys[:60]


def x_expr(keys) -> str:
    """ffmpeg expression: linear interpolation between camera keyframes over time t."""
    if len(keys) < 2 or all(abs(k[1] - keys[0][1]) < 0.5 for k in keys):
        return str(int(keys[0][1]))
    expr = f"{keys[-1][1]:.1f}"
    for i in range(len(keys) - 2, -1, -1):
        t0, x0 = keys[i]
        t1, x1 = keys[i + 1]
        if t1 <= t0:
            continue
        seg = f"{x0:.1f}+({x1 - x0:.1f})*(t-{t0:.2f})/{t1 - t0:.2f}"
        expr = f"if(lt(t,{t1:.2f}),{seg},{expr})"
    return expr


def build_graph(plan, sw, sh, dur):
    """ffmpeg filter graph (ends at label [v0]) for the chosen layout."""
    layout, frames = plan["layout"], plan["frames"]
    cw = even(sh * 9 / 16)
    if layout in ("follow", "follow_group", "center"):
        if layout == "center":
            keys = [(0.0, (sw - cw) / 2), (dur, (sw - cw) / 2)]
        else:
            keys = follow_keys(frames, sw, cw, dur, group=(layout == "follow_group"))
        return f"[0:v]crop={cw}:{sh}:'{x_expr(keys)}':0,scale=1080:1920,setsar=1[v0]"

    if layout == "stack" and plan.get("cam"):
        cx, cy, fw, fh = plan["cam"]
        rw = max(fw * 3.2, sw * 0.16)
        rh = rw / 1.5
        if rh > sh:
            rh, rw = sh, sh * 1.5
        rw, rh = even(rw), even(rh)
        fx = even(clamp(cx - rw / 2, 0, sw - rw))
        fy = even(clamp(cy - rh / 2, 0, sh - rh))
        gw = even(sh * 0.9)
        gx = even((sw - gw) / 2)
        return (
            "[0:v]split=2[f][g];"
            f"[f]crop={rw}:{rh}:{fx}:{fy},scale=1080:720[top];"
            f"[g]crop={gw}:{sh}:{gx}:0,scale=1080:1200[bot];"
            "[top][bot]vstack=inputs=2,setsar=1[v0]"
        )

    # "fit": whole frame over a blurred copy of itself
    return (
        "[0:v]split=2[a][b];"
        "[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=25:5[bg];"
        "[b]scale=1080:-2[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[v0]"
    )


# ---------------------------------------------------------------- captions
def ass_time(t: float) -> str:
    t = max(t, 0)
    cs = int(round(t * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    sec, cs = divmod(rem, 100)
    return f"{h}:{m:02}:{sec:02}.{cs:02}"


def wrap_two_lines(text: str) -> str:
    if len(text) <= CAP_MAX_LINE or " " not in text:
        return text
    words = text.split(" ")
    best, best_cost = None, 10 ** 9
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        cost = max(len(a), len(b))
        if cost < best_cost:
            best, best_cost = f"{a}\n{b}", cost
    return best


def build_cards(chunk, start, end):
    """Short captions: <= 2 lines of <= CAP_MAX_LINE chars, each shown <= CAP_MAX_SEC."""
    words = [w for seg in chunk for w in seg.get("words", [])]
    if not words:      # fallback for transcripts without word timestamps
        words = [{"start": s["start"], "end": s["end"], "word": s["text"]} for s in chunk]
    ws = [w for w in words if w["end"] > start and w["start"] < end]

    cards, cur = [], []

    def flush():
        if not cur:
            return
        text = " ".join(x["t"] for x in cur)
        text = text.upper() if CAPS else text
        cards.append({"start": max(cur[0]["start"] - start, 0.0),
                      "end": min(cur[-1]["end"] - start, end - start),
                      "text": wrap_two_lines(text)})
        cur.clear()

    for w in ws:
        token = w["word"].strip()
        if not token:
            continue
        if cur:
            joined = " ".join(x["t"] for x in cur) + " " + token
            gap = w["start"] - cur[-1]["end"]
            dur = w["end"] - cur[0]["start"]
            if len(joined) > CAP_MAX_LINE * 2 or gap > 0.5 or dur > CAP_MAX_SEC or cur[-1]["t"][-1] in ".?!":
                flush()
        cur.append({"t": token, "start": w["start"], "end": w["end"]})
    flush()

    for i, c in enumerate(cards):               # hold a moment, never overlap the next card
        c["end"] = c["end"] + 0.15
        if i + 1 < len(cards):
            c["end"] = min(c["end"], cards[i + 1]["start"])
    return [c for c in cards if c["end"] - c["start"] > 0.1]


def write_ass(cards, path: Path):
    """ASS file with a real 1080x1920 canvas, so sizes/margins below are in pixels."""
    head = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\nWrapStyle: 2\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{CAP_FONT},{CAP_FONT_SIZE},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,"
        f"1,0,0,0,100,100,0,0,1,{CAP_OUTLINE},0,2,60,60,{CAP_MARGIN_V},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = [
        f"Dialogue: 0,{ass_time(c['start'])},{ass_time(c['end'])},Default,,0,0,0,,"
        f"{c['text'].replace(chr(10), chr(92) + 'N')}"
        for c in cards
    ]
    path.write_text(head + "\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- cutting
def cut_clip(video: Path, sw: int, sh: int, chunk, plan, name: str) -> bool:
    start, end = chunk[0]["start"], chunk[-1]["end"]
    dur = end - start
    sub_path, out_path = OUT_DIR / f"{name}.ass", OUT_DIR / f"{name}.mp4"
    write_ass(build_cards(chunk, start, end), sub_path)

    graph = build_graph(plan, sw, sh, dur)
    graph_caps = f"{graph};[v0]subtitles={name}.ass[v]"
    graph_plain = f"{graph};[v0]null[v]"

    ok = False
    for attempt in range(1, FFMPEG_TRIES + 1):
        captions = attempt < FFMPEG_TRIES or FFMPEG_TRIES == 1
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
               "-ss", f"{start:.2f}", "-i", str(video.resolve()), "-t", f"{dur:.2f}",
               "-filter_complex", graph_caps if captions else graph_plain,
               "-map", "[v]", "-map", "0:a:0?"]
        # single-threaded filters: this ffmpeg build crashes (0xC0000005) on multi-threaded
        # filter graphs, which is what made attempt 1 fail and attempt 2 work
        cmd += ["-filter_threads", "1", "-filter_complex_threads", "1", "-threads", "4"]
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-c:a", "aac", "-b:a", "160k", f"{name}.mp4"]
        # cwd=OUT_DIR so the subtitles filter gets a simple relative path (avoids Windows path escaping)
        r = subprocess.run(cmd, cwd=OUT_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if r.returncode == 0:
            ok = True
            if not captions:
                print("    note: this clip was made WITHOUT captions (ffmpeg kept crashing with them)")
            break
        err = r.stderr.decode(errors="ignore").strip()[-300:]
        print(f"    ffmpeg failed (exit {r.returncode}) attempt {attempt}/{FFMPEG_TRIES} {err}")
        out_path.unlink(missing_ok=True)
        time.sleep(1)

    sub_path.unlink(missing_ok=True)
    return ok


# ---------------------------------------------------------------- optional LLM ranking
def llm_rank(items):
    """items: [{id, activity, text}] -> {id: {score, title}}. Returns {} if disabled or on any error."""
    if not USE_LLM or not items:
        return {}
    prompt = (
        "You are an expert editor of high-retention vertical clips (TikTok/Shorts/Reels) for a creator who "
        "games, hangs out with friends and talks to chat. Below are candidate moments from one long video. "
        "Each has what is visually happening and the transcript. Rate each 1-10 using these rules:\n"
        "1. HOOK: it must open in the middle of the action, a punchline or a controversial statement. "
        "Penalize hard if it starts with a slow intro, greetings or filler ('umm', 'so yeah').\n"
        "2. COMPLETENESS: clear beginning and a payoff; penalize being cut off mid-thought "
        "(unless it is a dramatic cliffhanger).\n"
        "3. EMOTION: reward laughter, surprise, realization, chaos or hot takes.\n"
        "4. VIBE: gaming -> rage, clutch plays, funny bugs, banter with teammates. "
        "Hanging out -> inside jokes, roasting each other, unexpected interruptions. "
        "Talking to chat -> direct addresses, reading unhinged messages, hot takes.\n"
        "5. TITLE: punchy, curiosity-inducing, all lower-case, max 6 words, must NOT spoil the punchline.\n"
        'Reply with ONLY a JSON array like [{"id":0,"score":7,"title":"he really said that"}].\n\n'
        + "\n\n".join(f"ID {i['id']} | visual: {i['activity']}\n{i['text'][:700]}" for i in items)
    )
    try:
        body = json.dumps({"model": LLM_MODEL, "max_tokens": 1500,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=body,
            headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                     "anthropic-version": "2023-06-01", "content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as r:
            data = json.loads(r.read())
        text = "".join(b.get("text", "") for b in data["content"])
        arr = json.loads(text[text.index("["): text.rindex("]") + 1])
        out = {}
        for x in arr:
            title = " ".join(str(x.get("title", "")).lower().split()[:6])    # lower-case, max 6 words
            out[int(x["id"])] = {"score": float(x["score"]), "title": title}
        return out
    except Exception as e:
        print(f"  (LLM ranking skipped: {e})")
        return {}


# ---------------------------------------------------------------- pipeline
def process(video: Path):
    global detector
    t0 = time.time()
    print(f"\n{video.name}")
    base = safe_name(video.stem)
    segs = get_segments(video, base)
    if not segs:
        print("No speech found, skipping.")
        return
    if detector is None:
        detector = FaceDetector()
    sw, sh = probe(video)

    cands = pick_candidates(segs, N_VISUAL)
    print(f"Looking at {len(cands)} candidate moments ...")
    for i, c in enumerate(cands):
        chunk = c["chunk"]
        start, end = chunk[0]["start"], chunk[-1]["end"]
        frames, diffs = analyze_frames(video, sw, sh, start, end)
        info = decide(frames, diffs, sw, sh)
        c["plan"] = {**info, "frames": frames}
        c["info"] = info
        c["chat"] = chat_hits(chunk) >= 2
        if c["chat"]:
            info["activity"] += " + talking to chat"
        c["score"] = c["text_score"] + visual_bonus(info)
        c["id"] = i
        c["text"] = " ".join(s["text"].strip() for s in chunk)
        print(f"  #{i} {start:.0f}s-{end:.0f}s  {info['activity']:<32} layout={info['layout']:<13} "
              f"faces={info['face_ratio']:.0%} motion={info['motion']:.1f}")

    ranks = llm_rank([{"id": c["id"], "activity": c["info"]["activity"], "text": c["text"]} for c in cands])
    for c in cands:
        r = ranks.get(c["id"])
        c["title"] = r["title"] if r else ""
        if r:
            c["score"] += 2.5 * r["score"]

    best = sorted(cands, key=lambda c: c["score"], reverse=True)[:CLIPS_PER_VIDEO]
    made, failed = 0, 0
    for n, c in enumerate(best, 1):
        chunk, name = c["chunk"], f"{base}_clip{n}"
        print(f"  cutting {name} ({chunk[0]['start']:.0f}s-{chunk[-1]['end']:.0f}s) "
              f"[{c['info']['activity']} -> {c['info']['layout']}]")
        if cut_clip(video, sw, sh, chunk, c["plan"], name):
            made += 1
            (OUT_DIR / f"{name}.txt").write_text(
                f"Title idea: {c['title'] or '(set ANTHROPIC_API_KEY for auto titles)'}\n"
                f"Activity: {c['info']['activity']}\nLayout: {c['info']['layout']}\n"
                f"Score: {c['score']:.1f}\n\nTranscript:\n{c['text']}\n", encoding="utf-8")
        else:
            failed += 1
            print(f"    gave up on {name}")
    print(f"Done: {made} clips made, {failed} failed, {(time.time() - t0) / 60:.1f} min total")


def main():
    ensure_model()
    print(f"LLM ranking: {'ON' if USE_LLM else 'off (set ANTHROPIC_API_KEY to enable)'}")
    print(f"Watching {IN_DIR} ... (Ctrl+C to stop)")
    while True:
        for f in sorted(IN_DIR.iterdir()):
            if f.suffix.lower() in EXTS:
                try:
                    process(f)
                    shutil.move(str(f), DONE_DIR / f.name)
                except Exception as e:
                    print(f"Failed on {f.name}: {e}")
                    shutil.move(str(f), DONE_DIR / f"FAILED_{f.name}")
        time.sleep(10)


if __name__ == "__main__":
    main()
