#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
영상 색감 일괄 보정 도구 (로컬 웹앱)

  1) 영상을 여러 개 올리면 각 영상의 첫 화면(또는 고른 장면)이 미리보기로 나와요.
  2) 노출 / 화이트밸런스 / 대비 / 채도 슬라이더(또는 Auto)로 미리보기 색감을 정해요.
  3) [색감 보정]을 누르면 컷(장면)마다 현재 색을 측정해서, 미리보기와 같은 느낌이 되도록
     컷별 보정량을 따로 계산해 영상 전체에 적용해요. (원본은 건드리지 않고 새 mp4로 저장)

실행:  python app.py   ->   http://127.0.0.1:5000
필요:  Python 3.9+, ffmpeg / ffprobe, pip install -r requirements.txt
"""
from __future__ import annotations

import base64
import collections
import io
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, request, send_file, send_from_directory
from PIL import Image

# ----------------------------------------------------------------------------
# 설정
# ----------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("WORK_DIR", BASE / "work")).resolve()
UPLOADS = WORK / "uploads"
OUTPUTS = WORK / "outputs"
JOBS_DIR = WORK / "jobs"
for _d in (UPLOADS, OUTPUTS, JOBS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE", "ffprobe")

PREVIEW_W = 900          # 미리보기 이미지 최대 가로(px)
PROXY_W = 1280           # 재생용 변환본 최대 가로(px)
MIN_CLIP = 0.6           # 이보다 짧은 컷은 앞 컷에 합침(초)
GRID = 33                # 3D LUT 크기
SENS = {"low": 0.45, "normal": 0.30, "high": 0.20}   # 컷 감지 민감도(낮을수록 민감)
QUALITY = {"draft": ("veryfast", 23), "balanced": ("fast", 19), "high": ("slow", 16)}
AUDIO_COPY_OK = {"aac", "mp3", "ac3", "eac3"}

app = Flask(__name__)
FILES: dict[str, dict] = {}
JOBS: dict[str, dict] = {}
FRAME_CACHE: dict[str, tuple] = {}
LOCK = threading.Lock()
FRAME_LOCK = threading.Lock()   # 미리보기 장면 추출을 한 번에 하나씩
PROXY_LOCK = threading.Lock()   # 재생용 변환을 한 번에 하나씩


class Cancelled(Exception):
    pass


# ----------------------------------------------------------------------------
# 색감 계산 (미리보기·컷별 보정값 계산·LUT 생성이 모두 이 함수를 공유)
#   p = (exposure, wb, contrast, saturation), 슬라이더 -100~100 이 -1~1 에 대응
# ----------------------------------------------------------------------------
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def sat_factor(s: float) -> float:
    s = max(-1.0, s)
    return 1.0 + s if s <= 0 else 1.0 + 1.8 * s


def grade(rgb: np.ndarray, p) -> np.ndarray:
    """rgb: (..., 3) float32 0~1  ->  같은 shape, 0~1"""
    e, w, c, s = (float(v) for v in p)
    x = np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)

    # 1) 노출: 감마 곡선 (하이라이트가 날아가지 않게)
    if abs(e) > 1e-4:
        x = np.power(x, np.float32(2.0 ** (-0.8 * e)))

    # 2) 화이트밸런스: R/B 게인 (밝기는 유지되게 정규화)
    if abs(w) > 1e-4:
        r, b = 2.0 ** (0.30 * w), 2.0 ** (-0.30 * w)
        norm = LUMA[0] * r + LUMA[1] + LUMA[2] * b
        gains = np.array([r, 1.0, b], dtype=np.float32) / np.float32(norm)
        x = np.clip(x * gains, 0.0, 1.0)

    # 3) 채도: 0 = 흑백 ... 1 = 원본 ... 최대 약 2.8배
    f = sat_factor(s)
    if abs(f - 1.0) > 1e-4:
        y = (x @ LUMA)[..., None]
        x = np.clip(y + (x - y) * np.float32(f), 0.0, 1.0)

    # 4) 대비: 올릴 땐 S자 곡선(밝은 곳이 확 날아가지 않음), 내릴 땐 가운데로 압축
    if c > 1e-4:
        a = 1.8 * c
        x = 0.5 + 0.5 * np.tanh(a * (2.0 * x - 1.0)) / np.float32(math.tanh(a))
    elif c < -1e-4:
        x = 0.5 + (x - 0.5) * np.float32(1.0 + 0.5 * max(c, -1.0))

    return np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)


def stats(px: np.ndarray) -> dict:
    """픽셀 (N,3) -> 밝기 평균 L, 밝기 편차(대비) C, 채도 S, 색온도 치우침 W"""
    px = px.reshape(-1, 3)
    y = px @ LUMA
    chroma = px.max(axis=1) - px.min(axis=1)
    warm = math.log((float(px[:, 0].mean()) + 0.02) / (float(px[:, 2].mean()) + 0.02))
    return {"L": float(y.mean()), "C": float(y.std()), "S": float(chroma.mean()), "W": warm}


IDX = {"e": 0, "w": 1, "c": 2, "s": 3}
STAT_OF = {"e": "L", "w": "W", "c": "C", "s": "S"}
BOUNDS = {"e": (-1.5, 1.5), "w": (-1.5, 1.5), "c": (-1.0, 1.5), "s": (-1.0, 1.5)}


def bisect_inc(f, target, lo, hi, default, n=14):
    flo, fhi = f(lo), f(hi)
    if abs(fhi - flo) < 1e-5:      # 이 값을 바꿔도 통계가 안 변함(예: 흑백일 때 색온도)
        return default
    if target <= flo:
        return lo
    if target >= fhi:
        return hi
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_match(px, target, p_start, enabled, bounds=BOUNDS, rounds=3):
    """px(컷의 픽셀)에 보정을 걸었을 때 통계가 target과 같아지는 파라미터를 찾는다."""
    if px.shape[0] > 24000:
        px = px[:: int(math.ceil(px.shape[0] / 24000))]
    p = list(p_start)
    for _ in range(rounds):
        for key in ("w", "e", "c", "s"):
            if key not in enabled:
                continue
            i, stat = IDX[key], STAT_OF[key]
            lo, hi = bounds[key]

            def f(v, i=i, stat=stat):
                q = list(p)
                q[i] = v
                return stats(grade(px, q))[stat]

            p[i] = bisect_inc(f, target[stat], lo, hi, p[i])
    return p


def auto_params(arr: np.ndarray):
    """미리보기 화면을 보고 무난한 보정값을 추천 (슬라이더 단위 -100~100)"""
    px = arr.reshape(-1, 3)
    if px.shape[0] > 60000:
        px = px[:: int(math.ceil(px.shape[0] / 60000))]
    st = stats(px)
    tgt = {
        "L": st["L"] + 0.65 * (0.46 - st["L"]),
        "C": st["C"] + 0.5 * (0.21 - st["C"]) if st["C"] < 0.21 else st["C"],
        "S": st["S"] if (st["S"] < 0.03 or st["S"] >= 0.30) else st["S"] * 1.15,
        "W": st["W"] * 0.4,
    }
    bounds = {"e": (-0.7, 0.7), "w": (-0.5, 0.5), "c": (-0.5, 0.6), "s": (-0.3, 0.7)}
    p = solve_match(px, tgt, (0, 0, 0, 0), ("e", "w", "c", "s"), bounds=bounds)
    note = ""
    if st["L"] < 0.04:
        note = "이 장면이 거의 검정이라 추천이 정확하지 않아요. 아래 슬라이더로 다른 장면을 골라 보세요."
    return [int(round(v * 100)) for v in p], note


def write_cube(path: Path, p) -> None:
    ax = np.linspace(0.0, 1.0, GRID, dtype=np.float32)
    r, g, b = np.meshgrid(ax, ax, ax, indexing="ij")
    pts = np.stack([r, g, b], axis=-1).transpose(2, 1, 0, 3).reshape(-1, 3)  # R이 가장 빨리 변함
    out = grade(pts, p)
    with open(path, "w", newline="\n") as fh:
        fh.write(f"LUT_3D_SIZE {GRID}\nDOMAIN_MIN 0.0 0.0 0.0\nDOMAIN_MAX 1.0 1.0 1.0\n")
        np.savetxt(fh, out, fmt="%.5f")


def is_identity(p) -> bool:
    return all(abs(v) < 0.004 for v in p)


# ----------------------------------------------------------------------------
# ffmpeg / ffprobe 도우미
# ----------------------------------------------------------------------------
def to_float(v, default=0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def err(msg: str, code: int = 400):
    return jsonify(error=msg), code


def ffmpeg_version() -> str:
    try:
        r = subprocess.run([FFMPEG, "-version"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=20)
        return r.stdout.splitlines()[0] if r.returncode == 0 else ""
    except Exception:
        return ""


def parse_fps(s) -> float:
    try:
        a, _, b = str(s).partition("/")
        return float(a) / float(b) if b else float(a)
    except Exception:
        return 0.0


KNOWN_PRIM = {"bt709", "bt470m", "bt470bg", "smpte170m", "smpte240m", "film", "bt2020", "smpte428", "smpte431", "smpte432"}
KNOWN_TRC = {"bt709", "gamma22", "gamma28", "smpte170m", "smpte240m", "linear", "iec61966-2-1",
             "bt2020-10", "bt2020-12", "smpte2084", "arib-std-b67"}


def probe(path: Path) -> dict:
    r = subprocess.run([FFPROBE, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    if r.returncode != 0:
        raise ValueError("영상 파일을 읽지 못했어요. (지원하지 않거나 손상된 파일)")
    j = json.loads(r.stdout or "{}")
    streams = j.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"
              and not s.get("disposition", {}).get("attached_pic", 0)), None)
    if not v:
        raise ValueError("영상 트랙이 없는 파일이에요.")
    audio = [s for s in streams if s.get("codec_type") == "audio"]

    cs = (v.get("color_space") or "").lower()
    height = int(v.get("height") or 0)
    if cs == "bt709":
        mat = "bt709"
    elif cs in ("smpte170m", "bt470bg"):
        mat = "bt601"
    elif cs.startswith("bt2020"):
        mat = "bt2020"
    else:
        mat = "bt709" if height >= 720 else "bt601"
    rng = "pc" if (v.get("color_range") == "pc" or str(v.get("pix_fmt", "")).startswith("yuvj")) else "tv"
    default_tag = {"bt709": ("bt709", "bt709", "bt709"),
                   "bt601": ("smpte170m", "smpte170m", "smpte170m"),
                   "bt2020": ("bt2020nc", "bt2020", "bt2020-10")}[mat]
    prim = v.get("color_primaries") if v.get("color_primaries") in KNOWN_PRIM else default_tag[1]
    trc = v.get("color_transfer") if v.get("color_transfer") in KNOWN_TRC else default_tag[2]

    dur = to_float(j.get("format", {}).get("duration")) or to_float(v.get("duration"))
    return {
        "duration": dur,
        "start_time": to_float(v.get("start_time")),
        "vw": int(v.get("width") or 0), "vh": height,
        "fps": parse_fps(v.get("avg_frame_rate") or v.get("r_frame_rate")),
        "audio": bool(audio),
        "audio_codec": (audio[0].get("codec_name") if audio else None),
        "mat": mat, "rng": rng,
        "tags": {"space": default_tag[0], "prim": prim, "trc": trc},
        "hdr": trc in ("smpte2084", "arib-std-b67"),
    }


def src_path(meta: dict) -> Path:
    return UPLOADS / meta["id"] / ("source" + meta["ext"])


def frame_path(meta: dict) -> Path:
    return UPLOADS / meta["id"] / "frame.png"


def proxy_path(meta: dict) -> Path:
    return UPLOADS / meta["id"] / "proxy.mp4"


def save_meta(meta: dict) -> None:
    (UPLOADS / meta["id"] / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def extract_frame(meta: dict, t: float) -> np.ndarray:
    """영상의 t초 장면을 미리보기 크기 PNG로 저장하고 float 배열로 돌려줌"""
    out = frame_path(meta)
    tmp = out.with_name("frame_tmp.png")
    vf = (f"scale=w='min({PREVIEW_W},iw)':h=-2:in_color_matrix={meta['mat']}:in_range={meta['rng']}"
          f":flags=bicubic+accurate_rnd+full_chroma_int,format=rgb24")
    for tt in (t, 0.0):
        tmp.unlink(missing_ok=True)
        cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{tt:.3f}", "-i", str(src_path(meta)),
               "-map", "0:v:0", "-frames:v", "1", "-vf", vf, str(tmp)]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
        if tmp.exists():
            os.replace(tmp, out)
            meta["preview_t"] = tt
            break
    else:
        raise ValueError("화면을 가져오지 못했어요: " + (r.stderr or "")[-300:])
    arr = load_frame(meta, force=True)
    meta["fh"], meta["fw"] = arr.shape[0], arr.shape[1]
    meta["mean_luma"] = round(float((arr.reshape(-1, 3) @ LUMA).mean()), 4)
    meta["ver"] = meta.get("ver", 0) + 1
    save_meta(meta)
    return arr


def load_frame(meta: dict, force: bool = False) -> np.ndarray:
    key = meta["id"]
    ver = frame_path(meta).stat().st_mtime_ns
    hit = FRAME_CACHE.get(key)
    if hit and hit[0] == ver and not force:
        return hit[1]
    arr = np.asarray(Image.open(frame_path(meta)).convert("RGB"), dtype=np.float32) / 255.0
    if len(FRAME_CACHE) > 8:
        FRAME_CACHE.pop(next(iter(FRAME_CACHE)))
    FRAME_CACHE[key] = (ver, arr)
    return arr


def jpeg_bytes(arr: np.ndarray, quality: int = 88) -> bytes:
    img = Image.fromarray((np.clip(arr, 0, 1) * 255.0 + 0.5).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def file_public(m: dict) -> dict:
    keys = ("id", "name", "duration", "fw", "fh", "vw", "vh", "fps", "audio", "preview_t", "mean_luma", "ver", "hdr")
    return {k: m.get(k) for k in keys}


def safe_stem(name: str) -> str:
    stem = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", Path(name).stem).strip(" .")
    return stem[:80] or "video"


def parse_params(d: dict):
    def g(k):
        return max(-100.0, min(100.0, to_float((d or {}).get(k), 0.0))) / 100.0
    return [g("exposure"), g("wb"), g("contrast"), g("saturation")]


# ----------------------------------------------------------------------------
# API: 업로드 / 미리보기
# ----------------------------------------------------------------------------
@app.after_request
def no_cache(resp):
    if request.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/")
def index():
    return send_from_directory(BASE, "index.html")


@app.get("/api/files")
def api_files():
    return jsonify(ffmpeg=ffmpeg_version(), files=[file_public(m) for m in FILES.values()])


@app.post("/api/upload")
def api_upload():
    name = request.args.get("name", "video.mp4")
    ext = Path(name).suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,6}", ext):
        ext = ".bin"
    fid = uuid.uuid4().hex[:12]
    d = UPLOADS / fid
    d.mkdir(parents=True)
    meta = {"id": fid, "name": name, "ext": ext}
    try:
        with open(d / ("source" + ext), "wb") as f:
            shutil.copyfileobj(request.stream, f, 1 << 20)
        meta.update(probe(d / ("source" + ext)))
        extract_frame(meta, 0.0)
    except Exception as e:  # noqa: BLE001
        shutil.rmtree(d, ignore_errors=True)
        return err(str(e))
    FILES[fid] = meta
    return jsonify(file_public(meta))


@app.delete("/api/files/<fid>")
def api_delete(fid):
    meta = FILES.pop(fid, None)
    FRAME_CACHE.pop(fid, None)
    if meta:
        shutil.rmtree(UPLOADS / fid, ignore_errors=True)
    return jsonify(ok=True)


@app.post("/api/reset")
def api_reset():
    """새로고침 시 호출: 진행 중인 작업 취소 + 올린 영상/결과물 전부 삭제"""
    for jb in list(JOBS.values()):
        if jb["status"] in ("queued", "running"):
            jb["cancel"].set()
            proc = jb.get("proc")
            if proc:
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
    time.sleep(0.3)  # 작업 스레드가 정리될 시간
    with LOCK:
        FILES.clear()
        FRAME_CACHE.clear()
        JOBS.clear()
    for d in (UPLOADS, OUTPUTS, JOBS_DIR):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
    return jsonify(ok=True)


@app.post("/api/frame")
def api_frame():
    j = request.get_json(force=True, silent=True) or {}
    meta = FILES.get(j.get("id"))
    if not meta:
        return err("영상을 찾을 수 없어요.", 404)
    t = max(0.0, min(to_float(j.get("t")), max(0.0, meta["duration"] - 0.1)))
    try:
        with FRAME_LOCK:
            extract_frame(meta, t + meta["start_time"] if t > 0 else 0.0)
            meta["preview_t"] = t
            save_meta(meta)
    except Exception as e:  # noqa: BLE001
        return err(str(e))
    return jsonify(file_public(meta))


SCRUB_W = 480   # 끄는 동안 보여줄 저해상도 미리보기 가로(px)


@app.post("/api/scrub")
def api_scrub():
    """슬라이더를 끄는 중 쓰는 고속 경로: 파일 저장 없이 키프레임 근처 화면을 바로 돌려줌"""
    j = request.get_json(force=True, silent=True) or {}
    meta = FILES.get(j.get("id"))
    if not meta:
        return err("영상을 찾을 수 없어요.", 404)
    t = max(0.0, min(to_float(j.get("t")), max(0.0, meta["duration"] - 0.1))) + meta["start_time"]
    w = max(2, min(SCRUB_W, meta["fw"]))
    h = max(2, int(round(w * meta["fh"] / meta["fw"])))
    vf = (f"scale={w}:{h}:in_color_matrix={meta['mat']}:in_range={meta['rng']}:flags=bilinear,format=rgb24")
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-noaccurate_seek", "-ss", f"{t:.3f}",
           "-i", str(src_path(meta)), "-map", "0:v:0", "-an", "-sn", "-dn", "-frames:v", "1",
           "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60)
    except Exception as e:  # noqa: BLE001
        return err(str(e))
    if len(r.stdout) < w * h * 3:
        return err("화면을 가져오지 못했어요.")
    arr = np.frombuffer(r.stdout[: w * h * 3], dtype=np.uint8).reshape(h, w, 3).astype(np.float32) / 255.0

    def uri(a):
        return "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes(a, 80)).decode()

    return jsonify(before=uri(arr), after=uri(grade(arr, parse_params(j.get("params")))))


@app.get("/api/original/<fid>")
def api_original(fid):
    meta = FILES.get(fid)
    if not meta:
        return err("영상을 찾을 수 없어요.", 404)
    return Response(jpeg_bytes(load_frame(meta)), mimetype="image/jpeg")


@app.post("/api/preview")
def api_preview():
    j = request.get_json(force=True, silent=True) or {}
    meta = FILES.get(j.get("id"))
    if not meta:
        return err("영상을 찾을 수 없어요.", 404)
    return Response(jpeg_bytes(grade(load_frame(meta), parse_params(j.get("params")))), mimetype="image/jpeg")


@app.post("/api/auto")
def api_auto():
    j = request.get_json(force=True, silent=True) or {}
    meta = FILES.get(j.get("id"))
    if not meta:
        return err("영상을 찾을 수 없어요.", 404)
    vals, note = auto_params(load_frame(meta))
    return jsonify(exposure=vals[0], wb=vals[1], contrast=vals[2], saturation=vals[3], note=note)


# ----------------------------------------------------------------------------
# API: 재생 (브라우저 <video> 가 읽는 주소)
#   /api/video/<id>  : 업로드한 원본을 그대로 (Range 지원 -> 탐색 가능)
#   /api/proxy/<id>  : 브라우저가 못 읽는 형식일 때 쓰는 H.264 변환본 (처음 한 번만 만들고 재사용)
# ----------------------------------------------------------------------------
@app.get("/api/video/<fid>")
def api_video(fid):
    meta = FILES.get(fid)
    if not meta or not src_path(meta).exists():
        return err("영상을 찾을 수 없어요.", 404)
    mt = mimetypes.guess_type(str(src_path(meta)))[0] or "video/mp4"
    if not mt.startswith("video/"):
        mt = "video/mp4"
    return send_file(src_path(meta), mimetype=mt, conditional=True)


@app.get("/api/proxy/<fid>")
def api_proxy(fid):
    meta = FILES.get(fid)
    if not meta or not src_path(meta).exists():
        return err("영상을 찾을 수 없어요.", 404)
    out = proxy_path(meta)
    if not out.exists():
        with PROXY_LOCK:
            if not out.exists():
                tmp = out.with_name("proxy_tmp.mp4")
                tmp.unlink(missing_ok=True)
                vf = (f"scale=w='min({PROXY_W},iw)':h=-2:in_color_matrix={meta['mat']}:in_range={meta['rng']}"
                      f":out_color_matrix=bt709:out_range=tv:flags=bicubic+accurate_rnd+full_chroma_int,"
                      "format=yuv420p")
                cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src_path(meta)),
                       "-map", "0:v:0", "-map", "0:a:0?", "-vf", vf,
                       "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p",
                       "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                       "-color_range", "tv",
                       "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(tmp)]
                try:
                    r = subprocess.run(cmd, capture_output=True, timeout=1800)
                except Exception as e:  # noqa: BLE001
                    return err(str(e), 500)
                if r.returncode != 0 or not tmp.exists():
                    tmp.unlink(missing_ok=True)
                    return err("재생용 변환에 실패했어요: " + r.stderr[-300:].decode("utf-8", "replace"), 500)
                os.replace(tmp, out)
    return send_file(out, mimetype="video/mp4", conditional=True)


# ----------------------------------------------------------------------------
# 분석: 컷 감지 + 컷별 색 샘플 (영상 한 번만 읽음)
# ----------------------------------------------------------------------------
def analyze_file(job: dict, meta: dict, opts: dict, report) -> dict:
    fw, fh = meta["fw"], meta["fh"]
    W = 320
    Hs = max(2, int(round(W * fh / fw / 2)) * 2)
    SW = 96
    SH = max(2, int(round(SW * fh / fw)))
    dur = meta["duration"]
    F = min(2.0, max(0.05, 1500.0 / dur)) if dur > 0 else 1.0

    small = (f"scale={SW}:{SH}:in_color_matrix={meta['mat']}:in_range={meta['rng']}:flags=area,format=rgb24")
    if opts["cuts"]:
        graph = (f"[0:v:0]scale={W}:{Hs}:flags=bilinear,split=2[a][b];"
                 f"[a]select='gt(scene,{opts['thr']})',showinfo[c];"
                 f"[b]fps={F:.5f},{small}[d]")
        maps = ["-map", "[c]", "-f", "null", "-", "-map", "[d]"]
        loglevel = "info"
    else:
        graph = f"[0:v:0]fps={F:.5f},{small}[d]"
        maps = ["-map", "[d]"]
        loglevel = "error"
    cmd = [FFMPEG, "-hide_banner", "-nostats", "-loglevel", loglevel, "-i", str(src_path(meta)),
           "-filter_complex", graph, *maps, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    job["proc"] = proc
    cuts: list[float] = []
    tail = collections.deque(maxlen=30)

    def pump():
        for raw in proc.stderr:
            line = raw.decode("utf-8", "replace").rstrip()
            tail.append(line)
            if "showinfo" in line:
                m = re.search(r"pts_time:\s*(-?[0-9.]+)", line)
                if m:
                    cuts.append(float(m.group(1)))

    th = threading.Thread(target=pump, daemon=True)
    th.start()

    frames = []
    fbytes = SW * SH * 3
    expected = max(1.0, dur * F) if dur > 0 else 100.0
    while True:
        buf = proc.stdout.read(fbytes)
        if len(buf) < fbytes:
            break
        frames.append(np.frombuffer(buf, dtype=np.uint8).reshape(SH, SW, 3))
        if len(frames) % 8 == 0:
            report(min(0.99, len(frames) / expected))
        if job["cancel"].is_set():
            proc.kill()
            break
    proc.wait()
    th.join(timeout=5)
    job["proc"] = None
    if job["cancel"].is_set():
        raise Cancelled()
    if not frames:
        raise RuntimeError("영상에서 화면을 읽지 못했어요. " + " ".join(list(tail)[-3:]))

    t0 = meta["start_time"]
    end_t = t0 + dur if dur > 0 else None
    starts = [t0]
    for c in sorted(set(round(x, 4) for x in cuts)):
        if c - starts[-1] < MIN_CLIP:
            continue
        if end_t is not None and end_t - c < MIN_CLIP:
            continue
        starts.append(c)

    ts = t0 + np.arange(len(frames)) / F
    cid = np.clip(np.searchsorted(np.array(starts), ts, side="right") - 1, 0, len(starts) - 1)
    clips = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else None
        idx = np.where(cid == i)[0]
        if len(idx) == 0:   # 샘플 간격보다 짧은 컷: 가장 가까운 샘플 사용
            center = s + (((e if e is not None else (end_t or s + 1.0)) - s) / 2)
            idx = np.array([int(np.clip(round((center - t0) * F), 0, len(frames) - 1))])
        if len(idx) > 6:
            idx = idx[np.linspace(0, len(idx) - 1, 6).round().astype(int)]
        px = np.concatenate([frames[j].reshape(-1, 3) for j in idx]).astype(np.float32) / 255.0
        clips.append({"start": s, "end": e, "px": px})
    report(1.0)
    return {"starts": starts, "clips": clips}


def clip_index(starts, t: float) -> int:
    return int(max(0, np.searchsorted(np.array(starts), t + 1e-3, side="right") - 1))


# ----------------------------------------------------------------------------
# 내보내기: 컷마다 3D LUT를 만들어 한 번에 인코딩
# ----------------------------------------------------------------------------
def export_file(job: dict, meta: dict, an: dict, params: list, ent: dict) -> Path:
    jdir = JOBS_DIR / job["id"] / meta["id"]
    jdir.mkdir(parents=True, exist_ok=True)
    clips = an["clips"]

    names: dict[tuple, str] = {}
    filters = []
    for i, (clip, p) in enumerate(zip(clips, params)):
        if is_identity(p):
            continue
        key = tuple(round(v, 3) for v in p)
        if key not in names:
            names[key] = f"lut_{len(names):03d}.cube"
            write_cube(jdir / names[key], p)
        cond = f"gte(t,{clip['start'] - 0.0005:.4f})"
        if clip["end"] is not None:
            cond += f"*lt(t,{clip['end'] - 0.0005:.4f})"
        filters.append(f"lut3d=file={names[key]}:interp=tetrahedral:enable='{cond}'")

    flags = "bicubic+accurate_rnd+full_chroma_int"
    if filters:
        vf = (f"scale=in_color_matrix={meta['mat']}:in_range={meta['rng']}:flags={flags},format=gbrp,"
              + ",".join(filters) + ","
              f"scale=w=trunc(iw/2)*2:h=trunc(ih/2)*2:out_color_matrix={meta['mat']}:out_range={meta['rng']}:flags={flags},"
              "format=yuv420p")
    else:
        vf = "scale=w=trunc(iw/2)*2:h=trunc(ih/2)*2,format=yuv420p"

    preset, crf = QUALITY[job["options"]["quality"]]
    out = OUTPUTS / f"{job['id']}_{meta['id']}.mp4"
    tg = meta["tags"]
    cmd = [FFMPEG, "-y", "-hide_banner", "-nostats", "-loglevel", "error", "-progress", "pipe:1",
           "-i", str(src_path(meta)), "-map", "0:v:0", "-map", "0:a?", "-vf", vf,
           "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
           "-colorspace", tg["space"], "-color_primaries", tg["prim"], "-color_trc", tg["trc"],
           "-color_range", "pc" if meta["rng"] == "pc" else "tv"]
    if meta["audio"]:
        cmd += ["-c:a", "copy"] if meta["audio_codec"] in AUDIO_COPY_OK else ["-c:a", "aac", "-b:a", "192k"]
    cmd += ["-movflags", "+faststart", str(out)]

    log_path = jdir / "ffmpeg.log"
    with open(log_path, "wb") as logf:
        proc = subprocess.Popen(cmd, cwd=jdir, stdout=subprocess.PIPE, stderr=logf, text=True, bufsize=1)
        job["proc"] = proc
        dur = meta["duration"]
        for line in proc.stdout:
            if job["cancel"].is_set():
                proc.kill()
                break
            k, _, v = line.strip().partition("=")
            if k in ("out_time_us", "out_time_ms") and v.lstrip("-").isdigit() and dur > 0:
                ent["progress"] = max(ent["progress"], min(0.99, 0.15 + 0.85 * (int(v) / 1e6) / dur))
        proc.wait()
    job["proc"] = None
    if job["cancel"].is_set():
        out.unlink(missing_ok=True)
        raise Cancelled()
    if proc.returncode != 0 or not out.exists():
        tail = log_path.read_bytes()[-600:].decode("utf-8", "replace")
        raise RuntimeError("인코딩에 실패했어요. " + tail)
    return out


# ----------------------------------------------------------------------------
# 작업 실행
# ----------------------------------------------------------------------------
def set_stage(ent: dict, stage: str, prog: float | None = None) -> None:
    ent["stage"] = stage
    if prog is not None:
        ent["progress"] = prog


def run_job(job: dict) -> None:
    opts, p_user = job["options"], job["p_user"]
    entries = {e["id"]: e for e in job["files"]}
    ref_id = job["ref_id"]
    try:
        job["status"] = "running"
        analyses: dict[str, dict] = {}

        # 1) 기준 영상 분석 -> 목표 색감(T) 계산
        ref_meta, ref_ent = FILES[ref_id], entries[ref_id]
        set_stage(ref_ent, "컷 분석 중", 0)
        an = analyze_file(job, ref_meta, opts, lambda p: ref_ent.__setitem__("progress", 0.15 * p))
        analyses[ref_id] = an
        ref_idx = clip_index(an["starts"], ref_meta["start_time"] + ref_meta["preview_t"])
        target = stats(grade(an["clips"][ref_idx]["px"], p_user))

        # 2) 영상별 처리
        for fid in job["ids"]:
            ent, meta = entries[fid], FILES[fid]
            try:
                if job["cancel"].is_set():
                    raise Cancelled()
                an = analyses.pop(fid, None)
                if an is None:
                    set_stage(ent, "컷 분석 중", 0)
                    an = analyze_file(job, meta, opts, lambda p, e=ent: e.__setitem__("progress", 0.15 * p))
                set_stage(ent, "컷별 보정값 계산 중", 0.15)

                params = []
                for i, clip in enumerate(an["clips"]):
                    if fid == ref_id and i == ref_idx:
                        p = list(p_user)
                    else:
                        solved = solve_match(clip["px"], target, p_user, opts["enabled"])
                        p = [u + opts["strength"] * (q - u) for u, q in zip(p_user, solved)]
                    params.append(p)
                ent["clips"] = [{"start": round(c["start"] - meta["start_time"], 2),
                                 "p": [int(round(v * 100)) for v in p]}
                                for c, p in list(zip(an["clips"], params))[:300]]
                ent["cuts"] = len(an["clips"]) - 1

                set_stage(ent, "인코딩 중", 0.15)
                out = export_file(job, meta, an, params, ent)
                ent["output"] = str(out)
                ent["name_out"] = safe_stem(meta["name"]) + "_graded.mp4"
                set_stage(ent, "완료", 1.0)
                if meta.get("hdr"):
                    ent["message"] = "HDR 영상이라 색이 다르게 보일 수 있어요. (SDR 변환은 하지 않았어요)"
            except Cancelled:
                raise
            except Exception as e:  # noqa: BLE001
                set_stage(ent, "오류", ent["progress"])
                ent["message"] = str(e)
        job["status"] = "done"
    except Cancelled:
        job["status"] = "cancelled"
        for e in job["files"]:
            if e["stage"] not in ("완료", "오류"):
                e["stage"] = "취소됨"
    except Exception as e:  # noqa: BLE001
        job["status"] = "error"
        job["error"] = str(e)
    finally:
        job["proc"] = None


def public_job(job: dict) -> dict:
    return {
        "id": job["id"], "status": job["status"], "error": job["error"],
        "files": [{"id": e["id"], "name": e["name"], "stage": e["stage"], "progress": round(e["progress"], 3),
                   "message": e["message"], "done": bool(e.get("output")), "cuts": e.get("cuts"),
                   "clips": e.get("clips")} for e in job["files"]],
    }


@app.post("/api/process")
def api_process():
    j = request.get_json(force=True, silent=True) or {}
    ids = [i for i in j.get("ids", []) if i in FILES]
    if not ids:
        return err("보정할 영상이 없어요.")
    ref_id = j.get("ref_id") if j.get("ref_id") in FILES else ids[0]
    if ref_id not in ids:
        ids.insert(0, ref_id)

    o = j.get("options", {}) or {}
    match = o.get("match", {}) or {}
    enabled = [k for k, name in (("e", "exposure"), ("w", "wb"), ("c", "contrast"), ("s", "saturation"))
               if match.get(name, True)]
    options = {
        "cuts": bool(o.get("cuts", True)),
        "thr": SENS.get(o.get("sens"), SENS["normal"]),
        "enabled": enabled,
        "strength": max(0.0, min(100.0, to_float(o.get("strength"), 100.0))) / 100.0,
        "quality": o.get("quality") if o.get("quality") in QUALITY else "balanced",
    }
    with LOCK:
        if any(jb["status"] in ("queued", "running") for jb in JOBS.values()):
            return err("이미 보정이 진행 중이에요. 끝나거나 취소한 뒤 다시 눌러 주세요.", 409)
        job = {
            "id": uuid.uuid4().hex[:10], "status": "queued", "error": None, "ids": ids, "ref_id": ref_id,
            "p_user": parse_params(j.get("params")), "options": options,
            "cancel": threading.Event(), "proc": None,
            "files": [{"id": i, "name": FILES[i]["name"], "stage": "대기 중", "progress": 0.0, "message": ""}
                      for i in ids],
        }
        JOBS[job["id"]] = job
    threading.Thread(target=run_job, args=(job,), daemon=True).start()
    return jsonify(job_id=job["id"])


@app.get("/api/job/<jid>")
def api_job(jid):
    job = JOBS.get(jid)
    return jsonify(public_job(job)) if job else err("작업을 찾을 수 없어요.", 404)


@app.post("/api/job/<jid>/cancel")
def api_cancel(jid):
    job = JOBS.get(jid)
    if not job:
        return err("작업을 찾을 수 없어요.", 404)
    job["cancel"].set()
    proc = job.get("proc")
    if proc:
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
    return jsonify(ok=True)


@app.get("/api/download/<jid>/<fid>")
def api_download(jid, fid):
    job = JOBS.get(jid)
    ent = next((e for e in (job or {}).get("files", []) if e["id"] == fid), None)
    if not ent or not ent.get("output") or not Path(ent["output"]).exists():
        return err("파일이 아직 없어요.", 404)
    return send_file(ent["output"], as_attachment=True, download_name=ent["name_out"], mimetype="video/mp4")


# ----------------------------------------------------------------------------
def load_existing() -> None:
    for mp in UPLOADS.glob("*/meta.json"):
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
            if src_path(m).exists() and frame_path(m).exists():
                FILES[m["id"]] = m
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    if not ffmpeg_version():
        print("\n[!] ffmpeg 를 찾지 못했어요. ffmpeg 를 설치하고 PATH 에 추가한 뒤 다시 실행해 주세요.")
        print("    Windows: winget install ffmpeg   /   macOS: brew install ffmpeg\n")
        sys.exit(1)
    port = int(os.environ.get("PORT", "5000"))
    url = f"http://127.0.0.1:{port}"
    print(f"\n영상 색감 일괄 보정 도구 실행 중 -> {url}  (끝내려면 Ctrl+C)\n")
    if not os.environ.get("NO_BROWSER"):
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=port, threaded=True)