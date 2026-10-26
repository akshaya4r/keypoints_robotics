#!/usr/bin/env python3
"""21-keypoint hand estimation on video with MediaPipe HandLandmarker.

Pipeline (per video):
  1. detect   - HandLandmarker in VIDEO mode (uses temporal tracking between frames)
  2. smooth   - optional offline One Euro filter per (hand, keypoint, coordinate)
  3. render   - annotated .mp4 with skeletons
  4. save     - tidy long-format table: one row per (frame, hand, keypoint)

Optionally composes the annotated views side by side (assumes the clips are
roughly frame-synchronised; use --offset to shift the second one).

Stereo cameras that store both lenses side by side in one frame (e.g. 3840x1080
= 2 x 1920x1080): pass --view left (or right) so each hand is detected once.

Usage:
  python hand_keypoints.py ego.mp4 side.mp4 --out out/ --gpu --smooth --side-by-side
  python hand_keypoints.py stereo_ego.mp4 --view left --smooth

Inference metrics (no ground truth needed) are written per video:
  <stem>_metrics.json       speed (per-frame read / preprocess / inference latency,
                            throughput, real-time factor, per-stage wall time) and
                            resources (process CPU % and RSS memory, sampled in the
                            background; per stage). Needs `psutil` for CPU/memory
                            sampling; without it only peak memory is reported.
  <stem>_frame_timing.csv   one row per frame: read_ms, prep_ms, infer_ms, n_hands

Columns in <stem>_keypoints.parquet:
  video, frame, t_ms, hand_idx, handedness, handedness_score, label_fixed,
  kp, kp_name, x_px, y_px, z_rel, wx, wy, wz
  - x_px, y_px : image coordinates in pixels
  - z_rel      : MediaPipe relative depth (wrist ~ 0, smaller = closer to camera,
                 roughly in image-width units)
  - wx, wy, wz : "world" landmarks in metres, origin at the hand's geometric
                 centre. Good for intra-hand geometry (pinch aperture, finger
                 flexion), NOT for global hand trajectories.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

try:
    import psutil
except ImportError:  # metrics still work, minus CPU/memory sampling
    psutil = None

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
KP_NAMES = ["wrist"] + [
    f"{f}_{j}"
    for f in FINGERS
    for j in (["cmc", "mcp", "ip", "tip"] if f == "thumb" else ["mcp", "pip", "dip", "tip"])
]
assert len(KP_NAMES) == 21

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]
COLORS = {"Left": (255, 140, 0), "Right": (0, 200, 255)}  # BGR
FINGERTIPS = [4, 8, 12, 16, 20]

COLUMNS = [
    "video", "frame", "t_ms", "hand_idx", "handedness", "handedness_score",
    "label_fixed", "kp", "kp_name", "x_px", "y_px", "z_rel", "wx", "wy", "wz",
]


def crop_view(frame: np.ndarray, view: str) -> np.ndarray:
    """Return the left/right half of a side-by-side stereo frame, or the full frame."""
    if view == "full":
        return frame
    half = frame.shape[1] // 2
    return frame[:, :half] if view == "left" else frame[:, half:2 * half]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
class ResourceMonitor:
    """Samples this process's CPU % and resident memory in a background thread.

    CPU % is psutil's per-process value: 100 % = one fully used core, so it can
    exceed 100 on multi-core machines. `stage` tags each sample with the
    pipeline stage that was running.
    """

    def __init__(self, interval_s: float = 0.25):
        self.interval_s = interval_s
        self.samples: list[tuple[float, str, float, float]] = []
        self.stage = "init"
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        if psutil is not None:
            self._proc = psutil.Process()
            self._proc.cpu_percent(None)  # prime the counter
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _run(self):
        while not self._stop.wait(self.interval_s):
            self.samples.append((time.perf_counter(), self.stage,
                                 self._proc.cpu_percent(None),
                                 self._proc.memory_info().rss / 2**20))

    def summary(self) -> dict:
        out = {"available": psutil is not None, "peak_rss_mb_os": peak_rss_mb()}
        if psutil is None or not self.samples:
            return out
        df = pd.DataFrame(self.samples, columns=["t", "stage", "cpu_pct", "rss_mb"])
        n_cores = psutil.cpu_count(logical=True) or 1

        def _agg(g):
            return {
                "cpu_pct_mean": round(g.cpu_pct.mean(), 1),
                "cpu_pct_peak": round(g.cpu_pct.max(), 1),
                "cpu_cores_used_mean": round(g.cpu_pct.mean() / 100, 2),
                "rss_mb_mean": round(g.rss_mb.mean(), 1),
                "rss_mb_peak": round(g.rss_mb.max(), 1),
                "n_samples": int(len(g)),
            }

        out.update({
            "logical_cores": n_cores,
            "system_memory_gb": round(psutil.virtual_memory().total / 2**30, 1),
            "overall": _agg(df),
            "by_stage": {st: _agg(g) for st, g in df.groupby("stage", sort=False)},
        })
        return out


def peak_rss_mb() -> float | None:
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(r / 2**20 if sys.platform == "darwin" else r / 2**10, 1)  # bytes vs KiB
    except Exception:
        return None


def latency_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {}
    return {
        "mean_ms": round(float(x.mean()), 2),
        "median_ms": round(float(np.median(x)), 2),
        "p95_ms": round(float(np.percentile(x, 95)), 2),
        "p99_ms": round(float(np.percentile(x, 99)), 2),
        "max_ms": round(float(x.max()), 2),
        "std_ms": round(float(x.std()), 2),
    }


def speed_summary(timing: pd.DataFrame, video_fps: float, warmup: int = 10) -> dict:
    """Per-frame latency stats; the first `warmup` frames are reported separately."""
    steady = timing.iloc[warmup:] if len(timing) > warmup else timing
    total = steady.read_ms + steady.prep_ms + steady.infer_ms
    fps_infer = 1000.0 / steady.infer_ms.mean() if len(steady) else float("nan")
    fps_e2e = 1000.0 / total.mean() if len(steady) else float("nan")
    return {
        "frames": int(len(timing)),
        "warmup_frames_excluded": int(min(warmup, len(timing))),
        "first_frame_infer_ms": round(float(timing.infer_ms.iloc[0]), 2) if len(timing) else None,
        "read_decode": latency_stats(steady.read_ms),
        "preprocess": latency_stats(steady.prep_ms),
        "inference": latency_stats(steady.infer_ms),
        "end_to_end_per_frame": latency_stats(total),
        # MediaPipe re-runs the palm detector when tracking is lost, so latency
        # depends on how many hands are being tracked.
        "inference_by_n_hands": {
            str(n): {**latency_stats(g.infer_ms), "frames": int(len(g))}
            for n, g in steady.groupby("n_hands")
        },
        "throughput_fps_inference_only": round(fps_infer, 1),
        "throughput_fps_end_to_end": round(fps_e2e, 1),
        "real_time_factor": round(fps_e2e / video_fps, 2) if video_fps else None,
    }


def system_info(delegate: str) -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "python": platform.python_version(),
        "mediapipe": getattr(mp, "__version__", None),
        "opencv": cv2.__version__,
        "delegate": delegate,
        "logical_cores": os.cpu_count(),
    }
    if psutil is not None:
        info["physical_cores"] = psutil.cpu_count(logical=False)
    return info


# --------------------------------------------------------------------------- #
# Model / landmarker
# --------------------------------------------------------------------------- #
def ensure_model(path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading hand_landmarker.task -> {path}")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


def make_landmarker(model_path: Path, use_gpu: bool, num_hands: int,
                    det_conf: float, track_conf: float) -> tuple[vision.HandLandmarker, str]:
    def _create(delegate):
        opts = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(
                model_asset_path=str(model_path), delegate=delegate),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=det_conf,
            min_hand_presence_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )
        return vision.HandLandmarker.create_from_options(opts)

    if use_gpu:
        try:
            lm = _create(mp_tasks.BaseOptions.Delegate.GPU)
            print("Using GPU delegate")
            return lm, "GPU"
        except Exception as e:  # GPU delegate is platform-dependent
            print(f"GPU delegate unavailable ({e.__class__.__name__}: {e}); falling back to CPU")
    return _create(mp_tasks.BaseOptions.Delegate.CPU), "CPU"


# --------------------------------------------------------------------------- #
# 1. Detection
# --------------------------------------------------------------------------- #
def detect(video_path: Path, model_path: Path, args
           ) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.view != "full":
        W //= 2
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    meta = {"fps": fps, "W": W, "H": H, "n_frames": n_total, "view": args.view}

    swap = {"Left": "Right", "Right": "Left"}
    rows, timing = [], []
    t_load = time.perf_counter()
    lm, delegate = make_landmarker(model_path, args.gpu, args.num_hands,
                                   args.det_conf, args.track_conf)
    meta["model_load_ms"] = round((time.perf_counter() - t_load) * 1000, 1)
    meta["delegate"] = delegate
    try:
        i = 0
        while True:
            if args.max_frames and i >= args.max_frames:
                break
            t0 = time.perf_counter()
            ok, frame = cap.read()
            t1 = time.perf_counter()
            if not ok:
                break
            rgb = cv2.cvtColor(np.ascontiguousarray(crop_view(frame, args.view)),
                               cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            t2 = time.perf_counter()
            t_ms = int(round(i * 1000.0 / fps))
            res = lm.detect_for_video(img, t_ms)
            t3 = time.perf_counter()
            timing.append((i, (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3,
                           len(res.hand_landmarks)))

            # Handedness labels for this frame (optionally flipped).
            labels, scores = [], []
            for hd in res.handedness:
                lab = hd[0].category_name
                labels.append(swap[lab] if args.flip_handedness else lab)
                scores.append(hd[0].score)
            # If two hands got the same label, relabel the less confident one.
            fixed = [False] * len(labels)
            if len(labels) == 2 and labels[0] == labels[1]:
                j = int(np.argmin(scores))
                labels[j] = swap[labels[j]]
                fixed[j] = True

            for h, (lms, wlms) in enumerate(zip(res.hand_landmarks, res.hand_world_landmarks)):
                for k, (p, w) in enumerate(zip(lms, wlms)):
                    rows.append((
                        video_path.stem, i, t_ms, h, labels[h], scores[h], fixed[h],
                        k, KP_NAMES[k], p.x * W, p.y * H, p.z, w.x, w.y, w.z,
                    ))
            if i % 200 == 0:
                print(f"  {video_path.name}: frame {i}/{n_total}")
            i += 1
        meta["n_frames"] = i
    finally:
        lm.close()
        cap.release()
    timing_df = pd.DataFrame(timing, columns=["frame", "read_ms", "prep_ms", "infer_ms", "n_hands"])
    return pd.DataFrame(rows, columns=COLUMNS), meta, timing_df


# --------------------------------------------------------------------------- #
# 2. Smoothing (offline One Euro filter, reset across detection gaps)
# --------------------------------------------------------------------------- #
def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


def one_euro(x: np.ndarray, t: np.ndarray, min_cutoff: float, beta: float,
             d_cutoff: float = 1.0) -> np.ndarray:
    out = np.empty_like(x)
    out[0] = x_prev = x[0]
    dx_prev = 0.0
    for i in range(1, len(x)):
        dt = t[i] - t[i - 1]
        if dt <= 0:
            out[i] = x[i]
            continue
        dx = (x[i] - x_prev) / dt
        a_d = _alpha(d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * dx_prev
        a = _alpha(min_cutoff + beta * abs(dx_hat), dt)
        x_hat = a * x[i] + (1 - a) * x_prev
        out[i] = x_prev = x_hat
        dx_prev = dx_hat
    return out


def smooth(df: pd.DataFrame, W: int, min_cutoff: float, beta: float,
           gap_s: float = 0.25) -> pd.DataFrame:
    """Filter each (handedness, kp) track; restart the filter after gaps > gap_s.

    Signals are put on comparable scales before filtering (pixels / frame width,
    metres as-is), so one set of parameters works for all coordinates.
    """
    if df.empty:
        return df
    df = df.sort_values(["handedness", "kp", "frame"]).copy()
    scales = {"x_px": W, "y_px": W, "z_rel": 1.0, "wx": 1.0, "wy": 1.0, "wz": 1.0}
    for _, idx in df.groupby(["handedness", "kp"]).groups.items():
        g = df.loc[idx]
        t = g["t_ms"].to_numpy() / 1000.0
        seg_id = np.concatenate([[0], np.cumsum(np.diff(t) > gap_s)])
        for s in np.unique(seg_id):
            m = seg_id == s
            sub_idx = g.index[m]
            for col, sc in scales.items():
                v = g.loc[sub_idx, col].to_numpy(dtype=float) / sc
                df.loc[sub_idx, col] = one_euro(v, t[m], min_cutoff, beta) * sc
    return df.sort_values(["frame", "handedness", "kp"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 3. Rendering
# --------------------------------------------------------------------------- #
def draw_hand(frame: np.ndarray, pts: np.ndarray, label: str) -> None:
    color = COLORS.get(label, (0, 255, 0))
    p = np.round(pts).astype(int)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(p[a]), tuple(p[b]), color, 2, cv2.LINE_AA)
    for k, q in enumerate(p):
        r = 5 if k in FINGERTIPS else 3
        cv2.circle(frame, tuple(q), r, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, tuple(q), r, color, 1, cv2.LINE_AA)
    cv2.putText(frame, label, tuple(p[0] + np.array([8, 18])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def render(video_path: Path, df: pd.DataFrame, meta: dict, out_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             meta["fps"], (meta["W"], meta["H"]))
    by_frame = {f: g for f, g in df.groupby("frame")}
    for i in range(meta["n_frames"]):
        ok, frame = cap.read()
        if not ok:
            break
        frame = np.ascontiguousarray(crop_view(frame, meta.get("view", "full")))
        g = by_frame.get(i)
        if g is not None:
            for label, gh in g.groupby("handedness"):
                gh = gh.sort_values("kp")
                if len(gh) == 21:
                    draw_hand(frame, gh[["x_px", "y_px"]].to_numpy(), label)
        cv2.putText(frame, f"{i}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    cap.release()


def side_by_side(paths: list[Path], out_path: Path, offset: int, height: int = 540) -> None:
    """hconcat two annotated videos; second one starts `offset` frames later."""
    caps = [cv2.VideoCapture(str(p)) for p in paths]
    fps = caps[0].get(cv2.CAP_PROP_FPS) or 30.0
    if offset > 0:
        caps[1].set(cv2.CAP_PROP_POS_FRAMES, offset)
    elif offset < 0:
        caps[0].set(cv2.CAP_PROP_POS_FRAMES, -offset)
    writer = None
    while True:
        frames = []
        for c in caps:
            ok, f = c.read()
            if not ok:
                break
            frames.append(cv2.resize(f, (int(f.shape[1] * height / f.shape[0]), height)))
        if len(frames) != len(caps):
            break
        combo = cv2.hconcat(frames)
        if writer is None:
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (combo.shape[1], combo.shape[0]))
        writer.write(combo)
    if writer is not None:
        writer.release()
    for c in caps:
        c.release()


# --------------------------------------------------------------------------- #
# 4. Save
# --------------------------------------------------------------------------- #
def save_table(df: pd.DataFrame, path_stem: Path) -> Path:
    try:
        p = path_stem.with_suffix(".parquet")
        df.to_parquet(p, index=False)
    except (ImportError, ValueError):
        p = path_stem.with_suffix(".csv")
        df.to_csv(p, index=False)
    return p


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("hand_out"))
    ap.add_argument("--model", type=Path, default=Path("models/hand_landmarker.task"))
    ap.add_argument("--gpu", action="store_true", help="try the GPU delegate")
    ap.add_argument("--num-hands", type=int, default=2)
    ap.add_argument("--det-conf", type=float, default=0.5)
    ap.add_argument("--track-conf", type=float, default=0.5)
    ap.add_argument("--flip-handedness", action="store_true",
                    help="swap Left/Right labels (check on a few frames first)")
    ap.add_argument("--smooth", action="store_true", help="apply One Euro smoothing")
    ap.add_argument("--min-cutoff", type=float, default=1.0,
                    help="One Euro min cutoff (Hz): lower = smoother at rest, more lag")
    ap.add_argument("--beta", type=float, default=10.0,
                    help="One Euro speed coefficient: higher = less lag on fast motion")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    ap.add_argument("--view", choices=["full", "left", "right"], default="full",
                    help="for side-by-side stereo video: process only one half")
    ap.add_argument("--side-by-side", action="store_true")
    ap.add_argument("--offset", type=int, default=0,
                    help="frames by which video 2 lags video 1 (for --side-by-side)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    model_path = ensure_model(args.model)

    annotated = []
    for vp in args.videos:
        stage_s = {}
        with ResourceMonitor() as mon:
            print(f"[{vp.name}] detecting")
            mon.stage = "detect"
            t = time.perf_counter()
            df, meta, timing = detect(vp, model_path, args)
            stage_s["detect"] = time.perf_counter() - t
            cover = df.groupby("handedness")["frame"].nunique() / max(meta["n_frames"], 1)
            print(f"[{vp.name}] {meta['n_frames']} frames @ {meta['fps']:.1f} fps; "
                  f"coverage: {cover.round(3).to_dict()}")
            if args.smooth:
                print(f"[{vp.name}] smoothing")
                mon.stage = "smooth"
                t = time.perf_counter()
                df = smooth(df, meta["W"], args.min_cutoff, args.beta)
                stage_s["smooth"] = time.perf_counter() - t
            mon.stage = "save"
            t = time.perf_counter()
            table = save_table(df, args.out / f"{vp.stem}_keypoints")
            stage_s["save"] = time.perf_counter() - t
            out_vid = args.out / f"{vp.stem}_annotated.mp4"
            print(f"[{vp.name}] rendering -> {out_vid}")
            mon.stage = "render"
            t = time.perf_counter()
            render(vp, df, meta, out_vid)
            stage_s["render"] = time.perf_counter() - t
        annotated.append(out_vid)
        print(f"[{vp.name}] keypoints -> {table}")

        speed = speed_summary(timing, meta["fps"])
        metrics = {
            "video": {"file": vp.name, "fps": meta["fps"], "frames": meta["n_frames"],
                      "processed_resolution": [meta["W"], meta["H"]], "view": meta["view"],
                      "duration_s": round(meta["n_frames"] / meta["fps"], 2)},
            "settings": {"num_hands": args.num_hands, "det_conf": args.det_conf,
                         "track_conf": args.track_conf, "smooth": args.smooth,
                         "model": str(model_path)},
            "system": system_info(meta["delegate"]),
            "model_load_ms": meta["model_load_ms"],
            "speed": speed,
            "stage_wall_time_s": {k: round(v, 2) for k, v in stage_s.items()},
            "resources": mon.summary(),
        }
        mpath = args.out / f"{vp.stem}_metrics.json"
        mpath.write_text(json.dumps(metrics, indent=2))
        timing.to_csv(args.out / f"{vp.stem}_frame_timing.csv", index=False)
        inf = speed["inference"]
        res = metrics["resources"].get("by_stage", {}).get("detect", {})
        print(f"[{vp.name}] inference {inf.get('mean_ms')} ms mean / {inf.get('p95_ms')} ms p95; "
              f"end-to-end {speed['throughput_fps_end_to_end']} fps "
              f"({speed['real_time_factor']}x real time); "
              f"CPU {res.get('cpu_pct_mean', 'n/a')} % mean; "
              f"peak RSS {metrics['resources']['peak_rss_mb_os']} MB -> {mpath}")

    if args.side_by_side and len(annotated) >= 2:
        sbs = args.out / "side_by_side.mp4"
        print(f"composing -> {sbs}")
        side_by_side(annotated[:2], sbs, args.offset)


if __name__ == "__main__":
    main()
