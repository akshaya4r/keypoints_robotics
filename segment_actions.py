#!/usr/bin/env python3
"""Unsupervised action segmentation from the hand keypoint table.

Takes <stem>_keypoints.parquet from hand_keypoints.py and proposes action
phases without any labels:

  1. track     - pick one hand per frame (by label, or the hand that continues the
                 previous frame's track when labels flicker)
  2. features  - per-frame hand descriptors: finger curl, openness, pinch, palm
                 orientation, wrist position and speed; short gaps interpolated
  3. windows   - rolling mean / std over a short window (default 0.5 s) so each
                 frame describes local motion, not just a pose
  4. cluster   - standardise, PCA, k-means; k picked by silhouette unless --k
  5. cleanup   - majority filter + merge segments shorter than --min-seg
  6. outputs   - segments table, per-frame labels and features, a timeline plot,
                 a cluster profile heatmap, and (with --video) one row of example
                 frames per cluster plus an optional labelled video

The clusters are proposals: look at the example frames and the profile, then
name them (e.g. C1 = "grasp rod", C2 = "reach") or merge them.

Usage:
  python segment_actions.py hand_out/clip_keypoints.parquet \\
      --video hand_out/clip_annotated.mp4 --render

  # stereo source video instead of the annotated one:
  python segment_actions.py hand_out/clip_keypoints.parquet --video clip.mp4 --view left
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

NO_HAND = -1
FINGER_CHAINS = {  # joints from palm to tip; curl = sum of bend angles along the chain
    "thumb": [0, 1, 2, 3, 4],
    "index": [0, 5, 6, 7, 8],
    "middle": [0, 9, 10, 11, 12],
    "ring": [0, 13, 14, 15, 16],
    "pinky": [0, 17, 18, 19, 20],
}
TIPS = [4, 8, 12, 16, 20]
# features that get a rolling std (how much they vary inside the window)
MOTION_FEATURES = ["speed", "openness_mm", "pinch_mm", "curl_mean"]


# --------------------------------------------------------------------------- #
# 1. Hand track
# --------------------------------------------------------------------------- #
def load(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def infer_fps(df: pd.DataFrame) -> float:
    """Frame rate from frame index vs. (integer-rounded) timestamps over the whole clip."""
    f0, f1 = df["frame"].min(), df["frame"].max()
    t0, t1 = df["t_ms"].min(), df["t_ms"].max()
    return 1000.0 * (f1 - f0) / (t1 - t0) if t1 > t0 else 30.0


def hands_as_arrays(df: pd.DataFrame):
    """One entry per detected hand instance: keys table, pixel (N,21,2), world (N,21,3)."""
    df = df.sort_values(["frame", "hand_idx", "kp"])
    if len(df) % 21:
        raise ValueError("table does not contain complete 21-keypoint hands")
    keys = df[["frame", "hand_idx", "handedness", "handedness_score"]].iloc[::21].reset_index(drop=True)
    px = df[["x_px", "y_px"]].to_numpy(float).reshape(-1, 21, 2)
    w = df[["wx", "wy", "wz"]].to_numpy(float).reshape(-1, 21, 3)
    return keys, px, w


def select_track(keys: pd.DataFrame, px: np.ndarray, hand: str, fps: float) -> dict[int, int]:
    """Map frame -> row index of the chosen hand instance.

    hand="Left"/"Right": that label, highest score if duplicated.
    hand="any": the instance whose wrist is closest to the previously chosen one
    (robust to Left/Right label flicker); falls back to highest score after a gap.
    """
    chosen: dict[int, int] = {}
    if hand in ("Left", "Right"):
        k = keys[keys.handedness == hand]
        for f, g in k.groupby("frame"):
            chosen[int(f)] = int(g.handedness_score.idxmax())
        return chosen

    max_gap = int(0.5 * fps)
    prev_f, prev_xy = None, None
    for f, g in keys.groupby("frame"):
        idx = g.index.to_numpy()
        if prev_xy is not None and f - prev_f <= max_gap and len(idx) > 1:
            d = np.linalg.norm(px[idx, 0] - prev_xy, axis=1)
            j = int(idx[np.argmin(d)])
        else:
            j = int(g.handedness_score.idxmax())
        chosen[int(f)] = j
        prev_f, prev_xy = f, px[j, 0]
    return chosen


# --------------------------------------------------------------------------- #
# 2. Per-frame features
# --------------------------------------------------------------------------- #
def finger_curl_deg(w: np.ndarray, chain: list[int]) -> np.ndarray:
    """Sum of bend angles along a finger chain, for an (N,21,3) array."""
    total = np.zeros(len(w))
    for a, b, c in zip(chain[:-2], chain[1:-1], chain[2:]):
        u = w[:, b] - w[:, a]
        v = w[:, c] - w[:, b]
        cos = np.einsum("ij,ij->i", u, v) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
        total += np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return total


def frame_features(keys, px, w, chosen: dict[int, int], n_frames: int,
                   fps: float, max_gap_s: float) -> pd.DataFrame:
    frames = np.array(sorted(chosen))
    rows = np.array([chosen[f] for f in frames])
    P, Wd = px[rows], w[rows]

    # pixel quantities in units of the hand's own palm length (scale-free)
    palm_px = np.median(np.linalg.norm(P[:, 9] - P[:, 0], axis=1))
    feats = {
        "wrist_x": P[:, 0, 0] / palm_px,
        "wrist_y": P[:, 0, 1] / palm_px,
        "openness_mm": np.mean([np.linalg.norm(Wd[:, t] - Wd[:, 0], axis=1) for t in TIPS], axis=0) * 1000,
        "pinch_mm": np.linalg.norm(Wd[:, 4] - Wd[:, 8], axis=1) * 1000,
    }
    curls = {f"curl_{name}": finger_curl_deg(Wd, ch) for name, ch in FINGER_CHAINS.items()}
    feats.update(curls)
    feats["curl_mean"] = np.mean(list(curls.values()), axis=0)
    # palm normal (orientation): wrist -> index MCP x wrist -> pinky MCP
    n = np.cross(Wd[:, 5] - Wd[:, 0], Wd[:, 17] - Wd[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    feats.update({"palm_nx": n[:, 0], "palm_ny": n[:, 1], "palm_nz": n[:, 2]})

    F = pd.DataFrame(feats, index=frames).reindex(range(n_frames))
    F.index.name = "frame"
    detected = F["openness_mm"].notna()
    lim = max(int(round(max_gap_s * fps)), 0)
    if lim:
        F = F.interpolate(limit=lim, limit_area="inside")
    F["interpolated"] = F["openness_mm"].notna() & ~detected

    # wrist speed in palm lengths per second, from lightly smoothed positions
    xy = F[["wrist_x", "wrist_y"]].rolling(max(int(0.1 * fps), 1), center=True, min_periods=1).mean()
    speed = np.hypot(xy.wrist_x.diff(), xy.wrist_y.diff()) * fps
    # spikes are usually track jumps (e.g. to another person's hand), not motion
    F["speed"] = speed.clip(upper=speed.quantile(0.99))
    F["t_s"] = F.index / fps
    return F


# --------------------------------------------------------------------------- #
# 3-4. Windows + clustering
# --------------------------------------------------------------------------- #
BASE_FEATURES = ["wrist_x", "wrist_y", "speed", "openness_mm", "pinch_mm",
                 "curl_thumb", "curl_index", "curl_middle", "curl_ring", "curl_pinky",
                 "palm_nx", "palm_ny", "palm_nz"]


def window_features(F: pd.DataFrame, win: int) -> pd.DataFrame:
    roll = F[BASE_FEATURES].rolling(win, center=True, min_periods=max(win // 3, 1))
    X = roll.mean().add_suffix("_mean")
    X = X.join(F[MOTION_FEATURES].rolling(win, center=True, min_periods=max(win // 3, 1))
               .std().add_suffix("_std"))
    return X


def cluster(X: pd.DataFrame, valid: np.ndarray, k: int | None, k_range: tuple[int, int],
            position_weight: float, seed: int = 0):
    Xv = X[valid]
    Z = StandardScaler().fit_transform(Xv.to_numpy())
    Z = np.clip(Z, -4, 4)
    pos_cols = [X.columns.get_loc(c) for c in ("wrist_x_mean", "wrist_y_mean")]
    Z[:, pos_cols] *= position_weight
    Zp = PCA(n_components=0.95, random_state=seed).fit_transform(Z)

    scores = {}
    if k is None:
        rng = np.random.default_rng(seed)
        sub = rng.choice(len(Zp), size=min(4000, len(Zp)), replace=False)
        for kk in range(k_range[0], k_range[1] + 1):
            lab = KMeans(kk, n_init=10, random_state=seed).fit_predict(Zp)
            scores[kk] = float(silhouette_score(Zp[sub], lab[sub]))
        k = max(scores, key=scores.get)
    km = KMeans(k, n_init=20, random_state=seed).fit(Zp)
    labels = np.full(len(X), NO_HAND)
    labels[valid] = km.labels_
    return labels, k, scores


# --------------------------------------------------------------------------- #
# 5. Temporal cleanup
# --------------------------------------------------------------------------- #
def majority_filter(labels: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return labels
    vals = np.unique(labels)
    onehot = (labels[:, None] == vals[None, :]).astype(float)
    k = np.ones(win)
    counts = np.stack([np.convolve(onehot[:, j], k, mode="same") for j in range(len(vals))], axis=1)
    return vals[np.argmax(counts, axis=1)]


def runs(labels: np.ndarray) -> list[list[int]]:
    """[[label, start, end_exclusive], ...]"""
    edges = np.flatnonzero(np.diff(labels)) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, len(labels)]
    return [[int(labels[s]), int(s), int(e)] for s, e in zip(starts, ends)]


def merge_short(labels: np.ndarray, min_len: int) -> np.ndarray:
    """Repeatedly absorb the shortest too-short segment into its longer neighbour."""
    labels = labels.copy()
    while True:
        rs = runs(labels)
        short = [(e - s, i) for i, (_, s, e) in enumerate(rs) if e - s < min_len]
        if not short or len(rs) == 1:
            return labels
        _, i = min(short)
        nb = [j for j in (i - 1, i + 1) if 0 <= j < len(rs)]
        j = max(nb, key=lambda j: rs[j][2] - rs[j][1])
        labels[rs[i][1]:rs[i][2]] = rs[j][0]


def relabel_by_appearance(labels: np.ndarray) -> np.ndarray:
    order = [l for l in pd.unique(labels) if l != NO_HAND]
    mapping = {old: new for new, old in enumerate(order)}
    mapping[NO_HAND] = NO_HAND
    return np.array([mapping[l] for l in labels])


def name(l: int) -> str:
    return "no_hand" if l == NO_HAND else f"C{l + 1}"


# --------------------------------------------------------------------------- #
# 6. Outputs
# --------------------------------------------------------------------------- #
def colours(n: int) -> dict[int, tuple]:
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    c = {i: cmap(i % cmap.N) for i in range(n)}
    c[NO_HAND] = (0.85, 0.85, 0.85, 1.0)
    return c


def plot_timeline(F, labels, segs, k, path, title):
    col = colours(k)
    t = F.t_s.to_numpy()
    fig, ax = plt.subplots(5, 1, figsize=(14, 10), sharex=True,
                           gridspec_kw={"height_ratios": [0.6, 1, 1, 1, 1]})
    for _, s in segs.iterrows():
        ax[0].axvspan(s.start_s, s.end_s, color=col[s.label_id], lw=0)
        if s.duration_s > 0.03 * t[-1]:
            ax[0].text((s.start_s + s.end_s) / 2, 0.5, s.label, ha="center", va="center", fontsize=8)
    ax[0].set_yticks([])
    ax[0].set_ylabel("phase")
    for a in ax[1:]:
        for _, s in segs.iterrows():
            a.axvspan(s.start_s, s.end_s, color=col[s.label_id], alpha=0.12, lw=0)
    ax[1].plot(t, F.openness_mm, lw=0.8, label="openness")
    ax[1].plot(t, F.pinch_mm, lw=0.8, label="pinch")
    ax[1].set_ylabel("mm")
    ax[1].legend(loc="upper right")
    ax[2].plot(t, F.curl_mean, lw=0.8, color="tab:brown")
    ax[2].set_ylabel("mean finger\ncurl [deg]")
    ax[3].plot(t, F.speed, lw=0.6, color="tab:red")
    ax[3].set_ylabel("wrist speed\n[palm/s]")
    ax[4].plot(t, F.wrist_x, lw=0.8, label="x")
    ax[4].plot(t, F.wrist_y, lw=0.8, label="y")
    ax[4].set_ylabel("wrist pos\n[palm]")
    ax[4].legend(loc="upper right")
    ax[4].set_xlabel("time [s]")
    for a in ax[1:]:
        a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def cluster_profile(F, labels, k, path):
    cols = ["openness_mm", "pinch_mm", "curl_mean", "speed", "wrist_x", "wrist_y",
            "palm_nx", "palm_ny", "palm_nz"]
    D = F[cols].copy()
    D["label"] = labels
    D = D[D.label != NO_HAND]
    means = D.groupby("label")[cols].mean()
    z = (means - D[cols].mean()) / D[cols].std()
    frac = D.label.value_counts(normalize=True).sort_index()
    fig, ax = plt.subplots(figsize=(1.1 * len(cols) + 2, 0.6 * k + 1.5))
    im = ax.imshow(z.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{means.iat[i, j]:.2g}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right")
    ax.set_yticks(range(k), [f"{name(l)} ({frac[l]:.0%})" for l in means.index])
    ax.set_title("Cluster profiles: colour = z-score vs all frames, text = mean value")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return means.assign(share=frac)


def crop_view(frame, view):
    if view == "full":
        return frame
    h = frame.shape[1] // 2
    return frame[:, :h] if view == "left" else frame[:, h:2 * h]


def contact_sheet(video: Path, view: str, segs: pd.DataFrame, k: int, path: Path,
                  per_cluster: int = 4, thumb_w: int = 320):
    """One row per cluster: mid-frames of its longest segments."""
    col = colours(k)
    cap = cv2.VideoCapture(str(video))
    rows = []
    for l in range(k):
        s = segs[segs.label_id == l].nlargest(per_cluster, "duration_s")
        thumbs = []
        for _, r in s.iterrows():
            mid = (r.start_frame + r.end_frame) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ok, fr = cap.read()
            if not ok:
                continue
            fr = crop_view(fr, view)
            fr = cv2.resize(fr, (thumb_w, int(fr.shape[0] * thumb_w / fr.shape[1])))
            cv2.putText(fr, f"{r.start_s:.1f}-{r.end_s:.1f}s", (6, fr.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            thumbs.append(fr)
        if not thumbs:
            continue
        th = thumbs[0].shape[0]
        while len(thumbs) < per_cluster:
            thumbs.append(np.zeros_like(thumbs[0]))
        bgr = tuple(int(255 * c) for c in col[l][:3][::-1])
        tag = np.full((th, 90, 3), bgr, np.uint8)
        cv2.putText(tag, name(l), (10, th // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 0), 2, cv2.LINE_AA)
        rows.append(cv2.hconcat([tag] + thumbs))
    cap.release()
    if rows:
        cv2.imwrite(str(path), cv2.vconcat(rows))


def render_labels(video: Path, view: str, labels: np.ndarray, k: int, fps: float, path: Path):
    """Copy of the video with the current phase and a timeline bar burned in."""
    col = colours(k)
    bgr = {l: tuple(int(255 * c) for c in col[l][:3][::-1]) for l in col}
    cap = cv2.VideoCapture(str(video))
    writer, n = None, len(labels)
    for i in range(n):
        ok, fr = cap.read()
        if not ok:
            break
        fr = np.ascontiguousarray(crop_view(fr, view))
        H, W = fr.shape[:2]
        if writer is None:
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
            bar = np.zeros((14, W, 3), np.uint8)
            xs = (np.arange(W) * n / W).astype(int)
            bar[:] = np.array([bgr[l] for l in labels[xs]], np.uint8)[None]
        fr[H - 14:] = bar
        x = int(i * W / n)
        cv2.line(fr, (x, H - 18), (x, H), (255, 255, 255), 2)
        l = labels[i]
        cv2.rectangle(fr, (W - 170, 8), (W - 8, 48), bgr[l], -1)
        cv2.putText(fr, name(l), (W - 160, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
        writer.write(fr)
    cap.release()
    if writer is not None:
        writer.release()


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table", type=Path, help="<stem>_keypoints.parquet from hand_keypoints.py")
    ap.add_argument("--hand", choices=["any", "Left", "Right"], default="any",
                    help="which hand to segment; 'any' follows one hand across label flips")
    ap.add_argument("--k", type=int, default=None, help="number of clusters (default: silhouette)")
    ap.add_argument("--k-min", type=int, default=3)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--window", type=float, default=0.5, help="feature window [s]")
    ap.add_argument("--min-seg", type=float, default=0.5, help="shortest kept segment [s]")
    ap.add_argument("--max-gap", type=float, default=0.25,
                    help="interpolate detection gaps up to this long [s]")
    ap.add_argument("--position-weight", type=float, default=0.5,
                    help="weight of wrist image position (0 = ignore; head motion makes it noisy)")
    ap.add_argument("--video", type=Path, help="video for example frames (annotated or source)")
    ap.add_argument("--view", choices=["full", "left", "right"], default="full",
                    help="crop for --video; use left/right only for the raw stereo source")
    ap.add_argument("--render", action="store_true", help="also write a video with phase labels")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: next to table)")
    args = ap.parse_args()

    df = load(args.table)
    fps = infer_fps(df)
    n_frames = int(df["frame"].max()) + 1
    out = args.out or args.table.parent
    out.mkdir(parents=True, exist_ok=True)
    stem = out / args.table.stem.removesuffix("_keypoints")

    keys, px, w = hands_as_arrays(df)
    chosen = select_track(keys, px, args.hand, fps)
    F = frame_features(keys, px, w, chosen, n_frames, fps, args.max_gap)
    win = max(int(round(args.window * fps)), 1)
    X = window_features(F, win)
    valid = (F["openness_mm"].notna() & X.notna().all(axis=1)).to_numpy()

    labels, k, scores = cluster(X, valid, args.k, (args.k_min, args.k_max), args.position_weight)
    if scores:
        print("silhouette by k:", {kk: round(v, 3) for kk, v in scores.items()}, "-> k =", k)
    labels = majority_filter(labels, win)
    labels = merge_short(labels, max(int(round(args.min_seg * fps)), 1))
    labels = relabel_by_appearance(labels)
    k = int(labels.max()) + 1 if (labels != NO_HAND).any() else 0

    segs = pd.DataFrame(runs(labels), columns=["label_id", "start_frame", "end_frame"])
    segs.insert(1, "label", segs.label_id.map(name))
    segs["start_s"] = (segs.start_frame / fps).round(3)
    segs["end_s"] = (segs.end_frame / fps).round(3)
    segs["duration_s"] = (segs.end_s - segs.start_s).round(3)
    segs.index.name = "segment"
    segs.to_csv(f"{stem}_segments.csv")

    F.assign(label_id=labels, label=[name(l) for l in labels]).to_parquet(f"{stem}_segment_features.parquet")
    plot_timeline(F, labels, segs, k, f"{stem}_segmentation.png",
                  f"{args.table.name}: {k} clusters, {len(segs)} segments")
    prof = cluster_profile(F, labels, k, f"{stem}_cluster_profiles.png") if k else None

    print(f"{len(segs)} segments, {k} clusters; hand present in {valid.mean():.0%} of frames")
    summary = segs.groupby("label").agg(segments=("label", "size"), total_s=("duration_s", "sum"),
                                        median_s=("duration_s", "median"))
    print(summary.round(2).to_string())
    if prof is not None:
        print("\ncluster profiles (means):")
        print(prof[["openness_mm", "pinch_mm", "curl_mean", "speed", "share"]]
              .rename(index=name).round(2).to_string())

    if args.video:
        contact_sheet(args.video, args.view, segs, k, Path(f"{stem}_cluster_frames.jpg"))
        print(f"example frames -> {stem}_cluster_frames.jpg")
        if args.render:
            render_labels(args.video, args.view, labels, k, fps, Path(f"{stem}_segments.mp4"))
            print(f"labelled video -> {stem}_segments.mp4")
    print(f"segments -> {stem}_segments.csv; plots -> {stem}_segmentation.png, {stem}_cluster_profiles.png")


if __name__ == "__main__":
    main()
