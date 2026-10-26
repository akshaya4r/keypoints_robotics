#!/usr/bin/env python3
"""Quick exploratory plots for the keypoint table written by hand_keypoints.py.

Usage:
  python explore_keypoints.py hand_out/ego_keypoints.parquet [--show]

Plots (saved next to the table):
  - detection coverage per hand over time (rolling 1 s)
  - pinch aperture: thumb tip <-> index tip distance (mm, world coords)
  - hand openness: mean fingertip-to-wrist distance (mm, world coords)
  - wrist trajectory in the image, coloured by time
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"Left": "tab:orange", "Right": "tab:blue"}
TIPS = [4, 8, 12, 16, 20]


def load(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def wide_world(df: pd.DataFrame) -> pd.DataFrame:
    """Index (handedness, frame) -> columns (coord, kp) for world coordinates."""
    return df.pivot_table(index=["handedness", "frame"], columns="kp",
                          values=["wx", "wy", "wz"])


def kp_xyz(w: pd.DataFrame, kp: int) -> np.ndarray:
    return np.stack([w[("wx", kp)], w[("wy", kp)], w[("wz", kp)]], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("table", type=Path)
    ap.add_argument("--fps", type=float, default=0, help="override; default inferred from t_ms")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    df = load(args.table)
    fps = args.fps or 1000.0 / np.median(np.diff(np.sort(df["t_ms"].unique())))
    n_frames = int(df["frame"].max()) + 1
    t = np.arange(n_frames) / fps
    stem = args.table.with_suffix("")

    w = wide_world(df)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    for hand in sorted(df["handedness"].unique()):
        c = COLORS.get(hand)
        present = np.zeros(n_frames)
        present[df.loc[df.handedness == hand, "frame"].unique()] = 1
        win = max(int(fps), 1)
        axes[0].plot(t, pd.Series(present).rolling(win, min_periods=1).mean(), color=c, label=hand)

        wh = w.loc[hand]
        th = wh.index.to_numpy() / fps
        pinch = np.linalg.norm(kp_xyz(wh, 4) - kp_xyz(wh, 8), axis=1) * 1000
        wrist = kp_xyz(wh, 0)
        openness = np.mean([np.linalg.norm(kp_xyz(wh, k) - wrist, axis=1) for k in TIPS], axis=0) * 1000
        axes[1].plot(th, pinch, ".", ms=2, color=c, label=hand)
        axes[2].plot(th, openness, ".", ms=2, color=c, label=hand)

    axes[0].set_ylabel("detected (1 s mean)")
    axes[1].set_ylabel("pinch aperture [mm]")
    axes[2].set_ylabel("openness [mm]")
    axes[2].set_xlabel("time [s]")
    for ax in axes:
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
    fig.suptitle(args.table.name)
    fig.tight_layout()
    fig.savefig(f"{stem}_timeseries.png", dpi=130)

    fig2, ax = plt.subplots(figsize=(8, 6))
    wr = df[df.kp == 0]
    for hand, g in wr.groupby("handedness"):
        sc = ax.scatter(g.x_px, g.y_px, c=g.frame / fps, s=3, cmap="viridis",
                        marker="o" if hand == "Right" else "^", label=hand)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    ax.set_title("Wrist trajectory (colour = time [s])")
    ax.legend()
    fig2.colorbar(sc, ax=ax)
    fig2.tight_layout()
    fig2.savefig(f"{stem}_wrist_traj.png", dpi=130)

    print(f"saved {stem}_timeseries.png and {stem}_wrist_traj.png")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
