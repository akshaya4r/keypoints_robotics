# keypoints_robotics

21-keypoint hand pose estimation on video with MediaPipe HandLandmarker, as a first step towards learning from human demonstrations.

## Scripts

- `hand_keypoints.py`: detects up to two hands per frame (VIDEO mode with temporal tracking), optionally smooths keypoints with a One Euro filter, renders an annotated video and saves a long-format keypoint table (`frame, hand, kp, x_px, y_px, z_rel, wx, wy, wz, ...`).
  It also writes inference metrics that need no ground truth: `<stem>_metrics.json` (per-frame read / preprocess / inference latency with mean, median, p95, p99; throughput and real-time factor; wall time per stage; CPU % and memory per stage) and `<stem>_frame_timing.csv`.
- `explore_keypoints.py`: plots detection coverage, pinch aperture, hand openness, the wrist trajectory and, if the timing CSV is present, per-frame latency.
- `segment_actions.py`: unsupervised action segmentation. Builds per-frame hand features (finger curl, openness, pinch, palm orientation, wrist position and speed), windows them, clusters with k-means (k by silhouette), cleans up short segments, and writes a segments table, timeline and cluster-profile plots, example frames per cluster and optionally a labelled video.

## Usage

```bash
pip install -r requirements.txt

# single or multiple videos
python hand_keypoints.py ego.mp4 side.mp4 --out hand_out --smooth --side-by-side

# side-by-side stereo video (e.g. 3840x1080): process one lens only
python hand_keypoints.py stereo.mp4 --view left --smooth

python explore_keypoints.py hand_out/<video>_keypoints.parquet --show

python segment_actions.py hand_out/<video>_keypoints.parquet \
    --video hand_out/<video>_annotated.mp4 --render
```

Segmentation clusters are proposals: check `<video>_cluster_frames.jpg` and `<video>_cluster_profiles.png`, then name or merge them. Useful flags: `--k`, `--window`, `--min-seg`, `--position-weight` (0 ignores where the hand is in the image).

The hand model (`hand_landmarker.task`) is downloaded to `models/` on first run.

Useful flags: `--flip-handedness` (if Left/Right are swapped), `--min-cutoff` / `--beta` (smoothing), `--max-frames` (quick trial), `--offset` (sync for side-by-side output), `--gpu` (MediaPipe GPU delegate, mainly Linux; falls back to CPU).

World coordinates (`wx, wy, wz`, metres) are relative to the hand centre: use them for intra-hand geometry, and pixel coordinates for hand trajectories in the image.
