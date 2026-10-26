# keypoints_robotics

21-keypoint hand pose estimation on video with MediaPipe HandLandmarker, as a first step towards learning from human demonstrations.

## Scripts

- `hand_keypoints.py`: detects up to two hands per frame (VIDEO mode with temporal tracking), optionally smooths keypoints with a One Euro filter, renders an annotated video and saves a long-format keypoint table (`frame, hand, kp, x_px, y_px, z_rel, wx, wy, wz, ...`).
- `explore_keypoints.py`: plots detection coverage, pinch aperture, hand openness and the wrist trajectory from that table.

## Usage

```bash
pip install -r requirements.txt

# single or multiple videos
python hand_keypoints.py ego.mp4 side.mp4 --out hand_out --smooth --side-by-side

# side-by-side stereo video (e.g. 3840x1080): process one lens only
python hand_keypoints.py stereo.mp4 --view left --smooth

python explore_keypoints.py hand_out/<video>_keypoints.parquet --show
```

The hand model (`hand_landmarker.task`) is downloaded to `models/` on first run.

Useful flags: `--flip-handedness` (if Left/Right are swapped), `--min-cutoff` / `--beta` (smoothing), `--max-frames` (quick trial), `--offset` (sync for side-by-side output), `--gpu` (MediaPipe GPU delegate, mainly Linux; falls back to CPU).

World coordinates (`wx, wy, wz`, metres) are relative to the hand centre: use them for intra-hand geometry, and pixel coordinates for hand trajectories in the image.
