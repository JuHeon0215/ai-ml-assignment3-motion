# Assignment 3: 3D Motion Data Post Processing

Taekwondo (Taegeuk 1~3 jang) motion data is extracted from videos with MediaPipe BlazePose,
analyzed for problems, post-processed, and compared before and after.
ArmRaise and JumpingJack videos are used as a reference group.

## Environment
- Python 3.9, macOS (the plots use the AppleGothic font)
- `pip install -r requirements.txt`

## Pipeline
| Step | Script | Output |
|---|---|---|
| 1. Data Acquisition | `scripts/p1_create_raw_keypoints.py`, `scripts/a3_1_data_summary.py` | `data/raw_keypoints/*.npz`, `data/analysis/data_summary.csv` |
| 2. Problem Identification | `scripts/a3_2_problem_analysis.py` | `data/analysis/problem_metrics_raw.csv`, `figures/*_raw.png` |
| 3. Post-processing | `scripts/p1_2_post_processing.py` | `data/processed/*_target.npy`, `*_target.csv` |
| 4. Results & Analysis | `scripts/a3_3_compare.py` | `data/analysis/compare_metrics.csv`, `figures/*_compare.png` |
| 5. Bonus experiments (A: smoothing window, B: low-visibility interpolation, C: bone-length correction) | `scripts/extra_experiments.py` | `data/analysis/extra_<name>.txt`, `figures/extra_<name>_forearm.png` |

Run order: `p1` -> `a3_1` -> `a3_2` -> `p1_2` -> `a3_3`

## Reproducing
The full results for all 7 videos are saved in `data/analysis/`.
`data/processed/` contains Taegeuk 1, 2 and 3 jang (`<name>.npz` = raw keypoints with visibility,
`<name>_target.npy` = post-processed). Taegeuk 1 also has `_raw.npy` / `_target.csv`.
```
python scripts/a3_2_problem_analysis.py     # problem metrics / figures
python scripts/a3_3_compare.py              # raw vs target vs target+MA5
python scripts/extra_experiments.py 태극_1장   # bonus experiments A-C (also 태극_2장, 태극_3장)
```
The source videos are not included; run `p1` to download them again.

## Key results (Taegeuk 1 / 2 / 3 jang)
- Foot height: -0.81 m -> 0.00 m after `p1_2` (hip height 0.002 -> 0.805 m).
- 5-frame moving average: jitter -74 / -75 / -77%, mean joint shift 1.7 +- 0.3 cm.
- Bone-length correction (after MA5): bone-length CV -> 0.000, but 1.4 / 2.1 / 1.6% of joint-frames fall below the floor.
- Low-visibility interpolation was rejected (bone-length CV gets worse in all three videos).

## Videos
| Group | Video | URL |
|---|---|---|
| Taekwondo | Taegeuk 1 jang | https://youtube.com/shorts/XI4XXqZchNQ |
| Taekwondo | Taegeuk 2 jang | https://youtube.com/shorts/py4XUu6Hw4I |
| Taekwondo | Taegeuk 3 jang | https://youtube.com/shorts/UoFxuhfj9rQ |
| ArmRaise | Single_Arm_Dumbbell_Lateral_Raise | https://www.youtube.com/watch?v=fB1BN3TNDAo |
| ArmRaise | How_To_Do_A_STANDING_SINGLE_ARM_DUMBBELL_LATERAL_RAISE | https://www.youtube.com/watch?v=jZYA0-uwbwI |
| JumpingJack | Jumping Jack 1 | https://youtube.com/shorts/bT2iY8IjEU0 |
| JumpingJack | Jumping Jack 2 | https://youtube.com/shorts/yg3KQQn3QWg |
