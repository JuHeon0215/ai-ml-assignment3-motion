# scripts/a3_1_data_summary.py
# Assignment 3 - Step 1: Data Acquisition summary

import os
import glob

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_KEYPOINT_DIR = os.path.join(ROOT, "data", "raw_keypoints")
RAW_VIDEO_DIR = os.path.join(ROOT, "data", "raw_videos")
OUT_DIR = os.path.join(ROOT, "data", "analysis")
os.makedirs(OUT_DIR, exist_ok=True)


def video_info(name):
    """Return (fps, total frames) if the source video exists, else (None, None)."""
    path = os.path.join(RAW_VIDEO_DIR, f"{name}.mp4")
    if not os.path.exists(path):
        return None, None
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return round(fps, 2), n_frames


def summarize(path):
    name = os.path.splitext(os.path.basename(path))[0]
    data = np.load(path, allow_pickle=True)["data"]
    df = pd.DataFrame(
        data,
        columns=["frame", "landmark", "x", "y", "z", "visibility"]
    )

    if df.empty:
        return {"name": name, "detected_frames": 0}

    df["frame"] = df["frame"].astype(int)
    detected = df["frame"].nunique()
    fps, video_frames = video_info(name)
    total = video_frames if video_frames else int(df["frame"].max()) + 1

    return {
        "name": name,
        "fps": fps,
        "video_frames": total,
        "detected_frames": detected,
        "missing_frames": max(total - detected, 0),
        "shape": f"({detected}, 33, 3)",
        "rows_complete": len(df) == detected * 33,
        "duration_s": round(total / fps, 1) if fps else None,
        "mean_visibility": round(df["visibility"].mean(), 3),
        "low_vis_ratio": round((df["visibility"] < 0.5).mean(), 3),
    }


def main():
    paths = sorted(glob.glob(os.path.join(RAW_KEYPOINT_DIR, "*.npz")))
    if not paths:
        print(f"No .npz files found in {RAW_KEYPOINT_DIR}")
        return

    table = pd.DataFrame([summarize(p) for p in paths])

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 40)
    print(table.to_string(index=False))

    out_path = os.path.join(OUT_DIR, "data_summary.csv")
    table.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()