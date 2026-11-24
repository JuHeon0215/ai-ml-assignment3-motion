# scripts/a3_2_problem_analysis.py
# Assignment 3 - Step 2: Problem Identification (raw motion data)

import os
import re
import sys
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
OUT_DIR = os.path.join(ROOT, "data", "analysis")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams["font.family"] = "AppleGothic"   # macOS Korean font
plt.rcParams["axes.unicode_minus"] = False

# BlazePose landmark indices
FOOT_IDS = [27, 28, 29, 30, 31, 32]   # ankles, heels, toes
BONES = {
    "L upper arm": (11, 13),
    "L forearm":   (13, 15),
    "L thigh":     (23, 25),
    "L shank":     (25, 27),
}


# ---------- metrics (reused in step 4 for raw vs processed) ----------
def lowest_foot_height(motion):
    """(T,) height of the lowest foot joint per frame (y axis = up)."""
    return motion[:, FOOT_IDS, 1].min(axis=1)


def jitter(motion):
    """(T-2,) mean joint acceleration magnitude per frame (second difference)."""
    acc = motion[2:] - 2 * motion[1:-1] + motion[:-2]
    return np.linalg.norm(acc, axis=2).mean(axis=1)


def bone_lengths(motion):
    return {
        name: np.linalg.norm(motion[:, a] - motion[:, b], axis=1)
        for name, (a, b) in BONES.items()
    }


def compute_metrics(name, motion):
    foot = lowest_foot_height(motion)
    jit = jitter(motion)
    bl = bone_lengths(motion)
    return {
        "name": name,
        "frames": len(motion),
        "foot_mean_y": round(float(foot.mean()), 4),
        "foot_range": round(float(foot.max() - foot.min()), 4),
        "foot_std": round(float(foot.std()), 4),
        "jitter_mean": round(float(jit.mean()), 5),
        "jitter_p95": round(float(np.percentile(jit, 95)), 5),
        "bone_len_cv": round(float(np.mean([l.std() / l.mean() for l in bl.values()])), 4),
    }


# ---------- plotting ----------
def plot_problems(name, motion, path):
    foot = lowest_foot_height(motion)
    jit = jitter(motion)
    bl = bone_lengths(motion)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    fig.suptitle(f"Raw motion problems: {name}")

    axes[0].plot(foot)
    axes[0].axhline(0, color="r", ls="--", lw=1, label="ground (y=0)")
    axes[0].set_title("Lowest foot height per frame")
    axes[0].set_ylabel("y (m)")
    axes[0].legend()

    axes[1].plot(np.arange(1, len(jit) + 1), jit)
    axes[1].set_title("Jitter (mean joint acceleration per frame)")
    axes[1].set_ylabel("m / frame^2")

    for n, l in bl.items():
        axes[2].plot(l / l.mean(), label=n)
    axes[2].set_title("Bone length over time (normalized by mean)")
    axes[2].set_xlabel("frame")
    axes[2].legend(ncol=4, fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def safe_name(name):
    return re.sub(r"[^\w\-]+", "_", name)[:60]


def visualize(name):
    sys.path.insert(0, ROOT)
    from utils.viser_test import PoseViser
    motion = np.load(os.path.join(PROCESSED_DIR, f"{name}_raw.npy"))
    print("shape:", motion.shape)
    PoseViser(fps=30).play_sequence(motion)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viz", help="video name (without _raw.npy) to play in the viser viewer")
    args = parser.parse_args()

    if args.viz:
        visualize(args.viz)
        return

    paths = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*_raw.npy")))
    if not paths:
        print(f"No *_raw.npy files in {PROCESSED_DIR}")
        return

    rows = []
    for p in paths:
        name = os.path.basename(p)[: -len("_raw.npy")]
        motion = np.load(p)
        rows.append(compute_metrics(name, motion))
        plot_problems(name, motion, os.path.join(FIG_DIR, f"{safe_name(name)}_raw.png"))

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 40)
    print(table.to_string(index=False))

    table.to_csv(os.path.join(OUT_DIR, "problem_metrics_raw.csv"),
                 index=False, encoding="utf-8-sig")
    print(f"\nSaved metrics → {OUT_DIR}/problem_metrics_raw.csv")
    print(f"Saved figures → {FIG_DIR}")


if __name__ == "__main__":
    main()