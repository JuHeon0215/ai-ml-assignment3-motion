# scripts/a3_3_compare.py
# Assignment 3 - Step 3/4: raw vs post-processed (+ smoothing bonus)

import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from a3_2_problem_analysis import (
    compute_metrics, lowest_foot_height, jitter, safe_name,
    PROCESSED_DIR, OUT_DIR, FIG_DIR,
)

VARIANTS = ["raw", "target", "target+MA5"]


def moving_average(motion, window=5):
    """Centered moving average over time. motion: (T, 33, 3)."""
    pad = window // 2
    padded = np.pad(motion, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    return sum(padded[i:i + len(motion)] for i in range(window)) / window


def hip_height(motion):
    return motion[:, [23, 24], 1].mean(axis=1)


def load_variants(name):
    raw_p = os.path.join(PROCESSED_DIR, f"{name}_raw.npy")
    tgt_p = os.path.join(PROCESSED_DIR, f"{name}_target.npy")
    if not (os.path.exists(raw_p) and os.path.exists(tgt_p)):
        return None
    raw = np.load(raw_p)
    tgt = np.load(tgt_p)
    return {"raw": raw, "target": tgt, "target+MA5": moving_average(tgt, 5)}


def plot_compare(name, variants, path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    fig.suptitle(f"Before / after post-processing: {safe_name(name)}")

    for v in ("raw", "target"):
        axes[0].plot(lowest_foot_height(variants[v]), label=v)
        axes[1].plot(hip_height(variants[v]), label=v)
    axes[0].set_title("Lowest foot height (m)")
    axes[1].set_title("Hip height (m)")

    for v in VARIANTS:
        axes[2].plot(np.arange(1, len(variants[v]) - 1), jitter(variants[v]), label=v)
    axes[2].set_title("Jitter (mean joint acceleration)")
    axes[2].set_xlabel("frame")

    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    names = sorted(
        os.path.basename(p)[: -len("_raw.npy")]
        for p in glob.glob(os.path.join(PROCESSED_DIR, "*_raw.npy"))
    )

    rows = []
    for name in names:
        variants = load_variants(name)
        if variants is None:
            print(f"Skip (missing raw/target): {name}")
            continue

        for v in VARIANTS:
            m = compute_metrics(name, variants[v])
            m["video"] = safe_name(name)[:22]
            m["variant"] = v
            m["hip_mean_y"] = round(float(hip_height(variants[v]).mean()), 3)
            rows.append(m)

        plot_compare(name, variants, os.path.join(FIG_DIR, f"{safe_name(name)}_compare.png"))

    cols = ["video", "variant", "foot_mean_y", "hip_mean_y",
            "foot_std", "jitter_mean", "jitter_p95", "bone_len_cv"]
    table = pd.DataFrame(rows)[cols]

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    print(table.to_string(index=False))

    table.to_csv(os.path.join(OUT_DIR, "compare_metrics.csv"),
                 index=False, encoding="utf-8-sig")
    print(f"\nSaved → {OUT_DIR}/compare_metrics.csv")
    print(f"Saved figures → {FIG_DIR}")


if __name__ == "__main__":
    main()