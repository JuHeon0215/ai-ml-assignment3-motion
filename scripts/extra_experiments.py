# scripts/extra_experiments.py
# Bonus experiments for Assignment 3 (Taegeuk 1 jang):
#   A  smoothing window size      B  low-visibility interpolation      C  bone-length correction
# Also prints per-joint-group jitter and the effect of low visibility.
#
# Usage (from the repo root):  python scripts/extra_experiments.py 태극_1장
# Needs: data/processed/<name>_target.npy. B and the visibility analysis also need the
# raw .npz (data/raw_keypoints/ or data/processed/); without it they are skipped.

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Tee:                             # print to the console and keep a copy for the text report
    def __init__(self, stream):
        self.stream, self.lines = stream, []

    def write(self, s):
        self.stream.write(s)
        self.lines.append(s)

    def flush(self):
        self.stream.flush()


tee = Tee(sys.stdout)
sys.stdout = tee

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
name = sys.argv[1] if len(sys.argv) > 1 else "태극_1장"
raw_npz = os.path.join(ROOT, "data", "raw_keypoints", f"{name}.npz")
tgt_npy = os.path.join(ROOT, "data", "processed", f"{name}_target.npy")
# the repo only ships the processed sample; fall back to a local copy of the npz
if not os.path.exists(raw_npz):
    raw_npz = os.path.join(ROOT, "data", "processed", f"{name}.npz")

t = np.load(tgt_npy)                  # (T, 33, 3) after p1_2
HAS_VIS = os.path.exists(raw_npz)     # B and the visibility analysis need the .npz
if HAS_VIS:
    d = np.load(raw_npz, allow_pickle=True)["data"]
    vis = d[:, 5].reshape(len(t), 33)  # column 5 = visibility


def ma(x, k=5):
    if k == 1:
        return x
    p = k // 2
    xp = np.pad(x, ((p, p), (0, 0), (0, 0)), mode="edge")
    return np.stack([xp[i:i + len(x)] for i in range(k)]).mean(0)


def jit(x):                            # |2nd difference| per joint and frame
    return np.linalg.norm(x[2:] - 2 * x[1:-1] + x[:-2], axis=-1)


LEFT_BONES = [(11, 13), (13, 15), (23, 25), (25, 27)]   # same 4 bones as a3_2


def bone_cv(x, bones=LEFT_BONES):
    out = []
    for a, b in bones:
        L = np.linalg.norm(x[:, a] - x[:, b], axis=1)
        out.append(L.std() / L.mean())
    return float(np.mean(out))


def interp_lowvis(x, v, th=0.5):       # Experiment B
    y = x.copy()
    f = np.arange(len(x))
    for j in range(33):
        ok = v[:, j] >= th
        if ok.all() or ok.sum() < 2:
            continue
        for c in range(3):
            y[:, j, c] = np.interp(f, f[ok], x[ok, j, c])
    return y


def fix_bones(x):                      # Experiment C: rescale limb bones to median length
    y = x.copy()
    chains = [(11, 13, 15), (12, 14, 16), (23, 25, 27), (24, 26, 28)]
    extra = {15: (17, 19, 21), 16: (18, 20, 22), 27: (29, 31), 28: (30, 32)}
    for p, m, e in chains:
        for a, b in ((p, m), (m, e)):
            L = np.median(np.linalg.norm(x[:, a] - x[:, b], axis=1))
            dv = y[:, b] - y[:, a]
            n = np.linalg.norm(dv, axis=1, keepdims=True) + 1e-9
            shift = y[:, a] + dv / n * L - y[:, b]
            y[:, b] += shift
            if b == m:
                y[:, e] += shift
            else:
                for k in extra.get(e, ()):
                    y[:, k] += shift
    return y


def shift_cm(a):                       # mean joint displacement vs. target, all joints
    return float(np.linalg.norm(a - t, axis=-1).mean() * 100)


print("== A. window size ==")
print("window  jitter  mean_shift_cm  p95_shift_cm")
for k in (1, 3, 5, 9, 15):
    m = ma(t, k)
    dd = np.linalg.norm(m - t, axis=-1)
    print(f"{k:>6}  {jit(m).mean():.4f}  {dd.mean()*100:6.2f}  {np.percentile(dd, 95)*100:6.2f}")

print("\n== B / C pipelines ==")
print("pipeline              jitter  bone_cv  shift_cm")
pipes = [("target", t), ("MA5", ma(t))]
if HAS_VIS:
    pipes.append(("B lowvis+MA5", ma(interp_lowvis(t, vis))))
pipes.append(("C MA5+bonefix", fix_bones(ma(t))))
for label, x in pipes:
    print(f"{label:<20}  {jit(x).mean():.4f}  {bone_cv(x):.3f}   {shift_cm(x):.2f}")
D = fix_bones(ma(t))
print("p95 joint jitter  MA5:", round(float(np.percentile(jit(ma(t)), 95)), 4), " C:", round(float(np.percentile(jit(D), 95)), 4),
      "| joint-frames below floor (< -1 cm):", round(float((D[:, :, 1] < -0.01).mean()) * 100, 2), "%",
      "| lowest y:", round(float(D[:, :, 1].min()), 3))

print("\n== jitter by joint group (target) ==")
j = jit(t).mean(0)
groups = {"hands 17-22": j[17:23], "wrists 15,16": j[[15, 16]], "elbows 13,14": j[[13, 14]],
          "feet 29-32": j[29:33], "ankles 27,28": j[[27, 28]], "knees 25,26": j[[25, 26]],
          "shoulders 11,12": j[[11, 12]], "hips 23,24": j[[23, 24]]}
for k, v in groups.items():
    print(f"{k:<16}{v.mean():.4f}")

if HAS_VIS:
    print("\n== visibility ==")
    jj, vv = jit(t), vis[1:-1]
    low = vv < 0.5
    print(f"low-visibility share: {low.mean()*100:.1f}%  jitter low={jj[low].mean():.3f}  high={jj[~low].mean():.3f}")

fa = np.linalg.norm(t[:, 13] - t[:, 15], axis=1)
print(f"left forearm length: min={fa.min():.3f} max={fa.max():.3f} cv={fa.std()/fa.mean():.3f}")

# ---- save results (text report + forearm-length figure for Experiment C) ----
out_dir = os.path.join(ROOT, "data", "analysis")
os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
with open(os.path.join(out_dir, f"extra_{name}.txt"), "w", encoding="utf-8") as f:
    f.write("".join(tee.lines))
fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=200)
ax.plot(fa, color="#E07B00", lw=1, label="target (forearm length varies)")
ax.plot(np.linalg.norm(D[:, 13] - D[:, 15], axis=1), color="#006CB8", lw=1.4, label="MA5 + bone-length correction")
ax.set_xlabel("frame")
ax.set_ylabel("left forearm length (m)")
ax.set_title(f"Experiment C: bone-length correction ({name})", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "figures", f"extra_{name}_forearm.png"))
sys.stdout = tee.stream
print("saved:", os.path.join("data", "analysis", f"extra_{name}.txt"), "and figures/extra_%s_forearm.png" % name)
