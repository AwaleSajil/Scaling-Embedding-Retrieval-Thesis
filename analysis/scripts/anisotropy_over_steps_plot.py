"""
Plot from spectra.npz produced by anisotropy_over_steps.py.

Fig 1 (anisotropy_steps_spectrum_final.png): variance-share + cumulative-90%
       spectrum at the final step, 3 variants, PRE-SIGN float (corrected repro
       of the original cross-backbone figure).
Fig 2 (anisotropy_steps_effdim.png): effective-dim (participation ratio) vs
       training step, 3 variants, float (solid) + sign (dashed).
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/nas/rhome/sawale/thesis")
OUT = ROOT / "analysis/figures/anisotropy_steps"
D = 768
FINAL = 16378
COLORS = {"baseline": "#d62728", "BAT-STE": "#1f77b4", "atanh": "#2ca02c"}

dat = np.load(OUT / "spectra.npz", allow_pickle=True)
meta = [m.split("|") for m in dat["meta"]]   # variant, step, prf, fr, ps
rows = [(v, int(s), float(prf), float(fr), float(ps)) for v, s, prf, fr, ps in meta]
variants = ["baseline", "BAT-STE", "atanh"]


def lam(variant, step, kind):
    return dat[f"{variant}@{step}@{kind}"]


# ---------- Fig 1: spectrum at final step, float (solid) vs sign (dashed) ----------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5))
for v in variants:
    for kind, ls in (("float", "-"), ("sign", "--")):
        l = lam(v, FINAL, kind)
        share = np.sort(l / l.sum())[::-1]
        pr = (l.sum() ** 2) / np.sum(l ** 2)
        axL.plot(share, color=COLORS[v], lw=1.7, ls=ls,
                 label=f"{v} {kind} (eff-dim≈{pr:.0f})")
        cum = np.cumsum(np.sort(l)[::-1]) / l.sum()
        axR.plot(np.arange(1, D + 1), cum, color=COLORS[v], lw=1.7, ls=ls)
axL.set_yscale("log"); axL.set_xlabel("dimension rank")
axL.set_ylabel("variance share (log)")
axL.set_title("Variance spectrum at final step\nsolid = pre-sign float, dashed = sign() output")
axL.legend(fontsize=8)
axR.axhline(0.9, color="gray", ls=":", lw=1); axR.text(540, 0.91, "90% of variance", color="gray")
axR.set_xlabel("number of top dimensions"); axR.set_ylabel("cumulative variance")
axR.set_title("How many dims to reach 90% variance\n(flatter / further right = more isotropic)")
fig.tight_layout(); fig.savefig(OUT / "anisotropy_steps_spectrum_final.png", dpi=150)
print("saved", OUT / "anisotropy_steps_spectrum_final.png")

# ---------- Fig 2: eff-dim vs step ----------
fig, ax = plt.subplots(figsize=(8, 5))
for v in variants:
    rs = sorted([r for r in rows if r[0] == v], key=lambda r: r[1])
    steps = [r[1] for r in rs]
    eff_f = [r[2] for r in rs]
    eff_s = [r[4] for r in rs]
    ax.plot(steps, eff_f, "-o", color=COLORS[v], lw=1.8, ms=4, label=f"{v} (float)")
    ax.plot(steps, eff_s, "--s", color=COLORS[v], lw=1.4, ms=3, alpha=0.7,
            label=f"{v} (sign)")
ax.set_xlabel("training step"); ax.set_ylabel("effective dim (participation ratio /768)")
ax.set_title("Effective dimensionality over training\nfloat = pre-sign geometry, sign = binarized output")
ax.legend(fontsize=8, ncol=3); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "anisotropy_steps_effdim.png", dpi=150)
print("saved", OUT / "anisotropy_steps_effdim.png")
