"""Colour-vision-deficiency check (Machado et al. 2009, severity 1.0) of the shared palette and of every figure. Writes cvd/ next to the figures. Reports only; changes nothing."""

import itertools
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import plot_style as P

OUT = Path(sys.argv[1])
(OUT / "cvd").mkdir(exist_ok=True)
M = {
    "deuteranopia": np.array(
        [
            [0.367322, 0.860646, -0.227968],
            [0.280085, 0.672501, 0.047413],
            [-0.011820, 0.042940, 0.968881],
        ]
    ),
    "protanopia": np.array(
        [
            [0.152286, 1.052583, -0.204868],
            [0.114503, 0.786281, 0.099216],
            [-0.003882, -0.048116, 1.051998],
        ]
    ),
}


def lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def gam(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def sim(rgb, k):
    return gam(lin(rgb) @ M[k].T)


def lab(rgb):
    l = lin(rgb)
    xyz = (
        l
        @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
        / np.array([0.95047, 1, 1.08883])
    )
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def hx(h):
    return np.array([int(h[i : i + 2], 16) / 255 for i in (1, 3, 5)])


def de(a, b):
    return float(np.linalg.norm(lab(a) - lab(b)))


sets = {
    "rooms": {r: P.ROOM_COL[r] for r in P.ROOMS},
    "blocks": P.COL,
    "feature-block lines (fig6b)": {k: P.COL[k] for k in ("morph", "audio_bio", "yolo")},
    "gap recovery / age-block": {
        "morph/backward": "#1b4965",
        "all-sensor/forward": "#d1495b",
        "internal": "#8bbdd9",
    },
    "negative controls": P.NEG_CMAP,
    "rooms + pooled": {**P.ROOM_COL, "pooled": P.POOLED_COL},
}
rows = []
for sname, d in sets.items():
    for (a, ca), (b, cb) in itertools.combinations(d.items(), 2):
        r = dict(set=sname, a=a, b=b, normal=de(hx(ca), hx(cb)))
        for k in M:
            r[k] = de(sim(hx(ca), k), sim(hx(cb), k))
        rows.append(r)
import pandas as pd

df = pd.DataFrame(rows).round(1)
df.to_csv(OUT / "cvd" / "cvd_colour_pairs.csv", index=False)
print(df.to_string())
for p in sorted(OUT.glob("fig*.png")):
    im = Image.open(p).convert("RGB")
    im.thumbnail((1800, 1800))
    a = np.asarray(im) / 255.0
    tiles = [a] + [sim(a.reshape(-1, 3), k).reshape(a.shape) for k in M]
    Image.fromarray((np.concatenate(tiles, 1) * 255).astype(np.uint8)).save(
        OUT / "cvd" / f"{p.stem}_normal_deuter_prot.png"
    )
