"""Figures for the notebook and report (static matplotlib, light surface).

Palette: validated categorical slots 1-3 (blue, orange, aqua; all-pairs CVD-safe), sequential blue
for magnitude. Aqua is below 3:1 contrast on the surface, so every multi-series chart carries a legend
and the notebook shows the matching table. One y-scale per axes; text stays in ink colours.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = [BLUE, ORANGE, AQUA]
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def save(fig, path: Path | None):
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    return fig


def describe(rec: dict) -> str:
    """Human-readable config label from a run record."""
    parts = [rec["arch"].replace("_", "-")]
    parts.append("fine-tune" if rec["finetune"] else "frozen")
    if rec["augment"]:
        parts.append("aug")
    if rec["class_weights"]:
        parts.append("class-weighted")
    if rec["label_smoothing"]:
        parts.append(f"LS {rec['label_smoothing']}")
    if rec["input_size"] != 224:
        parts.append(f"{rec['input_size']}px")
    if rec["finetune"] and rec["lr_backbone"] != 1e-4:
        parts.append(f"lr-bb {rec['lr_backbone']:g}")
    if rec["epochs"] != 30:
        parts.append(f"{rec['epochs']} ep")
    return " · ".join(parts)


def class_counts(man: pd.DataFrame, path=None):
    t = man.pivot_table(index="class_name", columns="partition", values="image_id", aggfunc="count", fill_value=0)
    t = t[["train", "val", "test"]]
    t = t.loc[t.sum(1).sort_values().index]
    fig, ax = plt.subplots(figsize=(7.2, 8.6))
    left = np.zeros(len(t))
    for col, c in zip(t.columns, SERIES):
        ax.barh(t.index, t[col], left=left, color=c, edgecolor=SURFACE, linewidth=1, height=0.75, label=col)
        left += t[col].to_numpy()
    for y, tot in enumerate(left):
        ax.text(tot + 2, y, f"{int(tot)}", va="center", fontsize=7, color=INK2)
    ax.set_xlabel("images (after exclusions)")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", visible=False)
    ax.margins(y=0.005)
    ax.set_title(f"Images per host–disease class and partition (n = {len(man):,}, 43 classes)")
    ax.legend(loc="lower right", title="partition")
    return save(fig, path)


def image_grid(items: list[tuple[Path, str]], ncols: int, title: str, path=None, cell=1.55, cap_h=0.45):
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell * ncols, (cell + cap_h) * nrows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (p, cap) in zip(np.atleast_1d(axes).ravel(), items):
        with Image.open(p) as im:
            ax.imshow(im.convert("RGB").resize((224, 224)))
        cap = "\n".join(textwrap.fill(line, 30) for line in cap.split("\n"))
        ax.set_title(cap, fontsize=6.5, fontweight="normal", loc="center", color=INK)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return save(fig, path)


def comparison(table: pd.DataFrame, path=None):
    """Grouped bars: validation vs test macro-F1 (3-seed mean +- sd) per system."""
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(table))
    w = 0.38
    for i, (col, sd, lab, c) in enumerate([("val_macro_f1", "val_sd", "validation", BLUE),
                                           ("test_macro_f1", "test_sd", "test", ORANGE)]):
        ax.bar(x + (i - 0.5) * w, table[col], w, yerr=table[sd].fillna(0), color=c, edgecolor=SURFACE,
               linewidth=1, error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 2}, label=lab)
    for xi, v in zip(x, table["test_macro_f1"]):
        ax.text(xi + 0.5 * w, v + 0.025, f"{v:.3f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(x, table["label"], fontsize=7.5)
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncols=2)
    ax.set_title("Baselines, controlled changes and final system (error bars: sd over 3 seeds)")
    return save(fig, path)


def sweep(summary: pd.DataFrame, labels: dict, final: str, final_note: str, path=None):
    s = summary.sort_values("val_macro_f1")
    fig, ax = plt.subplots(figsize=(7.2, 0.24 * len(s) + 1.0))
    y = np.arange(len(s))
    ax.errorbar(s["val_macro_f1"], y, xerr=s["sd"], fmt="o", ms=5, color=BLUE, ecolor=BLUE, elinewidth=1.2,
                capsize=0, markeredgecolor=SURFACE, markeredgewidth=1)
    ax.set_yticks(y, [labels[n] for n in s.index], fontsize=7)
    i = list(s.index).index(final)
    ax.annotate(final_note, (s["val_macro_f1"].iloc[i], i), xytext=(10, -2), textcoords="offset points",
                fontsize=7, color=INK, va="center")
    ax.set_xlabel("validation macro-F1 (mean ± sd over 3 seeds)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Validation results of all 22 configurations (selection uses validation only)")
    return save(fig, path)


def curves(runs_dir: Path, config: str, seeds=(0, 1, 2), path=None):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for s, c in zip(seeds, SERIES):
        h = pd.read_csv(runs_dir / f"{config}_s{s}" / "hist.csv")
        best = json.load(open(runs_dir / f"{config}_s{s}" / "run.json"))["best_epoch"]
        a1.plot(h.epoch, h.val_macro_f1, color=c, lw=2, label=f"seed {s}")
        a1.plot(best, h.val_macro_f1[h.epoch == best].iloc[0], "o", ms=7, color=c, markeredgecolor=SURFACE, markeredgewidth=1.5)
        a2.plot(h.epoch, h.train_loss, color=c, lw=2, label=f"seed {s} train")
        a2.plot(h.epoch, h.val_loss, color=c, lw=1.2, ls="--", label=f"seed {s} val")
    a1.set_xlabel("epoch"), a1.set_ylabel("validation macro-F1"), a1.set_title("validation macro-F1 (dot = selected checkpoint)", fontsize=8.5)
    a2.set_xlabel("epoch"), a2.set_ylabel("cross-entropy loss"), a2.set_title("loss (solid train, dashed validation)", fontsize=8.5)
    a1.legend(fontsize=7)
    fig.suptitle("Training curves of the final configuration", x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return save(fig, path)


def confusion(conf: np.ndarray, classes: list[str], subset: list[int] | None = None, title="", path=None):
    """Row-normalised confusion (rows = true, columns = predicted); counts printed where > 0."""
    idx = list(range(len(classes))) if subset is None else subset
    c = conf[np.ix_(idx, idx)]
    rown = conf[idx].sum(1, keepdims=True)
    norm = c / np.maximum(rown, 1)
    n = len(idx)
    size = max(4.5, 0.19 * n + 2.2)
    fig, ax = plt.subplots(figsize=(size + 1.0, size))
    im = ax.imshow(norm, cmap=SEQ_BLUE, vmin=0, vmax=1)
    names = [classes[i] for i in idx]
    fs = 6 if n > 20 else 7
    ax.set_xticks(range(n), names, rotation=90, fontsize=fs)
    ax.set_yticks(range(n), [f"{nm} (n={int(r)})" for nm, r in zip(names, rown.ravel())], fontsize=fs)
    ax.grid(False)
    if n <= 20:
        for i in range(n):
            for j in range(n):
                if c[i, j]:
                    ax.text(j, i, int(c[i, j]), ha="center", va="center", fontsize=6.5,
                            color="#ffffff" if norm[i, j] > 0.5 else INK)
    ax.set_xlabel("predicted class"), ax.set_ylabel("true class")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("share of true-class test images", color=INK2)
    ax.set_title(title)
    return save(fig, path)


def perclass_f1(pc: pd.DataFrame, train_support: pd.Series, path=None):
    """Per-class test F1 (final system), sorted; label shows test support and training count."""
    p = pc.sort_values("f1")
    fig, ax = plt.subplots(figsize=(7.2, 8.0))
    ax.barh(p["class"], p["f1"], color=BLUE, edgecolor=SURFACE, linewidth=1, height=0.72)
    for y, (f, s, c) in enumerate(zip(p["f1"], p["support"], p["class"])):
        ax.text(f + 0.01, y, f"{f:.2f}  (test {int(s)}, train {int(train_support[c])})", va="center", fontsize=6.5, color=INK2)
    ax.set_xlim(0, 1.25)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", visible=False)
    ax.margins(y=0.005)
    ax.set_xlabel("test F1")
    ax.set_title("Per-class test F1 of the final system")
    return save(fig, path)
