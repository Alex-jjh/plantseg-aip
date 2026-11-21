"""Report figures and tables, built from committed results (shared by the notebook and scripts/05).

Each function returns a matplotlib figure (and saves it under results/figures/ when `save=True`)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import viz
from .data import HOSTS, load_kept
from .experiments import SEEDS, load_runs, logits, summarize_val
from .metrics import evaluate
from .paths import Paths

LABELS = {"E0_frozen": "E0 frozen\nResNet-18", "E1_finetune": "E1 fine-tune", "E2_finetune_cw": "E2 + class\nweights",
          "E3_finetune_aug": "E3 + aug"}


def _out(P: Paths, name: str, save: bool):
    return P.figures / name if save else None


def fig_class_counts(P: Paths, save=True):
    man, _ = load_kept(P.data_results)
    return viz.class_counts(man, _out(P, "fig1_class_counts.png", save))


def fig_samples(P: Paths, save=True):
    """Two training photos per host (different classes, fixed random seed)."""
    man, _ = load_kept(P.data_results)
    tr = man[man.partition == "train"]
    items = []
    for h in HOSTS:
        cl = tr[tr.host.str.lower() == h].drop_duplicates("class_name").sample(2, random_state=0)
        for r in cl.itertuples():
            items.append((P.data_root / r.path, r.class_name.replace(" ", "\n", 1)))
    return viz.image_grid(items, 6, "Representative training photographs (two classes per host)",
                          _out(P, "fig2_samples.png", save))


def fig_conflicts(P: Paths, n_groups=4, save=True):
    """Duplicate photos that carry different labels (excluded from all partitions)."""
    man = pd.read_csv(P.data_results / "manifest.csv").set_index("image_id")
    groups = json.load(open(P.data_results / "audit.json"))["duplicate_groups"]
    conf = [g for g in groups if man.loc[g, "class_id"].nunique() > 1 and len(g) == 2][:n_groups]
    items = [(P.data_root / man.loc[i, "path"], f"{man.loc[i, 'class_name']}\n[{man.loc[i, 'partition']}]")
             for g in conf for i in g]
    return viz.image_grid(items, 4, "Same photo, different labels: duplicate groups excluded by the audit",
                          _out(P, "figA_label_conflicts.png", save), cell=1.75)


def comparison_table(P: Paths, sel: dict) -> pd.DataFrame:
    man, classes = load_kept(P.data_results)
    yt = man[man.partition == "test"]["class_id"].to_numpy()
    yv = man[man.partition == "val"]["class_id"].to_numpy()
    ytr = man[man.partition == "train"]["class_id"].to_numpy()
    maj = np.bincount(ytr).argmax()
    rows = [{"system": "Majority", "label": "Majority\nclass",
             "val_macro_f1": evaluate(yv, np.full_like(yv, maj), classes)["macro_f1"], "val_sd": np.nan,
             "test_macro_f1": evaluate(yt, np.full_like(yt, maj), classes)["macro_f1"], "test_sd": np.nan}]
    for n in sel["comparisons"] + [sel["final_config"]]:
        v = [evaluate(yv, logits(P.runs, f"{n}_s{s}", "val").argmax(1).numpy(), classes)["macro_f1"] for s in SEEDS]
        t = [evaluate(yt, logits(P.runs, f"{n}_s{s}", "test").argmax(1).numpy(), classes)["macro_f1"] for s in SEEDS]
        rows.append({"system": n, "label": LABELS.get(n, "ConvNeXt-T\nper seed"), "val_macro_f1": np.mean(v),
                     "val_sd": np.std(v, ddof=1), "test_macro_f1": np.mean(t), "test_sd": np.std(t, ddof=1)})
    if sel["final_kind"].startswith("ensemble"):
        n = sel["final_config"]
        pv = sum(logits(P.runs, f"{n}_s{s}", "val").softmax(1) for s in SEEDS).argmax(1).numpy()
        pt = sum(logits(P.runs, f"{n}_s{s}", "test").softmax(1) for s in SEEDS).argmax(1).numpy()
        rows.append({"system": n + " ensemble", "label": "Final\nensemble",
                     "val_macro_f1": evaluate(yv, pv, classes)["macro_f1"], "val_sd": np.nan,
                     "test_macro_f1": evaluate(yt, pt, classes)["macro_f1"], "test_sd": np.nan})
    return pd.DataFrame(rows)


def fig_comparison(P: Paths, sel: dict, save=True):
    return viz.comparison(comparison_table(P, sel), _out(P, "fig3_comparison.png", save))


def config_labels(P: Paths) -> dict:
    runs = load_runs(P.runs)
    return {r["name"]: viz.describe(r) for r in runs.drop_duplicates("name").to_dict("records")}


def fig_sweep(P: Paths, sel: dict, save=True):
    s = summarize_val(P.runs)
    note = f"selected as {sel['final_kind'].split(' (')[0]} (val {sel['final_val_macro_f1']:.3f})"
    return viz.sweep(s, config_labels(P), sel["final_config"], note, _out(P, "fig4_sweep_val.png", save))


def fig_curves(P: Paths, sel: dict, save=True):
    return viz.curves(P.runs, sel["final_config"], SEEDS, _out(P, "fig5_curves.png", save))


def worst_confusion_subset(conf: np.ndarray, k_pairs=8) -> list[int]:
    off = conf.copy()
    np.fill_diagonal(off, 0)
    pairs = np.dstack(np.unravel_index(np.argsort(-off.ravel()), off.shape))[0][:k_pairs]
    idx = []
    for i, j in pairs:
        for c in (int(i), int(j)):
            if c not in idx:
                idx.append(c)
    return sorted(idx)


def fig_confusion(P: Paths, full: bool, save=True):
    conf_df = pd.read_csv(P.summary / "test_confusion_final.csv", index_col=0)
    classes, conf = list(conf_df.index), conf_df.to_numpy()
    if full:
        return viz.confusion(conf, classes, None, "Test confusion matrix of the final system (all 43 classes)",
                             _out(P, "figB_confusion_full.png", save))
    return viz.confusion(conf, classes, worst_confusion_subset(conf),
                         "Test confusion among the classes in the 8 most frequent confusions",
                         _out(P, "fig6_confusion_subset.png", save))


def fig_perclass(P: Paths, save=True):
    man, _ = load_kept(P.data_results)
    pc = pd.read_csv(P.summary / "test_perclass_final.csv")
    tr = man[man.partition == "train"].class_name.value_counts()
    return viz.perclass_f1(pc, tr, _out(P, "fig7_perclass_f1.png", save))


def fig_gallery(P: Paths, kind: str, n=12, save=True):
    """kind='success': correct predictions from distinct classes (seeded sample);
    kind='failure': the most confident wrong predictions."""
    _, classes = load_kept(P.data_results)
    pr = pd.read_csv(P.summary / "test_predictions_final.csv")
    if kind == "success":
        sel = pr[pr.true == pr.pred].drop_duplicates("true").sample(n, random_state=0)
    else:
        sel = pr[pr.true != pr.pred].sort_values("score", ascending=False).head(n)
    items = [(P.data_root / r.path, f"true: {classes[r.true]}\npred: {classes[r.pred]} ({r.score:.2f})") for r in sel.itertuples()]
    title = ("Correct test predictions (true / predicted / ensemble probability)" if kind == "success"
             else "Most confident errors on test (true / predicted / ensemble probability)")
    return viz.image_grid(items, 6, title, _out(P, f"fig8_gallery_{kind}.png", save), cell=1.75, cap_h=0.8)


def all_figures(P: Paths, sel: dict) -> None:
    viz.style()
    for f in (fig_class_counts, fig_samples, fig_conflicts, fig_perclass):
        f(P)
    for f in (fig_comparison, fig_sweep, fig_curves):
        f(P, sel)
    fig_confusion(P, full=False), fig_confusion(P, full=True)
    fig_gallery(P, "success"), fig_gallery(P, "failure")
