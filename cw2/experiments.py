"""Experiment runner, validation-only selection and the one-off test report.

Every run is <config>_s<seed>. A run writes its light record to results/runs/<run>/
(run.json, hist.csv, logits.npz with val+test logits) and its weights to outputs/checkpoints/<run>.pt.
Selection reads only validation fields; the test report runs only after selection.json is frozen.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import load_kept
from .metrics import evaluate
from .paths import Paths
from .train import Config, build_cache, predict, train_run

SEEDS = (0, 1, 2)
COMPARISONS = ["E0_frozen", "E1_finetune", "E2_finetune_cw", "E3_finetune_aug"]  # planned controlled changes
K = ("macro_f1", "accuracy", "balanced_accuracy")


def load_spec(paths: Paths, stages=("stage0", "stage1", "stage2", "stage3")) -> dict:
    """Configurations of the given stages, minus those pruned on validation (sweeps/pruned.json)."""
    pruned = json.load(open(paths.sweep("pruned")))
    spec = {}
    for s in stages:
        spec.update({k: v for k, v in json.load(open(paths.sweep(s))).items() if k not in pruned})
    return spec


def run_job(name: str, kw: dict, seed: int, paths: Paths, runs_dir: Path, caches: dict, device,
            ckpt_dir: Path | None = None) -> dict:
    man, classes = load_kept(paths.data_results)
    cfg = Config(name=name, seed=seed, **kw)
    if cfg.cache_size not in caches:
        caches[cfg.cache_size] = build_cache(man, paths.data_root, paths.cache(cfg.cache_size), cfg.cache_size)
    cache = caches[cfg.cache_size]
    r = train_run(cfg, cache, classes, device)
    lv = predict(r["model"], cache["val"][0], device, size=cfg.input_size)
    lt = predict(r["model"], cache["test"][0], device, size=cfg.input_size)
    d = runs_dir / f"{name}_s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "logits.npz", val=lv.numpy(), test=lt.numpy())
    r["hist"].to_csv(d / "hist.csv", index=False)
    ckpt_dir = ckpt_dir or paths.checkpoints
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(r["model"].state_dict(), ckpt_dir / f"{name}_s{seed}.pt")
    vm = evaluate(cache["val"][1].numpy(), lv.argmax(1).numpy(), classes)
    rec = {**r["cfg"], "best_epoch": r["best_epoch"], "train_seconds": round(r["train_seconds"]),
           "val_macro_f1": vm["macro_f1"], "val_accuracy": vm["accuracy"], "val_balanced_accuracy": vm["balanced_accuracy"],
           "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}
    json.dump(rec, open(d / "run.json", "w"), indent=1)
    return rec


def run_sweep(spec: dict, paths: Paths, runs_dir: Path, device, log=print, ckpt_dir: Path | None = None) -> None:
    """Claim-queue: a worker claims a run by creating its directory, so several processes can share
    one GPU. Finished, running or skipped (SKIPPED.txt) runs are never started twice."""
    caches: dict = {}
    for name, kw in spec.items():
        for seed in SEEDS:
            d = runs_dir / f"{name}_s{seed}"
            try:
                d.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                continue
            rec = run_job(name, kw, seed, paths, runs_dir, caches, device, ckpt_dir)
            log(f"{name} s{seed}: val macro-F1 {rec['val_macro_f1']:.4f} (epoch {rec['best_epoch']}, {rec['train_seconds']}s)")


def load_runs(runs_dir: Path) -> pd.DataFrame:
    return pd.DataFrame([json.load(open(f)) for f in sorted(runs_dir.glob("*/run.json"))])


def logits(runs_dir: Path, run: str, part: str) -> torch.Tensor:
    return torch.tensor(np.load(runs_dir / run / "logits.npz")[part])


def summarize_val(runs_dir: Path) -> pd.DataFrame:
    """Per config: 3-seed mean/sd of validation metrics (never reads test)."""
    df = load_runs(runs_dir)
    g = df.groupby("name").agg(n=("seed", "count"), val_macro_f1=("val_macro_f1", "mean"), sd=("val_macro_f1", "std"),
                               val_accuracy=("val_accuracy", "mean"), val_balanced_accuracy=("val_balanced_accuracy", "mean"),
                               best_epoch=("best_epoch", "mean"), train_seconds=("train_seconds", "mean"),
                               arch=("arch", "first"), input_size=("input_size", "first"))
    return g.sort_values("val_macro_f1", ascending=False)


def select_final(paths: Paths, runs_dir: Path, sel_file: Path, n_ensembles: int = 5) -> dict:
    """VALIDATION ONLY. Candidates: every complete config (3-seed mean val macro-F1) and the 3-seed
    softmax-averaged ensemble of the top `n_ensembles` configs. Highest validation macro-F1 wins.
    The rule was fixed before results were seen; the decision is frozen in `sel_file` and never
    recomputed once test results exist."""
    if sel_file.exists():
        return json.load(open(sel_file))
    man, classes = load_kept(paths.data_results)
    yv = man[man.partition == "val"]["class_id"].to_numpy()
    cfg = summarize_val(runs_dir)
    cfg = cfg[cfg["n"] == len(SEEDS)]
    cands = [{"candidate": n, "kind": "single (3-seed mean)", "val_macro_f1": r.val_macro_f1, "sd": r.sd}
             for n, r in cfg.iterrows()]
    for name in cfg.index[:n_ensembles]:
        probs = sum(logits(runs_dir, f"{name}_s{s}", "val").softmax(1) for s in SEEDS)
        cands.append({"candidate": name, "kind": "ensemble (3 seeds)",
                      "val_macro_f1": evaluate(yv, probs.argmax(1).numpy(), classes)["macro_f1"], "sd": None})
    cands = sorted(cands, key=lambda c: -c["val_macro_f1"])
    best = cands[0]
    runs = load_runs(runs_dir)
    demo_seed = int(runs[runs.name == best["candidate"]].sort_values("val_macro_f1", ascending=False).iloc[0].seed)
    sel = {"final_config": best["candidate"], "final_kind": best["kind"], "final_val_macro_f1": best["val_macro_f1"],
           "demo_seed": demo_seed, "comparisons": COMPARISONS,
           "frozen_at": datetime.now(timezone.utc).isoformat(), "candidates": cands}
    sel_file.parent.mkdir(parents=True, exist_ok=True)
    json.dump(sel, open(sel_file, "w"), indent=1, default=float)
    return sel


def final_probs(runs_dir: Path, sel: dict, part: str) -> torch.Tensor:
    """Class probabilities of the selected final system on a partition."""
    n = sel["final_config"]
    if sel["final_kind"].startswith("ensemble"):
        return sum(logits(runs_dir, f"{n}_s{s}", part).softmax(1) for s in SEEDS) / len(SEEDS)
    return logits(runs_dir, f"{n}_s{sel['demo_seed']}", part).softmax(1)


def test_report(paths: Paths, runs_dir: Path, sel: dict) -> dict:
    """TEST - only after selection is frozen. Majority baseline, planned comparisons (3-seed mean+-sd),
    the final system, its per-class table/confusion/predictions, and a post-freeze supplementary table
    of the top validation candidates (reported for transparency, never used to re-select)."""
    man, classes = load_kept(paths.data_results)
    ytr = man[man.partition == "train"]["class_id"].to_numpy()
    yt = man[man.partition == "test"]["class_id"].to_numpy()
    rows = []
    maj = evaluate(yt, np.full_like(yt, np.bincount(ytr).argmax()), classes)
    rows.append({"system": "Majority class", **{k: maj[k] for k in K}})
    for n in sel["comparisons"] + [sel["final_config"]]:
        ms = [evaluate(yt, logits(runs_dir, f"{n}_s{s}", "test").argmax(1).numpy(), classes) for s in SEEDS]
        rows.append({"system": f"{n} (3 seeds)", **{k: np.mean([m[k] for m in ms]) for k in K},
                     **{f"{k}_sd": np.std([m[k] for m in ms], ddof=1) for k in K}})
    probs = final_probs(runs_dir, sel, "test")
    final = evaluate(yt, probs.argmax(1).numpy(), classes)
    if sel["final_kind"].startswith("ensemble"):
        rows.append({"system": f"{sel['final_config']} ensemble (final)", **{k: final[k] for k in K}})
    sup = []
    for c in sel["candidates"][:8]:
        n = c["candidate"]
        if not all((runs_dir / f"{n}_s{s}" / "run.json").exists() for s in SEEDS):
            continue  # e.g. a partial rerun that did not retrain this candidate
        if c["kind"].startswith("ensemble"):
            p = sum(logits(runs_dir, f"{n}_s{s}", "test").softmax(1) for s in SEEDS).argmax(1).numpy()
            sup.append({**c, "test_macro_f1": evaluate(yt, p, classes)["macro_f1"], "test_sd": None})
        else:
            f = [evaluate(yt, logits(runs_dir, f"{n}_s{s}", "test").argmax(1).numpy(), classes)["macro_f1"] for s in SEEDS]
            sup.append({**c, "test_macro_f1": float(np.mean(f)), "test_sd": float(np.std(f, ddof=1))})
    test_man = man[man.partition == "test"]
    preds = pd.DataFrame({"image_id": test_man["image_id"].to_numpy(), "path": test_man["path"].to_numpy(),
                          "true": yt, "pred": probs.argmax(1).numpy(), "score": probs.max(1).values.numpy()})
    return {"table": pd.DataFrame(rows), "final": final, "supplementary": pd.DataFrame(sup), "predictions": preds}


def run_config(runs_dir: Path, name: str) -> Config:
    rec = json.load(open(runs_dir / f"{name}_s0" / "run.json"))
    return Config(**{k: rec[k] for k in Config.__dataclass_fields__})


def save_test_report(rep: dict, classes: list[str], summary_dir: Path) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    rep["table"].to_csv(summary_dir / "test_results.csv", index=False)
    rep["final"]["per_class"].to_csv(summary_dir / "test_perclass_final.csv", index=False)
    pd.DataFrame(rep["final"]["confusion"], index=classes, columns=classes).to_csv(summary_dir / "test_confusion_final.csv")
    rep["predictions"].to_csv(summary_dir / "test_predictions_final.csv", index=False)
    rep["supplementary"].to_csv(summary_dir / "test_supplementary_top_candidates.csv", index=False)
