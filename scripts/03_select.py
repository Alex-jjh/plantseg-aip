"""VALIDATION ONLY: summarise all runs and freeze the final choice in results/summary/selection.json."""
from _common import P

from cw2.experiments import select_final, summarize_val

s = summarize_val(P.runs)
P.summary.mkdir(parents=True, exist_ok=True)
s.to_csv(P.summary / "sweep_summary_val.csv")
print(s.round(4).to_string())
sel = select_final(P, P.runs, P.summary / "selection.json")
print("frozen selection:", {k: sel[k] for k in ("final_config", "final_kind", "final_val_macro_f1", "demo_seed", "frozen_at")})
