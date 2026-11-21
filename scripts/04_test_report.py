"""TEST: evaluate the frozen selection once and write results/summary/test_*.csv."""
import json

from _common import P

from cw2.data import load_kept
from cw2.experiments import save_test_report, test_report

sel = json.load(open(P.summary / "selection.json"))
_, classes = load_kept(P.data_results)
rep = test_report(P, P.runs, sel)
save_test_report(rep, classes, P.summary)
print(rep["table"].round(4).to_string(index=False))
print("\npost-freeze supplementary (not used for selection):\n", rep["supplementary"].round(4).to_string(index=False))
