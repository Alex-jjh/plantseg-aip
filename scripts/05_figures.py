"""Regenerate every report figure into results/figures/ from committed results."""
import json

from _common import P

from cw2.report import all_figures

all_figures(P, json.load(open(P.summary / "selection.json")))
print("figures ->", P.figures)
