"""Run sweep stages (default: all). Several copies can share one GPU (claim queue):
    python scripts/02_sweep.py [stage0 stage1 ...]   # run in N terminals for N workers
"""
import sys

import torch
from _common import P

from cw2.experiments import load_spec, run_sweep

stages = sys.argv[1:] or ["stage0", "stage1", "stage2", "stage3"]
torch.set_num_threads(2)
run_sweep(load_spec(P, stages), P, P.runs, torch.device("cuda" if torch.cuda.is_available() else "cpu"),
          log=lambda m: print(m, flush=True))
