"""Publish checkpoints to a PUBLIC Hugging Face model repo so the notebook's demo mode can download them.

    huggingface-cli login            # once, with a write token
    python scripts/06_hf_upload.py <user>/aip304-cw2-plantseg-option1

Uploads checkpoints/<run>.pt for the final configuration and the planned comparisons E0-E3 (3 seeds each)
plus a model card. Then set cw2/inference.py HF_REPO_ID to the same repo id.
"""
import json
import sys

from _common import P
from huggingface_hub import HfApi

from cw2.experiments import COMPARISONS, SEEDS

repo_id = sys.argv[1]
sel = json.load(open(P.summary / "selection.json"))
runs = [f"{n}_s{s}" for n in COMPARISONS + [sel["final_config"]] for s in SEEDS]
missing = [r for r in runs if not (P.checkpoints / f"{r}.pt").exists()]
assert not missing, f"missing checkpoints: {missing}"

card = f"""---
license: cc-by-nc-4.0
tags: [image-classification, plant-disease, pytorch, coursework]
---
# AIP304 CW2 Option 1 - PlantSeg v7 nine-host classifier checkpoints

PyTorch `state_dict`s (torchvision architectures, 43-way head) trained on the PlantSeg v7 nine-host subset
(Wei et al., 2025, Zenodo 17719108; images CC BY-NC / CC0 by their sources, hence non-commercial licence here).
Final system: 3-seed ensemble of `{sel['final_config']}` (test macro-F1 0.805). Also the planned comparisons E0-E3.
Load with `cw2.inference.load_model` from the coursework repository; preprocessing is described there.
"""
api = HfApi()
api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo_id)
for r in runs:
    api.upload_file(path_or_fileobj=str(P.checkpoints / f"{r}.pt"), path_in_repo=f"checkpoints/{r}.pt", repo_id=repo_id)
    print("uploaded", r, flush=True)
print(f"done: https://huggingface.co/{repo_id}")
