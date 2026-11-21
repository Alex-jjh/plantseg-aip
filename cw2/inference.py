"""Load trained checkpoints and predict from image files (RGB only - no masks, no metadata).

Preprocessing matches evaluation exactly: whole photo resized to the cache side with PIL bilinear,
then resized to the model input side and ImageNet-normalised (see train.to_input).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .train import Config, make_model, to_input

HF_REPO_ID = "TODO-hf-user/aip304-cw2-plantseg-option1"  # public Hugging Face model repo with checkpoints/


def checkpoint_path(run: str, ckpt_dir: Path, repo_id: str = HF_REPO_ID) -> Path:
    """Local checkpoint if present, otherwise download checkpoints/<run>.pt from the public HF repo."""
    p = ckpt_dir / f"{run}.pt"
    if p.exists():
        return p
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo_id, filename=f"checkpoints/{run}.pt", local_dir=ckpt_dir.parent))


def load_model(cfg: Config, ckpt: Path, n_classes: int, device) -> torch.nn.Module:
    model, _ = make_model(n_classes, cfg.arch)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.to(device).eval()


def load_images(paths: list[Path], cache_size: int) -> torch.Tensor:
    out = torch.empty((len(paths), 3, cache_size, cache_size), dtype=torch.uint8)
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            im = im.convert("RGB").resize((cache_size, cache_size), Image.BILINEAR)
        out[i] = torch.from_numpy(np.array(im)).permute(2, 0, 1)
    return out


@torch.no_grad()
def predict_proba(models: list[torch.nn.Module], imgs: torch.Tensor, input_size: int, device) -> torch.Tensor:
    """Mean softmax over models (one model = single-model prediction)."""
    x = to_input(imgs, device, input_size)
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        return torch.stack([m(x).float().softmax(1) for m in models]).mean(0).cpu()
