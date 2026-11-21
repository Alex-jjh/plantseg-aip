"""Training for the Option 1 CNN experiments (ImageNet-pretrained CNNs from torchvision).

Images are decoded once and cached as uint8 256x256 tensors (whole photo squashed, no crop), so
training needs no DataLoader worker processes. Eval transform: resize 256->224 + ImageNet
normalisation (deterministic). Training augmentation (only when cfg.augment): see batch_aug -
random-resized crop with area>=60% (keeps most of the leaf so small lesions are rarely cut out),
flips, rotation up to 20 degrees, mild brightness/contrast. Hue/saturation are NOT jittered
because colour (yellowing, rust, chlorosis) is diagnostic.
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch import nn

from .metrics import evaluate

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CACHE_SIZE, INPUT_SIZE = 256, 224  # defaults; cache side = input side * 8/7 (256 for 224, 320 for 280)


@dataclass
class Config:
    name: str
    finetune: bool = True          # False = frozen backbone, train only the classifier layer
    class_weights: bool = False    # inverse-frequency weighted cross-entropy
    augment: bool = False
    epochs: int = 30
    batch_size: int = 32
    lr_head: float = 1e-3
    lr_backbone: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    arch: str = "resnet18"
    input_size: int = 224
    seed: int = 0

    @property
    def cache_size(self) -> int:
        return self.input_size * 8 // 7


def build_cache(manifest: pd.DataFrame, root: Path, cache_file: Path, size: int = CACHE_SIZE) -> dict:
    """{partition: (uint8 images N x 3 x size x size, labels, image_ids)}; built once, reused."""
    if cache_file.exists():
        return torch.load(cache_file, weights_only=False)
    cache = {}
    for p in ("train", "val", "test"):
        df = manifest[manifest.partition == p]
        imgs = torch.empty((len(df), 3, size, size), dtype=torch.uint8)
        for i, rel in enumerate(df["path"]):
            with Image.open(root / rel) as im:
                im = im.convert("RGB").resize((size, size), Image.BILINEAR)
            imgs[i] = torch.from_numpy(np.array(im)).permute(2, 0, 1)
        cache[p] = (imgs, torch.tensor(df["class_id"].to_numpy()), df["image_id"].tolist())
    torch.save(cache, cache_file)
    return cache


def batch_aug(x: torch.Tensor, size: int, gen: torch.Generator) -> torch.Tensor:
    """Batched GPU augmentation on float images in [0,1] (B,3,H,W) -> (B,3,size,size).

    One random affine per image (via affine_grid/grid_sample): random-resized crop covering
    60-100% of the image area with aspect ratio 0.8-1.25, horizontal and vertical flips (p=0.5),
    rotation in [-20, 20] degrees (corners filled with black); then brightness and contrast factors
    in [0.8, 1.2]. Hue/saturation are left unchanged because colour is diagnostic.
    """
    B, dev = x.shape[0], x.device
    u = lambda lo, hi: torch.rand(B, generator=gen, device=dev) * (hi - lo) + lo  # noqa: E731
    area, logr = u(0.6, 1.0), u(np.log(0.8), np.log(1.25))
    w = torch.sqrt(area * torch.exp(logr)).clamp(max=1.0)  # half-extent in normalised [-1,1] coords
    h = torch.sqrt(area / torch.exp(logr)).clamp(max=1.0)
    tx, ty = u(-1, 1) * (1 - w), u(-1, 1) * (1 - h)
    fx = torch.where(torch.rand(B, generator=gen, device=dev) < 0.5, -1.0, 1.0)
    fy = torch.where(torch.rand(B, generator=gen, device=dev) < 0.5, -1.0, 1.0)
    th = u(-20, 20) * np.pi / 180
    c, s_ = torch.cos(th), torch.sin(th)
    theta = torch.stack([torch.stack([c * w * fx, -s_ * h * fy, tx], 1),
                         torch.stack([s_ * w * fx, c * h * fy, ty], 1)], 1)
    grid = F.affine_grid(theta, (B, 3, size, size), align_corners=False)
    out = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    out = out * u(0.8, 1.2).view(B, 1, 1, 1)
    gray = out.mean(dim=(1, 2, 3), keepdim=True)
    return ((out - gray) * u(0.8, 1.2).view(B, 1, 1, 1) + gray).clamp_(0, 1)


def normalise(x: torch.Tensor) -> torch.Tensor:
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def to_input(x_uint8: torch.Tensor, device, size: int = INPUT_SIZE) -> torch.Tensor:
    x = x_uint8.to(device, non_blocking=True).float().div_(255)
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    return (x - MEAN.to(device)) / STD.to(device)


# Generic ImageNet-1k weights from torchvision (never PlantSeg/PlantWild checkpoints).
ARCHS = {
    "resnet18": ("resnet18", "ResNet18_Weights", "IMAGENET1K_V1"),
    "resnet50": ("resnet50", "ResNet50_Weights", "IMAGENET1K_V2"),
    "mobilenet_v3_large": ("mobilenet_v3_large", "MobileNet_V3_Large_Weights", "IMAGENET1K_V2"),
    "efficientnet_b0": ("efficientnet_b0", "EfficientNet_B0_Weights", "IMAGENET1K_V1"),
    "convnext_tiny": ("convnext_tiny", "ConvNeXt_Tiny_Weights", "IMAGENET1K_V1"),
    "convnext_small": ("convnext_small", "ConvNeXt_Small_Weights", "IMAGENET1K_V1"),
}


def make_model(n_classes: int, arch: str = "resnet18") -> tuple[nn.Module, str]:
    """Returns (model, name prefix of the new classifier layer) - the last nn.Linear is replaced."""
    fn, wcls, wname = ARCHS[arch]
    m = getattr(torchvision.models, fn)(weights=getattr(getattr(torchvision.models, wcls), wname))
    name = [n for n, mod in m.named_modules() if isinstance(mod, nn.Linear)][-1]
    parent, attr = (m.get_submodule(name.rsplit(".", 1)[0]), name.rsplit(".", 1)[1]) if "." in name else (m, name)
    setattr(parent, attr, nn.Linear(getattr(parent, attr).in_features, n_classes))
    return m, name + "."


@torch.no_grad()
def predict(model, imgs, device, bs=128, size: int = INPUT_SIZE):
    model.eval()
    out = []
    for i in range(0, len(imgs), bs):
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(model(to_input(imgs[i:i + bs], device, size)).float().cpu())
    return torch.cat(out)


def train_run(cfg: Config, cache: dict, classes: list[str], device) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    model, head_prefix = make_model(len(classes), cfg.arch)
    model = model.to(device)
    head = [p for n, p in model.named_parameters() if n.startswith(head_prefix)]
    backbone = [p for n, p in model.named_parameters() if not n.startswith(head_prefix)]
    aug_gen = torch.Generator(device=device).manual_seed(cfg.seed)
    if not cfg.finetune:
        for p in backbone:
            p.requires_grad_(False)
    groups = [{"params": head, "lr": cfg.lr_head}]
    if cfg.finetune:
        groups.append({"params": backbone, "lr": cfg.lr_backbone})
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    xtr, ytr, _ = cache["train"]
    steps = cfg.epochs * ((len(ytr) + cfg.batch_size - 1) // cfg.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=steps, pct_start=0.1)
    weight = None
    if cfg.class_weights:  # w_c = N / (K * n_c), from training counts only
        counts = torch.bincount(ytr, minlength=len(classes)).float()
        weight = (len(ytr) / (len(classes) * counts)).to(device)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    hist, best, t0 = [], {"val_macro_f1": -1.0}, time.time()
    for ep in range(1, cfg.epochs + 1):
        model.train()
        if not cfg.finetune:
            model.eval()  # frozen backbone: keep BatchNorm statistics fixed
        run_loss, n = 0.0, 0
        for idx in torch.randperm(len(ytr), generator=gen).split(cfg.batch_size):
            xb = xtr[idx].to(device, non_blocking=True)
            if cfg.augment:
                x = normalise(batch_aug(xb.float().div_(255), cfg.input_size, aug_gen))
            else:
                x = to_input(xb, device, cfg.input_size)
            y = ytr[idx].to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = F.cross_entropy(model(x), y, weight=weight, label_smoothing=cfg.label_smoothing)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * len(idx)
            n += len(idx)
        logits = predict(model, cache["val"][0], device, size=cfg.input_size)
        yv = cache["val"][1]
        vm = evaluate(yv.numpy(), logits.argmax(1).numpy(), classes)
        hist.append({"epoch": ep, "train_loss": run_loss / n, "val_loss": F.cross_entropy(logits, yv).item(),
                     "val_macro_f1": vm["macro_f1"], "val_accuracy": vm["accuracy"]})
        if vm["macro_f1"] > best["val_macro_f1"]:  # checkpoint rule: best validation macro-F1
            best = {"val_macro_f1": vm["macro_f1"], "epoch": ep, "state": copy.deepcopy(model.state_dict())}
    model.load_state_dict(best["state"])
    return {"cfg": asdict(cfg), "hist": pd.DataFrame(hist), "best_epoch": best["epoch"],
            "best_val_macro_f1": best["val_macro_f1"], "model": model, "train_seconds": time.time() - t0}
