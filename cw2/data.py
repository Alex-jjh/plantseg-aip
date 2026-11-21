"""Dataset access, manifest and audit for the AIP304 CW2 PlantSeg v7 nine-host subset."""
from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

# PlantSeg v7 (Wei et al., 2025), Zenodo record 17719108 - one fixed release, never mixed.
ZENODO_URL = "https://zenodo.org/api/records/17719108/files/plantseg.zip/content"
ZENODO_MD5 = "9358a66dff88cdd15c4fe009763c40a3"
ZENODO_BYTES = 1057281724

HOSTS = ["apple", "banana", "citrus", "corn", "potato", "soybean", "tomato", "wheat", "zucchini"]
SPLIT_DIR = {"Training": "train", "Validation": "val", "Test": "test"}
PARTITIONS = ["train", "val", "test"]


def build_manifest(root: Path) -> pd.DataFrame:
    """One row per subset image: image_id, path, host, class name, class_id, partition.

    Class = the supplied host-disease `Disease` label; ids follow alphabetical class order.
    """
    meta = pd.read_csv(root / "Metadata.csv", encoding="utf-8-sig")
    sub = meta[meta["Plant"].str.lower().isin(HOSTS)].copy()
    sub["partition"] = sub["Split"].map(SPLIT_DIR)
    assert sub["partition"].notna().all(), "unknown Split value"
    sub["path"] = [str(Path("images") / p / n) for p, n in zip(sub["partition"], sub["Name"])]
    classes = sorted(sub["Disease"].unique())
    sub["class_id"] = sub["Disease"].map({c: i for i, c in enumerate(classes)})
    out = sub.rename(columns={"Name": "image_id", "Plant": "host", "Disease": "class_name"})
    cols = ["image_id", "path", "host", "class_name", "class_id", "partition", "URL", "License"]
    return out[cols].sort_values(["partition", "class_id", "image_id"]).reset_index(drop=True)


def audit(manifest: pd.DataFrame, root: Path, phash_max_dist: int = 4) -> dict:
    """Unreadable files, exact (byte-identical) duplicates, and perceptual near-duplicates."""
    unreadable, md5, phash = [], {}, {}
    for row in manifest.itertuples():
        p = root / row.path
        try:
            data = p.read_bytes()
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.load()
                phash[row.image_id] = imagehash.phash(im)
            md5[row.image_id] = hashlib.md5(data).hexdigest()
        except Exception as e:  # noqa: BLE001 - any failure means unreadable
            unreadable.append((row.image_id, repr(e)))

    groups = defaultdict(list)
    for iid, h in md5.items():
        groups[h].append(iid)
    exact = [g for g in groups.values() if len(g) > 1]

    # Near-duplicates: pairwise pHash Hamming distance (n~4.3k -> ~9.4M pairs, fine with numpy).
    ids = list(phash)
    bits = np.stack([phash[i].hash.flatten() for i in ids]).astype(np.uint8)
    packed = np.packbits(bits, axis=1)  # (n, 8) bytes
    near = []
    for i in range(len(ids)):
        x = np.bitwise_xor(packed[i + 1 :], packed[i])
        d = np.unpackbits(x, axis=1).sum(1)
        for j in np.nonzero(d <= phash_max_dist)[0]:
            near.append((ids[i], ids[i + 1 + j], int(d[j])))
    return {"unreadable": unreadable, "exact_groups": exact, "near_pairs": near}


def duplicate_groups(exact_groups: list[list[str]], near_pairs) -> list[list[str]]:
    """Connected components over byte-identical groups + pHash near-duplicate pairs.
    (Visual check: pHash distance <=4 pairs are the same photo re-cropped/re-coloured/watermarked.)"""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges = [(g[0], o) for g in exact_groups for o in g[1:]] + [(a, b) for a, b, _ in near_pairs]
    for a, b in edges:
        parent[find(a)] = find(b)
    comps = defaultdict(list)
    for x in list(parent):
        comps[find(x)].append(x)
    return [sorted(c) for c in comps.values() if len(c) > 1]


def resolve_exclusions(manifest: pd.DataFrame, dup_groups: list[list[str]], unreadable) -> pd.DataFrame:
    """Rule (fixed before modelling): drop unreadable files; for each duplicate group keep one
    copy - the one in the latest partition (test > val > train, then name) so a photo never spans
    partitions; drop the whole group if its members disagree on class (contradictory labels)."""
    rank = {"train": 0, "val": 1, "test": 2}
    info = manifest.set_index("image_id")
    rows = [(iid, "unreadable") for iid, _ in unreadable]
    for g in dup_groups:
        if info.loc[g, "class_id"].nunique() > 1:
            rows += [(iid, "duplicate with conflicting labels") for iid in g]
            continue
        keep = max(g, key=lambda i: (rank[info.loc[i, "partition"]], i))
        rows += [(iid, f"duplicate of {keep}") for iid in g if iid != keep]
    return pd.DataFrame(rows, columns=["image_id", "reason"]).drop_duplicates("image_id")


def md5sum(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def download_plantseg(zip_path: Path, extract_to: Path) -> Path:
    """Download plantseg.zip (v7) if missing, verify the published MD5, extract. Returns data root."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists() or zip_path.stat().st_size != ZENODO_BYTES:
        print(f"downloading {ZENODO_URL} (~1.06 GB) ...")
        urllib.request.urlretrieve(ZENODO_URL, zip_path)
    got = md5sum(zip_path)
    assert got == ZENODO_MD5, f"checksum mismatch: {got} != {ZENODO_MD5}"
    root = extract_to / "plantseg"
    if not (root / "Metadata.csv").exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_to)
    print(f"plantseg.zip md5 OK ({got}); data root {root}")
    return root


def prepare_manifest(root: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build the fixed manifest, run the audit, apply the exclusion rule; write
    out_dir/{manifest.csv, exclusions.csv, audit.json}. Returns (manifest, exclusions, audit)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    man = build_manifest(root)
    a = audit(man, root)
    groups = duplicate_groups(a["exact_groups"], a["near_pairs"])
    excl = resolve_exclusions(man, groups, a["unreadable"])
    part, cls = man.set_index("image_id")["partition"], man.set_index("image_id")["class_name"]
    man["excluded"] = man["image_id"].isin(excl["image_id"])
    man.to_csv(out_dir / "manifest.csv", index=False)
    excl.to_csv(out_dir / "exclusions.csv", index=False)
    rec = {
        "n_images": len(man), "n_classes": int(man["class_id"].nunique()), "unreadable": a["unreadable"],
        "exact_groups": a["exact_groups"],
        "near_pairs": [{"a": x, "b": y, "dist": d, "pa": part[x], "pb": part[y], "ca": cls[x], "cb": cls[y]}
                       for x, y, d in a["near_pairs"]],
        "duplicate_groups": groups,
    }
    json.dump(rec, open(out_dir / "audit.json", "w"), indent=1)
    return man, excl, rec


def load_kept(data_results: Path) -> tuple[pd.DataFrame, list[str]]:
    """Manifest rows used for modelling (exclusions removed) and the fixed class order."""
    man = pd.read_csv(data_results / "manifest.csv")
    man = man[~man["excluded"]].reset_index(drop=True)
    classes = man.drop_duplicates("class_id").sort_values("class_id")["class_name"].tolist()
    return man, classes
