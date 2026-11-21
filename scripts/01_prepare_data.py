"""Download PlantSeg v7 (MD5-checked), build the manifest, audit, apply exclusions -> results/data/."""
from _common import P

from cw2.data import download_plantseg, prepare_manifest

root = download_plantseg(P.raw_zip, P.data_root.parent)
man, excl, audit = prepare_manifest(root, P.data_results)
kept = man[~man.excluded]
print(f"{len(man)} images, {man.class_id.nunique()} classes; excluded {len(excl)}; kept per partition:",
      kept.partition.value_counts().reindex(["train", "val", "test"]).to_dict())
