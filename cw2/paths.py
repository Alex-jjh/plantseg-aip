"""One place for every path the pipeline reads or writes.

Committed (small, reviewable):  results/  - manifest, audit, per-run records + logits, summaries, figures
Not committed (large/derived):  data/     - PlantSeg v7 zip + extracted images (downloaded from Zenodo)
                                outputs/  - image caches and model checkpoints (checkpoints are on Hugging Face)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    repo: Path

    @property
    def raw_zip(self) -> Path:
        return self.repo / "data" / "raw" / "plantseg.zip"

    @property
    def data_root(self) -> Path:  # contains Metadata.csv, images/, annotations/
        return self.repo / "data" / "plantseg" / "plantseg"

    @property
    def results(self) -> Path:
        return self.repo / "results"

    @property
    def data_results(self) -> Path:  # manifest.csv, exclusions.csv, audit.json
        return self.results / "data"

    @property
    def runs(self) -> Path:  # runs/<config>_s<seed>/{run.json, hist.csv, logits.npz}
        return self.results / "runs"

    @property
    def summary(self) -> Path:
        return self.results / "summary"

    @property
    def figures(self) -> Path:
        return self.results / "figures"

    @property
    def outputs(self) -> Path:
        return self.repo / "outputs"

    @property
    def checkpoints(self) -> Path:  # outputs/checkpoints/<config>_s<seed>.pt
        return self.outputs / "checkpoints"

    def cache(self, size: int) -> Path:
        return self.outputs / f"cache_{size}.pt"

    def sweep(self, stage: str) -> Path:
        return self.repo / "sweeps" / f"{stage}.json"
