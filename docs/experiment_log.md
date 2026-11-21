# Experiment log

What was run, in order, and every decision that affected the results. Dates are 2026.

## 1. Data (20 Nov)
- Downloaded PlantSeg v7 `plantseg.zip` from Zenodo 17719108; MD5 `9358a66dff88cdd15c4fe009763c40a3` matches the record.
- Nine-host subset: 4,349 images, 43 host–disease classes (matches the task sheet), supplied split ≈ 69/11/20.
- Audit: 0 unreadable files; 161 byte-identical groups. A pHash (≤ 4) pass found 419 more pairs; a visual check of sampled
  pairs at distances 2 and 4 showed the same photo re-cropped / re-coloured / watermarked, so the threshold was adopted
  and pairs merged into 327 duplicate groups (162 spanning partitions, 54 with conflicting labels).
- Exclusion rule fixed before modelling (keep the latest-partition copy; drop conflicting-label groups): 424 images
  excluded, 3,925 kept (train 2,659 / val 439 / test 827). Every class keeps ≥ 1 validation and ≥ 5 test images.

## 2. CPU baselines (20 Nov, development machine without GPU)
- Majority class and a frozen ResNet-18 + linear head on cached features (CPU). These were exploratory; the reported
  E0 is the same model re-run in the final GPU pipeline (Section 4) so that every system shares one code path.

## 3. Move to GPU (21 Nov, 1× NVIDIA L4)
- Images are decoded once into uint8 tensors (whole photo resized to a square, no crop) because `/dev/shm` is 64 MB and
  DataLoader worker processes crash; training reads from memory without workers.
- A first E0–E3 run used per-image CPU augmentation; it was too slow (≈18 s/epoch) and was stopped after E0 and E1.
  Augmentation was re-implemented as one batched affine transform on the GPU (same crop/flip/rotation/brightness/
  contrast ranges, checked visually) and **all** reported runs, E0–E3 included, use that implementation. The
  preliminary run is not reported.

## 4. Sweep (21 Nov) - every configuration with seeds 0, 1, 2; up to 4 runs sharing the GPU
- **stage 0** - planned comparison E0 frozen / E1 fine-tune / E2 + class weights / E3 + augmentation.
- **stage 1** - one change at a time on E3: class weights, backbone LR 3e-5 / 3e-4, 50 epochs, label smoothing 0.1,
  280 px, ResNet-50, EfficientNet-B0, ConvNeXt-Tiny, MobileNetV3-Large.
- **stage 2** - combinations of the helpful stage-1 changes (ConvNeXt-T 280 px / LS 0.1; ResNet-50 280 px;
  ResNet-18 280 px + LS; ResNet-18 336 px).
- **stage 3** - refinement around ConvNeXt: ConvNeXt-Small; ConvNeXt-T backbone LR 5e-5 / 2e-4.
- **Pruned on validation** (`sweeps/pruned.json`): ConvNeXt-T 280 px + LS (ConvNeXt-T 280 px was already below 224 px;
  2 seeds were stopped early and discarded) and ResNet-50 280 px + LS (ResNet-50 clearly below ConvNeXt-T).
- 66 completed runs; 12.5 GPU-hours recorded under 4-way sharing.

## 5. Selection (21 Nov, validation only)
- Rule written before results were seen: highest validation macro-F1 among all configs (3-seed mean) and the 3-seed
  ensembles of the top 5 configs. Frozen in `results/summary/selection.json` (14:16 UTC) before any test metric.
- Winner: ConvNeXt-T, augmentation, backbone LR 2e-4, 3-seed ensemble (val 0.835); runner-up ConvNeXt-T default-LR
  ensemble (0.833).
- Sanity check before testing: one ensemble scored below all of its members on validation. Labels/order and
  per-seed scores were re-verified (identical); the cause is the small validation set (3 classes have ≤ 2 images, so
  one image moves macro-F1 by ~0.01). The frozen rule was kept rather than overridden.

## 6. Test (21 Nov, once)
- Final ensemble: macro-F1 0.805, accuracy 0.842, balanced accuracy 0.800.
- Post-freeze, for transparency only: the top-8 validation candidates all score 0.79–0.81 on test.

## 7. Packaging (21 Nov)
- Code refactored into `cw2/` with thin scripts and one notebook; run records moved to `results/runs/`
  (logits as `.npz`), checkpoints to `outputs/checkpoints/`. The refactored pipeline reproduces the selection and every
  test number exactly, and reloaded checkpoints reproduce the stored test predictions (100% argmax agreement).
