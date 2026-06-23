# Leave-One-Well Local Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build experiment 21 with one common no-wide-target pretraining stage, four identically configured local-wideband calibration folds, locked held-out-well inference, and aggregate inline/crossline evaluation.

**Architecture:** Reuse experiment 20's F3 self-supervision, synthetic residual datasets, phase-consistent model, and leakage guards through experiment-local modules copied at scaffold time. Add a fold planner that owns all spatial boundaries, a local calibration extractor that emits auditable manifests, one common pretrainer, one fold fine-tuner, and locked inference/evaluation scripts that cannot read held-out references before checkpoint verification.

**Tech Stack:** Python 3, NumPy, SciPy, PyTorch, Matplotlib, pytest, existing `shared_code/segy_reader.py`, existing F3 SEG-Y and shared well arrays.

---

## File Map

Create the following experiment-local files:

```text
21_leave_one_well_calibration/
  README.md
  code/
    config.py                       # immutable coordinates, windows and training settings
    fold_geometry.py                # fold definitions and spatial leakage checks
    local_calibration_samples.py    # local window extraction and pair construction
    datasets.py                     # three-domain datasets and deterministic sampling
    phase_model.py                  # experiment-local model and frequency projector
    phase_loss.py                   # local-wide, synthetic and self-supervised losses
    phase_metrics.py                # spatial, spectral and phase metrics
    leakage_guard.py                # manifests, SHA locks and read authorization
    01_prepare_common_data.py       # F3 self-supervision and synthetic-data manifest
    02_pretrain_common.py           # common no-F3-wide-target pretraining
    03_generate_fold_samples.py     # four-fold local calibration sample generation
    04_finetune_fold.py             # identical per-fold fine-tuning
    05_predict_heldout.py           # locked inline/crossline held-out inference
    06_evaluate_folds.py            # per-fold and aggregate evaluation
  tests/
    test_config.py
    test_fold_geometry.py
    test_local_calibration_samples.py
    test_datasets.py
    test_losses.py
    test_common_pretraining.py
    test_fold_training.py
    test_leakage_guard.py
    test_inference.py
    test_aggregate_evaluation.py
```

Reuse root-level data rather than copying it:

```text
Rawdata/Seismic_data.sgy
shared_data/
shared_code/
16_geometry_realistic_samples/data/
20_curriculum_multiband/data/F3多频带自监督/
```

## Task 1: Scaffold Experiment 21 and Lock Configuration

**Files:**
- Create: `21_leave_one_well_calibration/README.md`
- Create: `21_leave_one_well_calibration/code/config.py`
- Create: `21_leave_one_well_calibration/tests/test_config.py`
- Create: `docs/23_21号实验改进思路.txt`

- [ ] **Step 1: Write the failing configuration test**

```python
from code.config import FOLDS, WELLS


def test_four_folds_leave_each_well_out_once():
    assert set(FOLDS) == {"fold_well1", "fold_well2", "fold_well3", "fold_well4"}
    assert {fold["heldout_well"] for fold in FOLDS.values()} == set(WELLS)
    for fold in FOLDS.values():
        assert len(fold["calibration_wells"]) == 3
        assert fold["heldout_well"] not in fold["calibration_wells"]


def test_fixed_window_and_training_configuration():
    from code.config import (
        CALIBRATION_INLINE_RADIUS,
        CALIBRATION_CROSSLINE_RADIUS,
        HELDOUT_INLINE_GUARD,
        HELDOUT_CROSSLINE_GUARD,
        LOCAL_TIME_PATCH,
        LOCAL_SPATIAL_PATCH,
    )
    assert (CALIBRATION_INLINE_RADIUS, CALIBRATION_CROSSLINE_RADIUS) == (8, 16)
    assert (HELDOUT_INLINE_GUARD, HELDOUT_CROSSLINE_GUARD) == (16, 32)
    assert (LOCAL_TIME_PATCH, LOCAL_SPATIAL_PATCH) == (256, 32)
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
D:\Anaconda\python.exe -m pytest 21_leave_one_well_calibration\tests\test_config.py -q
```

Expected: collection fails because `code/config.py` does not exist.

- [ ] **Step 3: Implement immutable configuration**

Define:

```python
WELLS = {
    "well1": {"inline": 244, "crossline": 336},
    "well2": {"inline": 362, "crossline": 387},
    "well3": {"inline": 442, "crossline": 848},
    "well4": {"inline": 722, "crossline": 1007},
}
FOLDS = {
    f"fold_{heldout}": {
        "heldout_well": heldout,
        "calibration_wells": tuple(name for name in WELLS if name != heldout),
    }
    for heldout in WELLS
}
CALIBRATION_INLINE_RADIUS = 8
CALIBRATION_CROSSLINE_RADIUS = 16
HELDOUT_INLINE_GUARD = 16
HELDOUT_CROSSLINE_GUARD = 32
LOCAL_TIME_PATCH = 256
LOCAL_SPATIAL_PATCH = 32
LOCAL_TIME_STRIDE = 128
LOCAL_SPATIAL_STRIDE = 8
```

Add paths, narrow-band filter `(3, 6, 25, 35)`, fold training ratios `0.6/0.2/0.2`, stage epochs `60/40`, random seed `42`, and directory creation.

- [ ] **Step 4: Add README and incremental TXT**

Document that experiment 21 uses local wideband labels only inside three calibration-well windows per fold, never initializes from experiment 18, and evaluates each held-out well once.

- [ ] **Step 5: Run test and commit**

```powershell
D:\Anaconda\python.exe -m pytest 21_leave_one_well_calibration\tests\test_config.py -q
git add F3_auxiliary_experiment/21_leave_one_well_calibration F3_auxiliary_experiment/docs/23_21号实验改进思路.txt
git commit -m "feat: scaffold leave-one-well calibration experiment"
```

## Task 2: Implement Fold Geometry and Spatial Leakage Audit

**Files:**
- Create: `21_leave_one_well_calibration/code/fold_geometry.py`
- Create: `21_leave_one_well_calibration/tests/test_fold_geometry.py`

- [ ] **Step 1: Write failing fold-boundary tests**

Test the wished-for API:

```python
from fold_geometry import (
    calibration_window,
    heldout_guard,
    plan_fold,
    validate_fold_manifest,
)


def test_calibration_window_has_expected_bounds():
    window = calibration_window({"inline": 362, "crossline": 387})
    assert window == {
        "inline_min": 354,
        "inline_max": 370,
        "crossline_min": 371,
        "crossline_max": 403,
    }


def test_fold_manifest_excludes_heldout_guard():
    manifest = plan_fold("fold_well1")
    validate_fold_manifest(manifest)
    guard = heldout_guard({"inline": 244, "crossline": 336})
    for sample in manifest["wide_sample_regions"]:
        assert sample["inline_max"] < guard["inline_min"] or sample["inline_min"] > guard["inline_max"] \
            or sample["crossline_max"] < guard["crossline_min"] or sample["crossline_min"] > guard["crossline_max"]
```

Also assert:

- every well is held out exactly once;
- every fold contains exactly three calibration windows;
- train inline offsets are `-8..+6`;
- validation offsets are `+7,+8`;
- no train/validation spatial coordinate overlaps.

- [ ] **Step 2: Run tests and verify RED**

```powershell
D:\Anaconda\python.exe -m pytest 21_leave_one_well_calibration\tests\test_fold_geometry.py -q
```

Expected: import failure for `fold_geometry`.

- [ ] **Step 3: Implement geometry functions**

Use integer inclusive bounds. `plan_fold()` must return only coordinate metadata, not seismic arrays. `validate_fold_manifest()` must raise `ValueError` for:

- held-out well present in calibration wells;
- calibration region outside `±8/±16`;
- held-out guard intersection;
- train/validation plane overlap;
- target held-out inline or crossline listed as a wide-label source.

- [ ] **Step 4: Run tests and commit**

```powershell
D:\Anaconda\python.exe -m pytest 21_leave_one_well_calibration\tests\test_fold_geometry.py -q
git add F3_auxiliary_experiment/21_leave_one_well_calibration/code/fold_geometry.py F3_auxiliary_experiment/21_leave_one_well_calibration/tests/test_fold_geometry.py
git commit -m "feat: add auditable leave-one-well geometry"
```

## Task 3: Build Local Calibration Pair Extraction

**Files:**
- Create: `21_leave_one_well_calibration/code/local_calibration_samples.py`
- Create: `21_leave_one_well_calibration/tests/test_local_calibration_samples.py`

- [ ] **Step 1: Write failing pair and patch tests**

```python
def test_pair_uses_narrow_p99_and_clean_label():
    narrow, residual, scale = make_local_pair(wide, dt=0.004)
    assert np.isclose(scale, np.percentile(np.abs(lowpass(wide)), 99))
    assert np.max(np.abs(narrow + residual - wide / scale)) < 1e-6


def test_patch_coordinates_stay_inside_calibration_window():
    rows = plan_local_patches(window, time_size=462)
    assert all(row["inline_min"] >= window["inline_min"] for row in rows)
    assert all(row["inline_max"] <= window["inline_max"] for row in rows)
    assert all(row["crossline_min"] >= window["crossline_min"] for row in rows)
    assert all(row["crossline_max"] <= window["crossline_max"] for row in rows)


def test_training_crossline_patch_never_uses_validation_inline():
    rows = plan_local_patches(window, time_size=462, split="train")
    assert all(not set(row["inline_values"]) & {well_inline + 7, well_inline + 8} for row in rows)
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
D:\Anaconda\python.exe -m pytest 21_leave_one_well_calibration\tests\test_local_calibration_samples.py -q
```

- [ ] **Step 3: Implement pair construction and in-window padding**

Implement:

```python
def make_local_pair(wide_patch, dt):
    narrow_raw = zero_phase_filter_section(wide_patch, dt, NARROW_BAND)
    scale = max(float(np.percentile(np.abs(narrow_raw), 99)), 1e-8)
    narrow = (narrow_raw / scale).astype(np.float32)
    residual = (wide_patch / scale - narrow).astype(np.float32)
    return narrow, residual, scale
```

Implement reflective padding from data already inside the local window only. Metadata must store:

```text
fold
well
split
source_axis
time_start
inline_values
crossline_values
valid_spatial_width
left_pad
right_pad
scale
closure_max_abs
```

- [ ] **Step 4: Add a tiny synthetic-cube integration test**

Use a deterministic cube whose sample values encode inline and crossline indices. Verify extracted arrays and metadata point to the same coordinates and that no outside-window values enter padding.

- [ ] **Step 5: Run tests and commit**

```powershell
D:\Anaconda\python.exe -m pytest 21_leave_one_well_calibration\tests\test_local_calibration_samples.py -q
git add F3_auxiliary_experiment/21_leave_one_well_calibration/code/local_calibration_samples.py F3_auxiliary_experiment/21_leave_one_well_calibration/tests/test_local_calibration_samples.py
git commit -m "feat: extract leakage-safe local calibration pairs"
```

## Task 4: Add Manifests, Hash Locks, and Reference-Read Authorization

**Files:**
- Create: `21_leave_one_well_calibration/code/leakage_guard.py`
- Create: `21_leave_one_well_calibration/tests/test_leakage_guard.py`

- [ ] **Step 1: Write failing security-boundary tests**

Test that:

- common pretraining manifests reject any `uses_f3_wide_target=True`;
- fold manifests reject the held-out well in local-wide sources;
- checkpoint SHA mismatch is rejected;
- manifest SHA mismatch is rejected;
- held-out reference read is rejected before lock verification;
- fold lock records the same common-pretrain SHA for all folds.

Example:

```python
with pytest.raises(ValueError, match="held-out"):
    authorize_heldout_reference(
        checkpoint_path,
        lock_path,
        requested_well="well1",
        requested_axis="inline",
        requested_number=244,
    )
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
D:\Anaconda\python.exe -m pytest 21_leave_one_well_calibration\tests\test_leakage_guard.py -q
```

- [ ] **Step 3: Implement canonical JSON hashing**

Hash manifests from UTF-8 JSON serialized with sorted keys and compact separators. Implement:

```python
sha256_file(path)
sha256_payload(payload)
create_common_lock(...)
verify_common_lock(...)
create_fold_lock(...)
verify_fold_lock(...)
authorize_heldout_reference(...)
```

`authorize_heldout_reference()` must verify the checkpoint, fold manifest, common checkpoint SHA, held-out well, and requested section number before returning metadata.

- [ ] **Step 4: Run tests and commit**

```powershell
D:\Anaconda\python.exe -m pytest 21_leave_one_well_calibration\tests\test_leakage_guard.py -q
git add F3_auxiliary_experiment/21_leave_one_well_calibration/code/leakage_guard.py F3_auxiliary_experiment/21_leave_one_well_calibration/tests/test_leakage_guard.py
git commit -m "feat: lock fold manifests and heldout reads"
```

## Task 5: Prepare Common Pretraining Data Without F3 Wide Targets

**Files:**
- Create: `21_leave_one_well_calibration/code/01_prepare_common_data.py`
- Create: `21_leave_one_well_calibration/tests/test_common_pretraining.py`

- [ ] **Step 1: Write failing manifest test**

Assert the preparation output contains:

```python
assert manifest["uses_f3_wide_target"] is False
assert manifest["f3_source"] == "narrow_multiband_self_supervision"
assert manifest["synthetic_source"] == "well_constrained_residual"
assert "18_real_domain_phase_consistent" not in json.dumps(manifest)
```

Also assert all referenced files exist under experiment 20's F3 self-supervision data or experiment 16/20 synthetic data, never experiment 18.

- [ ] **Step 2: Run test and verify RED**

```powershell
D:\Anaconda\python.exe -m pytest 21_leave_one_well_calibration\tests\test_common_pretraining.py -q
```

- [ ] **Step 3: Implement preparation script**

The script should validate and record existing reusable arrays rather than duplicate them. Save:

```text
data/公共预训练/common_data_manifest.json
data/公共预训练/common_data_lock.json
```

Include file paths, shapes, dtypes, SHA256 for small metadata files, sample counts, normalization, and label definitions.

- [ ] **Step 4: Run preparation and tests**

```powershell
D:\Anaconda\python.exe 21_leave_one_well_calibration\code\01_prepare_common_data.py
D:\Anaconda\python.exe -m pytest 21_leave_one_well_calibration\tests\test_common_pretraining.py -q
```

- [ ] **Step 5: Commit**

```powershell
git add F3_auxiliary_experiment/21_leave_one_well_calibration/code/01_prepare_common_data.py F3_auxiliary_experiment/21_leave_one_well_calibration/tests/test_common_pretraining.py
git commit -m "feat: audit common no-wide pretraining data"
```

## Task 6: Implement Three-Domain Datasets and Deterministic Sampling

**Files:**
- Create: `21_leave_one_well_calibration/code/datasets.py`
- Create: `21_leave_one_well_calibration/tests/test_datasets.py`

- [ ] **Step 1: Write failing dataset tests**

Cover:

- local samples return `input`, `label`, `target`, `domain="local_wide"`;
- synthetic samples return `domain="synthetic"`;
- F3 self-supervised samples return `domain="f3_self_supervised"` and a projector;
- only horizontal flips occur;
- sampler ratio over 1000 draws is within 3% of `0.6/0.2/0.2`;
- validation datasets do not augment;
- fold datasets only load their manifest-listed wells and planes.

- [ ] **Step 2: Run tests and verify RED**

```powershell
D:\Anaconda\python.exe -m pytest 21_leave_one_well_calibration\tests\test_datasets.py -q
```

- [ ] **Step 3: Implement datasets and domain cycle**

Use memory-mapped NumPy arrays. Provide:

```python
LocalCalibrationDataset
SyntheticResidualDataset
F3MaskedDataset
domain_cycle(seed, local_ratio=0.6, synthetic_ratio=0.2, f3_ratio=0.2)
```

Never call `np.flip(..., axis=0)`.

- [ ] **Step 4: Run tests and commit**

```powershell
D:\Anaconda\python.exe -m pytest 21_leave_one_well_calibration\tests\test_datasets.py -q
git add F3_auxiliary_experiment/21_leave_one_well_calibration/code/datasets.py F3_auxiliary_experiment/21_leave_one_well_calibration/tests/test_datasets.py
git commit -m "feat: add deterministic three-domain fold datasets"
```

## Task 7: Implement Model and Domain-Aware Losses

**Files:**
- Create: `21_leave_one_well_calibration/code/phase_model.py`
- Create: `21_leave_one_well_calibration/code/phase_loss.py`
- Create: `21_leave_one_well_calibration/code/phase_metrics.py`
- Create: `21_leave_one_well_calibration/tests/test_losses.py`

- [ ] **Step 1: Write failing loss tests**

Cover:

- model accepts `256x32` and `256x256`;
- projector emits negligible 0-22 Hz energy;
- perfect local-wide residual has near-zero total loss;
- phase-inverted residual has worse correlation and STFT loss;
- lateral discontinuity increases gradient loss;
- F3 self-supervised loss only sees its target band;
- reconstructed low-frequency body equals the input.

- [ ] **Step 2: Run tests and verify RED**

```powershell
D:\Anaconda\python.exe -m pytest 21_leave_one_well_calibration\tests\test_losses.py -q
```

- [ ] **Step 3: Implement the model**

Port the tested full-convolution UNet-CBAM and frequency projector from experiment 20. Keep projector boundaries:

```text
low_stop=22
low_pass=28
high_pass=85
high_stop=100
```

- [ ] **Step 4: Implement domain-aware losses**

Use explicit weight dictionaries. For local-wide samples emphasize:

```python
LOCAL_WEIGHTS = {
    "residual": 1.0,
    "correlation": 1.5,
    "complex_stft": 1.5,
    "log_amplitude": 0.25,
    "lateral": 0.35,
    "wide_waveform": 0.5,
    "leakage": 1.0,
}
```

Keep separate synthetic and F3 self-supervised weight sets.

- [ ] **Step 5: Run tests and commit**

```powershell
D:\Anaconda\python.exe -m pytest 21_leave_one_well_calibration\tests\test_losses.py -q
git add F3_auxiliary_experiment/21_leave_one_well_calibration/code/phase_model.py F3_auxiliary_experiment/21_leave_one_well_calibration/code/phase_loss.py F3_auxiliary_experiment/21_leave_one_well_calibration/code/phase_metrics.py F3_auxiliary_experiment/21_leave_one_well_calibration/tests/test_losses.py
git commit -m "feat: add phase-focused local calibration losses"
```

## Task 8: Implement and Run Common Pretraining

**Files:**
- Create: `21_leave_one_well_calibration/code/02_pretrain_common.py`
- Extend: `21_leave_one_well_calibration/tests/test_common_pretraining.py`

- [ ] **Step 1: Add failing smoke and lock tests**

Run one CPU epoch with tiny datasets and assert:

- both F3 and synthetic domains are visited;
- no local-wide domain is loaded;
- checkpoint metadata says `uses_f3_wide_target=False`;
- common lock SHA matches the checkpoint;
- no experiment 18 path appears in metadata.

- [ ] **Step 2: Run test and verify RED**

```powershell
D:\Anaconda\python.exe -m pytest 21_leave_one_well_calibration\tests\test_common_pretraining.py -q
```

- [ ] **Step 3: Implement common trainer**

Use experiment 20's 300-epoch curriculum:

```text
1-60: F3 self-supervision only
61-180: F3:synthetic = 2:1
181-300: F3:synthetic = 1:1
```

Save a common diagnostic candidate every time F3 correlation/phase improves. Create the formal common lock only if the existing F3 gate passes; otherwise record a separate diagnostic checkpoint and require an explicit experiment decision before folds begin.

- [ ] **Step 4: Run unit tests**

```powershell
D:\Anaconda\python.exe -m pytest 21_leave_one_well_calibration\tests\test_common_pretraining.py -q
```

- [ ] **Step 5: Run full common pretraining**

```powershell
D:\Anaconda\python.exe 21_leave_one_well_calibration\code\02_pretrain_common.py --epochs 300 *> 21_leave_one_well_calibration\logs\pretrain_common.log
```

Verify the process exits successfully and a common checkpoint plus metadata exists. Record the chosen epoch and SHA.

- [ ] **Step 6: Commit code**

```powershell
git add F3_auxiliary_experiment/21_leave_one_well_calibration/code/02_pretrain_common.py F3_auxiliary_experiment/21_leave_one_well_calibration/tests/test_common_pretraining.py
git commit -m "feat: train common no-wide calibration initializer"
```

## Task 9: Generate and Audit Four Fold Sample Sets

**Files:**
- Create: `21_leave_one_well_calibration/code/03_generate_fold_samples.py`
- Extend: `21_leave_one_well_calibration/tests/test_local_calibration_samples.py`

- [ ] **Step 1: Add failing end-to-end manifest tests**

For every fold assert:

```python
assert manifest["heldout_well"] not in manifest["calibration_wells"]
assert manifest["uses_heldout_well_wide_target"] is False
assert set(manifest["validation_inline_offsets"]) == {7, 8}
assert manifest["max_closure_abs"] < 1e-6
assert manifest["train_validation_coordinate_overlap"] == 0
assert manifest["heldout_guard_overlap"] == 0
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
D:\Anaconda\python.exe -m pytest 21_leave_one_well_calibration\tests\test_local_calibration_samples.py -q
```

- [ ] **Step 3: Implement fold generator**

Read SEG-Y once, use header-derived geometry, generate each fold under:

```text
data/局部宽频标定/fold_wellN/
  train_inputs.npy
  train_labels.npy
  val_inputs.npy
  val_labels.npy
  train_metadata.npy
  val_metadata.npy
  fold_manifest.json
```

Generate calibration-window maps and sample examples with well markers.

- [ ] **Step 4: Run generation**

```powershell
D:\Anaconda\python.exe 21_leave_one_well_calibration\code\03_generate_fold_samples.py *> 21_leave_one_well_calibration\logs\generate_fold_samples.log
```

- [ ] **Step 5: Audit generated manifests**

Run:

```powershell
D:\Anaconda\python.exe -m pytest 21_leave_one_well_calibration\tests\test_fold_geometry.py 21_leave_one_well_calibration\tests\test_local_calibration_samples.py -q
```

Expected: all folds pass, zero held-out guard overlap, zero train/validation overlap.

- [ ] **Step 6: Commit code**

```powershell
git add F3_auxiliary_experiment/21_leave_one_well_calibration/code/03_generate_fold_samples.py F3_auxiliary_experiment/21_leave_one_well_calibration/tests/test_local_calibration_samples.py
git commit -m "feat: generate four local calibration folds"
```

## Task 10: Implement Identical Per-Fold Fine-Tuning and Gates

**Files:**
- Create: `21_leave_one_well_calibration/code/04_finetune_fold.py`
- Create: `21_leave_one_well_calibration/tests/test_fold_training.py`

- [ ] **Step 1: Write failing training tests**

Cover:

- all folds load the same common checkpoint SHA;
- all fold hyperparameter dictionaries are identical;
- first 10 epochs freeze the first two encoder blocks;
- later epochs unfreeze all parameters;
- domain counts follow `60/20/20`;
- checkpoint selector rejects non-positive local residual correlation or phase;
- selector rejects F3 self-supervision below 95% of common baseline;
- selector rejects leakage above 0.01;
- selector tie-breaks correlation within 0.01 using phase.

- [ ] **Step 2: Run tests and verify RED**

```powershell
D:\Anaconda\python.exe -m pytest 21_leave_one_well_calibration\tests\test_fold_training.py -q
```

- [ ] **Step 3: Implement fold trainer**

CLI:

```text
04_finetune_fold.py --fold fold_well1
```

Implement stage A for 60 epochs and stage B for up to 40 epochs. Save:

```text
best_model.pth
last_model.pth
training_history.json
fold_lock.json
```

Do not read any held-out reference during training.

- [ ] **Step 4: Run smoke tests**

```powershell
D:\Anaconda\python.exe -m pytest 21_leave_one_well_calibration\tests\test_fold_training.py -q
```

- [ ] **Step 5: Run all four folds with identical commands**

```powershell
foreach ($fold in 'fold_well1','fold_well2','fold_well3','fold_well4') {
  D:\Anaconda\python.exe 21_leave_one_well_calibration\code\04_finetune_fold.py --fold $fold *> "21_leave_one_well_calibration\logs\train_$fold.log"
}
```

After every fold, verify `fold_lock.json`, checkpoint SHA, common-pretrain SHA, gate metrics, and absence of held-out paths in training metadata.

- [ ] **Step 6: Commit code**

```powershell
git add F3_auxiliary_experiment/21_leave_one_well_calibration/code/04_finetune_fold.py F3_auxiliary_experiment/21_leave_one_well_calibration/tests/test_fold_training.py
git commit -m "feat: fine-tune four locked calibration folds"
```

## Task 11: Implement Locked Held-Out Inference

**Files:**
- Create: `21_leave_one_well_calibration/code/05_predict_heldout.py`
- Create: `21_leave_one_well_calibration/tests/test_inference.py`

- [ ] **Step 1: Write failing authorization and recombination tests**

Cover:

- `fold_well1` only permits inline 244 and crossline 336;
- requesting another well is rejected;
- reference read is rejected before fold-lock verification;
- direct output equals narrow plus residual;
- highpass output preserves 0-22 Hz input with NRMSE below `1e-5`;
- section selection uses header-derived inline/crossline geometry;
- output metadata records lock SHA and `reference_read_after_lock=True`.

- [ ] **Step 2: Run tests and verify RED**

```powershell
D:\Anaconda\python.exe -m pytest 21_leave_one_well_calibration\tests\test_inference.py -q
```

- [ ] **Step 3: Implement inference**

CLI:

```text
05_predict_heldout.py --fold fold_well1
```

The script must:

1. verify fold lock;
2. load checkpoint;
3. validate requested held-out sections;
4. read SEG-Y;
5. form narrow input;
6. predict blended residual;
7. output direct and highpass recombinations;
8. save the held-out reference only after authorization.

- [ ] **Step 4: Run tests**

```powershell
D:\Anaconda\python.exe -m pytest 21_leave_one_well_calibration\tests\test_inference.py -q
```

- [ ] **Step 5: Run four held-out predictions**

```powershell
foreach ($fold in 'fold_well1','fold_well2','fold_well3','fold_well4') {
  D:\Anaconda\python.exe 21_leave_one_well_calibration\code\05_predict_heldout.py --fold $fold *> "21_leave_one_well_calibration\logs\predict_$fold.log"
}
```

- [ ] **Step 6: Commit code**

```powershell
git add F3_auxiliary_experiment/21_leave_one_well_calibration/code/05_predict_heldout.py F3_auxiliary_experiment/21_leave_one_well_calibration/tests/test_inference.py
git commit -m "feat: predict locked heldout well sections"
```

## Task 12: Evaluate Folds and Produce Final Comparison

**Files:**
- Create: `21_leave_one_well_calibration/code/06_evaluate_folds.py`
- Create: `21_leave_one_well_calibration/tests/test_aggregate_evaluation.py`

- [ ] **Step 1: Write failing aggregation tests**

Using small synthetic fold metrics, assert:

- each well contributes exactly one inline and one crossline result;
- aggregate metrics are weighted by sample count;
- no fold is duplicated or omitted;
- success requires both inline and crossline correlation improvement;
- success requires positive residual correlation and phase;
- success requires at least three wells not below their baseline;
- spectral improvement alone does not pass.

- [ ] **Step 2: Run tests and verify RED**

```powershell
D:\Anaconda\python.exe -m pytest 21_leave_one_well_calibration\tests\test_aggregate_evaluation.py -q
```

- [ ] **Step 3: Implement per-fold and aggregate evaluation**

Generate:

```text
figures/预测评价/fold_wellN_inline_sections.png
figures/预测评价/fold_wellN_crossline_sections.png
figures/预测评价/fold_wellN_spectra.png
figures/预测评价/leave_one_well_all_sections.png
figures/预测评价/leave_one_well_all_spectra.png
figures/预测评价/leave_one_well_final_report.txt
data/评价结果/fold_wellN_metrics.npy
data/评价结果/leave_one_well_aggregate_metrics.npy
```

Use shared color scales for narrow, direct, highpass, and reference panels. Include low-pass baseline, experiment 18 upper bound, and experiment 20 no-wide diagnostic results with explicit data-condition labels.

- [ ] **Step 4: Run tests and evaluation**

```powershell
D:\Anaconda\python.exe -m pytest 21_leave_one_well_calibration\tests\test_aggregate_evaluation.py -q
D:\Anaconda\python.exe 21_leave_one_well_calibration\code\06_evaluate_folds.py *> 21_leave_one_well_calibration\logs\evaluate_all_folds.log
```

- [ ] **Step 5: Inspect figures**

Open all aggregate section and spectrum figures. Confirm:

- no blank or incorrectly stitched sections;
- held-out geometry is correct;
- labels and titles identify the fold and held-out well;
- shared color scales are used;
- spectra show 35-80 Hz clearly;
- no panel overlap or clipped text.

- [ ] **Step 6: Write final experiment conclusion**

Update `docs/23_21号实验改进思路.txt` only with final new results:

- four-fold data boundary;
- common-pretrain SHA;
- each fold checkpoint SHA;
- per-well and aggregate metrics;
- direct versus highpass conclusion;
- success/failure against predeclared criteria;
- limitations relative to experiment 18.

- [ ] **Step 7: Commit code and tracked docs**

```powershell
git add F3_auxiliary_experiment/21_leave_one_well_calibration/code/06_evaluate_folds.py F3_auxiliary_experiment/21_leave_one_well_calibration/tests/test_aggregate_evaluation.py F3_auxiliary_experiment/docs/23_21号实验改进思路.txt
git commit -m "feat: evaluate leave-one-well calibration experiment"
```

## Task 13: Full Verification and Branch Integration

**Files:**
- Verify all experiment 21 files
- Do not modify unrelated deleted files under `14_latest_geometry_fixed`

- [ ] **Step 1: Compile all Python files**

```powershell
Get-ChildItem 21_leave_one_well_calibration\code -Filter *.py | ForEach-Object {
  D:\Anaconda\python.exe -m py_compile $_.FullName
}
```

Expected: exit code 0 for every file.

- [ ] **Step 2: Run the complete test suite**

```powershell
D:\Anaconda\python.exe -m pytest 21_leave_one_well_calibration\tests -q
```

Expected: all tests pass.

- [ ] **Step 3: Run leakage audit**

Verify:

```text
four folds exist
each well is held out once
each fold has three calibration wells
held-out guard overlap is zero
train/validation coordinate overlap is zero
all fold locks use the same common-pretrain SHA
no experiment 18 checkpoint appears in metadata
no held-out reference was read before lock verification
```

- [ ] **Step 4: Verify result artifacts**

Check every fold has:

```text
best_model.pth
fold_lock.json
inline prediction arrays
crossline prediction arrays
per-fold metrics
section figures
spectrum figures
```

Check the aggregate report and metrics are non-empty.

- [ ] **Step 5: Review Git diff**

```powershell
git diff --check
git status --short
```

Confirm unrelated existing deletions under `14_latest_geometry_fixed` were neither staged nor modified by this experiment.

- [ ] **Step 6: Request code review**

Review the implementation against:

```text
docs/superpowers/specs/2026-06-23-leave-one-well-local-calibration-design.md
```

Fix all critical and important findings, then rerun compilation and tests.

- [ ] **Step 7: Final commit**

```powershell
git add F3_auxiliary_experiment/21_leave_one_well_calibration F3_auxiliary_experiment/docs/23_21号实验改进思路.txt
git commit -m "chore: finalize experiment 21 verification"
```

Keep generated arrays, checkpoints, logs, and large figures local according to the repository ignore rules.
