# Wideband Direct Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create experiment 17, train `UNetCBAM` for 200 epochs to map noisy low-pass seismic patches directly to clean wideband patches, and evaluate four well inline and crossline sections.

**Architecture:** Copy only experiment-specific code structure from experiment 16 while continuing to import shared resources from root `shared_code/` and `shared_data/`. Replace residual labels, zero-mean model output, residual losses, and residual recombination with direct wideband supervision, a five-part wideband loss, and direct denormalization at inference.

**Tech Stack:** Python 3, NumPy, SciPy, PyTorch, Matplotlib, pytest, SEG-Y geometry reader

---

### Task 1: Scaffold experiment 17 without shared duplicates

**Files:**
- Create: `17_wideband_direct_prediction/README.md`
- Create: `17_wideband_direct_prediction/code/config.py`
- Create: `17_wideband_direct_prediction/code/common.py`
- Create: `17_wideband_direct_prediction/tests/test_wideband_pipeline.py`
- Create directories: `17_wideband_direct_prediction/data/`, `figures/`, `logs/`

- [ ] **Step 1: Create a failing layout test**

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_experiment_does_not_duplicate_shared_resources():
    assert (ROOT / "code" / "config.py").exists()
    assert (ROOT / "code" / "common.py").exists()
    forbidden = {
        "model.py",
        "segy_reader.py",
        "signal_utils.py",
        "01_rematch_wells_estimate_wavelets.py",
    }
    assert not forbidden.intersection(p.name for p in (ROOT / "code").glob("*.py"))
    assert not (ROOT / "data" / "井数据").exists()
```

- [ ] **Step 2: Run the layout test and verify failure**

Run:

```powershell
D:\Anaconda\python.exe -m pytest 17_wideband_direct_prediction/tests/test_wideband_pipeline.py::test_experiment_does_not_duplicate_shared_resources -v
```

Expected: FAIL because experiment 17 does not exist or cannot yet satisfy its layout.

- [ ] **Step 3: Create the experiment structure**

Copy experiment-specific utilities from 16 into `17_wideband_direct_prediction/code/`, but do not copy shared code or generated data. Set:

```python
ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
SOURCE_DATA_DIR = WORKSPACE_ROOT / "shared_data"
SEGY_PATH = WORKSPACE_ROOT / "Rawdata" / "Seismic_data.sgy"
PATCH_SIZE = 256
PATCH_STRIDE = 128
NOISE_LEVELS = [0.01, 0.03]
```

Create output subdirectories:

```text
data/样本数据
data/模型检查点
data/预测结果
data/评价结果
figures/合成剖面
figures/训练样本
figures/频谱分析
figures/预测评价
logs
```

- [ ] **Step 4: Run the layout test**

Expected: PASS.

### Task 2: Generate clean wideband labels

**Files:**
- Create: `17_wideband_direct_prediction/code/02_generate_synthetic_dataset.py`
- Modify: `17_wideband_direct_prediction/tests/test_wideband_pipeline.py`

- [ ] **Step 1: Add failing label-semantics tests**

```python
import numpy as np

from wideband_targets import prepare_training_pair


def test_label_is_clean_wideband_not_residual():
    clean_narrow = np.array([[1.0, -2.0]], dtype=np.float32)
    clean_wide = np.array([[1.5, -3.0]], dtype=np.float32)
    input_norm, label_norm, scale = prepare_training_pair(
        clean_narrow, clean_wide, noise_level=0.0, rng=np.random.default_rng(1)
    )
    expected = np.clip(clean_wide / scale, -1.0, 1.0)
    residual = expected - np.clip(clean_narrow / scale, -1.0, 1.0)
    np.testing.assert_allclose(label_norm, expected)
    assert not np.allclose(label_norm, residual)


def test_input_noise_does_not_enter_label():
    clean_narrow = np.linspace(-1, 1, 16, dtype=np.float32).reshape(4, 4)
    clean_wide = clean_narrow * 1.4
    _, label_a, _ = prepare_training_pair(
        clean_narrow, clean_wide, noise_level=0.01, rng=np.random.default_rng(2)
    )
    _, label_b, _ = prepare_training_pair(
        clean_narrow, clean_wide, noise_level=0.03, rng=np.random.default_rng(3)
    )
    np.testing.assert_allclose(label_a, label_b)
```

- [ ] **Step 2: Run tests and verify import failure**

Run:

```powershell
D:\Anaconda\python.exe -m pytest 17_wideband_direct_prediction/tests/test_wideband_pipeline.py -v
```

Expected: FAIL because `wideband_targets.prepare_training_pair` does not exist.

- [ ] **Step 3: Implement target preparation**

Create `17_wideband_direct_prediction/code/wideband_targets.py`:

```python
import numpy as np


def prepare_training_pair(clean_narrow, clean_wide, noise_level, rng):
    scale = max(float(np.percentile(np.abs(clean_narrow), 99)), 1e-8)
    clean_input_norm = np.clip(clean_narrow / scale, -1.0, 1.0).astype(np.float32)
    label_norm = np.clip(clean_wide / scale, -1.0, 1.0).astype(np.float32)
    input_norm = clean_input_norm.copy()
    if noise_level > 0:
        noise = rng.normal(0.0, noise_level, input_norm.shape).astype(np.float32)
        input_norm = np.clip(input_norm + noise, -1.0, 1.0)
    return input_norm.astype(np.float32), label_norm, scale
```

Use it in the copied synthetic generator. Save `wide_norm` directly as section and patch labels. Set every metadata occurrence to:

```text
label_type = wide_band
normalization = per_section_p99_abs_clean_narrow
```

Keep all 16 geometry, smoothing, interpolation, wavelet, noise and patch rules unchanged.

- [ ] **Step 4: Run target tests**

Expected: PASS.

- [ ] **Step 5: Generate the complete dataset**

Run:

```powershell
Set-Location 17_wideband_direct_prediction
D:\Anaconda\python.exe code\02_generate_synthetic_dataset.py 2>&1 | Tee-Object logs\generate_samples.log
```

Expected:

```text
num_sections = 66
num_patches = 1386
num_train = 1109
num_val = 277
```

- [ ] **Step 6: Verify generated data**

Load metadata and arrays with `D:\Anaconda\python.exe` and assert:

```python
meta["label_type"] == "wide_band"
meta["normalization"] == "per_section_p99_abs_clean_narrow"
meta["stride"] == 128
meta["num_sections"] == 66
train_inputs.shape == train_labels.shape
not np.allclose(train_labels[0], train_inputs[0])
```

Also use `rg` to confirm there is no `np.flip(... axis=0)` and no residual label assignment.

### Task 3: Implement the direct wideband model and loss

**Files:**
- Create: `17_wideband_direct_prediction/code/wideband_training.py`
- Create: `17_wideband_direct_prediction/code/01_train.py`
- Modify: `17_wideband_direct_prediction/tests/test_wideband_pipeline.py`

- [ ] **Step 1: Add failing model and loss tests**

```python
import torch

from wideband_training import WidebandModel, WidebandCompositeLoss


def test_model_output_is_not_forced_to_zero_mean():
    model = WidebandModel(base_c=4)
    model.net = torch.nn.Identity()
    x = torch.full((1, 1, 16, 16), 0.25)
    output = model(x)
    assert torch.allclose(output, x)
    assert output.mean().item() == 0.25


def test_composite_loss_is_zero_for_exact_prediction():
    criterion = WidebandCompositeLoss()
    target = torch.randn(2, 1, 32, 32)
    loss, parts = criterion(target, target)
    assert loss.item() < 1e-6
    assert set(parts) == {"total", "waveform", "spectrum", "phase", "gradient", "low_frequency"}
```

- [ ] **Step 2: Run tests and verify import failure**

Expected: FAIL because `wideband_training.py` does not exist.

- [ ] **Step 3: Implement the direct model**

```python
class WidebandModel(nn.Module):
    def __init__(self, base_c=32):
        super().__init__()
        self.net = UNetCBAM(base_c=base_c)

    def forward(self, x):
        return self.net(x)
```

Do not subtract the patch mean and do not add the input to the output.

- [ ] **Step 4: Implement the composite loss**

Implement:

```python
total = (
    waveform
    + 0.5 * spectrum
    + 0.2 * phase
    + 0.2 * gradient
    + 0.5 * low_frequency
)
```

Use:

- L1 for waveform.
- Normalized FFT magnitude L1 over 25–80 Hz for spectrum.
- `1 - cos(phase_pred - phase_target)` over 25–80 Hz for phase.
- L1 between `x[..., 1:, :] - x[..., :-1, :]` for time gradient.
- Differentiable FFT mask for 3–35 Hz followed by inverse FFT and L1 for low-frequency consistency.

- [ ] **Step 5: Run model and loss tests**

Expected: PASS.

- [ ] **Step 6: Implement the 200-epoch trainer**

Dataset returns only:

```text
x = noisy low-pass input
y = clean wideband label
```

Allow left-right flipping of both arrays. Use:

```text
epochs = 200
batch_size = 4
Adam lr = 1e-3
ReduceLROnPlateau patience = 10
seed = 42
```

Save checkpoints under `data/模型检查点/`. Configure inference to read this directory directly. Store `target_type="wide_band"` in checkpoints.

- [ ] **Step 7: Run a one-epoch smoke training**

Run:

```powershell
D:\Anaconda\python.exe code\01_train.py --epochs 1 --base-c 4 --checkpoint-dir data\smoke_checkpoints
```

Expected: one complete epoch, finite train/validation losses, and loadable smoke checkpoint.

### Task 4: Implement direct wideband inference

**Files:**
- Create: `17_wideband_direct_prediction/code/02_predict_f3.py`
- Modify: `17_wideband_direct_prediction/tests/test_wideband_pipeline.py`

- [ ] **Step 1: Add a failing inference denormalization test**

```python
import numpy as np

from wideband_inference import denormalize_wide_prediction


def test_prediction_is_directly_denormalized_without_residual_recombination():
    pred_norm = np.array([[0.5, -0.25]], dtype=np.float32)
    result = denormalize_wide_prediction(pred_norm, narrow_scale=20.0)
    np.testing.assert_allclose(result, [[10.0, -5.0]])
```

- [ ] **Step 2: Run test and verify import failure**

Expected: FAIL because `wideband_inference.py` does not exist.

- [ ] **Step 3: Implement direct denormalization**

Create:

```python
def denormalize_wide_prediction(pred_norm, narrow_scale):
    return np.asarray(pred_norm, dtype=np.float32) * float(narrow_scale)
```

Adapt the geometry-aware sliding-window predictor from experiment 16:

- Support `--section-axis inline|crossline`.
- Support explicit inline and crossline number lists.
- Normalize each complete low-pass section by its P99.
- Blend predicted wideband patches.
- Save low-pass input, direct wideband prediction, wideband reference and metadata.
- Do not bandpass, highpass, gain-correct or add the low-pass section to the model output.

- [ ] **Step 4: Run inference unit tests**

Expected: PASS.

- [ ] **Step 5: Run inference smoke test using the smoke checkpoint**

Predict one inline and verify output shape is `[1, time, crossline]`, arrays are finite, and metadata states `prediction_type="direct_wide_band"`.

### Task 5: Train the full model for 200 epochs

**Files:**
- Produce: `17_wideband_direct_prediction/data/模型检查点/best_model.pth`
- Produce: `17_wideband_direct_prediction/data/模型检查点/last_model.pth`
- Produce: `17_wideband_direct_prediction/logs/training_history.npy`
- Produce: `17_wideband_direct_prediction/logs/training_curves.png`
- Produce: `17_wideband_direct_prediction/logs/train_200.log`

- [ ] **Step 1: Remove smoke-only outputs**

Delete only `17_wideband_direct_prediction/data/smoke_checkpoints/` after verifying its resolved path remains inside experiment 17.

- [ ] **Step 2: Start full training**

Run:

```powershell
Set-Location 17_wideband_direct_prediction
D:\Anaconda\python.exe code\01_train.py --epochs 200 --batch-size 4 --lr 0.001 2>&1 | Tee-Object logs\train_200.log
```

- [ ] **Step 3: Verify training completion**

Assert:

```text
history epoch count = 200
all recorded losses are finite
best_model.pth loads
last_model.pth loads
checkpoint target_type = wide_band
```

### Task 6: Predict four well inline and crossline sections

**Files:**
- Produce prediction arrays and metadata under `17_wideband_direct_prediction/data/预测结果/`

- [ ] **Step 1: Predict four inline sections**

Run:

```powershell
D:\Anaconda\python.exe code\02_predict_f3.py --output-prefix wide17_200_inline_wells --section-axis inline --inline-values 244,362,442,722
```

- [ ] **Step 2: Predict four crossline sections**

Run:

```powershell
D:\Anaconda\python.exe code\02_predict_f3.py --output-prefix wide17_200_crossline_wells --section-axis crossline --crossline-values 336,387,848,1007
```

- [ ] **Step 3: Verify geometry metadata**

Assert:

```text
inline shape = [4, time, crossline]
inline section numbers = 244,362,442,722
crossline shape = [4, time, inline]
crossline section numbers = 336,387,848,1007
```

### Task 7: Evaluate and document the result

**Files:**
- Create: `17_wideband_direct_prediction/code/03_evaluate.py`
- Produce: `17_wideband_direct_prediction/figures/预测评价/*.png`
- Produce: `17_wideband_direct_prediction/data/评价结果/*.npy`
- Produce: `17_wideband_direct_prediction/figures/预测评价/wide17_200_final_analysis.txt`
- Modify: `17_wideband_direct_prediction/README.md`

- [ ] **Step 1: Implement baseline and prediction evaluation**

For inline and crossline runs, calculate overall and per-section:

```text
MAE
RMSE
PSNR
Correlation
35–80 Hz energy ratio
Spectrum L1 distance
```

Compare:

```text
low-pass input vs reference
direct wideband prediction vs reference
```

- [ ] **Step 2: Generate figures**

For each direction generate:

- Four-section comparison with shared color scale per row.
- Low-pass, predicted wideband, reference and error panels.
- Average amplitude spectrum with 35–80 Hz highlighted.
- Training curves.

- [ ] **Step 3: Write the final analysis**

The report must state:

- Best epoch and validation loss.
- Whether spatial metrics improve over the low-pass baseline.
- Whether spectrum L1 and 35–80 Hz ratio improve.
- Inline and crossline differences.
- Whether the result indicates true phase-consistent bandwidth extension or mainly added spectral energy.

- [ ] **Step 4: Run final verification**

Run:

```powershell
D:\Anaconda\python.exe -m pytest 17_wideband_direct_prediction/tests -v
D:\Anaconda\python.exe -m compileall -q 17_wideband_direct_prediction\code
```

Verify all required arrays, checkpoints, figures, metadata and reports exist and are non-empty.

- [ ] **Step 5: Commit experiment code and documentation**

Stage only experiment 17, its tests, README and final report. Do not stage unrelated existing worktree changes.
