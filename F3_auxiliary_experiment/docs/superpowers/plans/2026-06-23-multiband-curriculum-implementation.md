# 20号多频带课程联合训练 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 创建20号实验，使用F3窄频动态多频带遮蔽与测井合成残差联合课程训练，在不使用F3宽频监督的前提下优先提高预测相位和相关性。

**Architecture:** 基于19号单通道UNet-CBAM残差网络复用频率投影、相位损失、泄露防护和模型锁。新增动态遮蔽数据集、双域batch课程调度、分域验证指标和门控checkpoint选择；只有通过F3已知频带验证门槛的模型才能锁定并进入一次性F3宽频盲评。

**Tech Stack:** Python 3、NumPy、SciPy、PyTorch、Matplotlib、pytest、项目根目录 `shared_code`。

---

## 文件结构

```text
20_curriculum_multiband/
  README.md
  code/
    config.py                         # 路径、频带、课程阶段和门控参数
    leakage_guard.py                  # 训练路径审计与模型哈希锁
    phase_model.py                    # UNet-CBAM残差模型与频率投影
    phase_loss.py                     # 分域频带损失
    phase_metrics.py                  # 相关性、相位、包络和泄漏指标
    masking.py                        # 动态低通/带阻任务生成
    datasets.py                       # F3动态数据集和合成残差数据集
    curriculum.py                     # 阶段定义、batch来源调度和门控选模
    01_prepare_f3_patches.py          # 只缓存干净F3窄频基础patch
    02_generate_well_synthetic.py     # 生成测井约束合成样本
    03_train_curriculum.py            # 300轮联合课程训练
    04_predict_locked_f3.py           # 锁定后输出direct/highpass结果
    05_blind_evaluate.py              # 四井inline/crossline盲评
  tests/
    test_masking.py
    test_datasets.py
    test_curriculum.py
    test_leakage_and_lock.py
    test_phase_loss.py
    test_phase_metrics.py
    test_phase_model.py
```

## Task 1: 创建20号目录与配置

**Files:**
- Create: `20_curriculum_multiband/README.md`
- Create: `20_curriculum_multiband/code/config.py`
- Create: `20_curriculum_multiband/tests/test_config.py`
- Reuse: `19_no_wide_supervision/code/phase_model.py`
- Reuse: `19_no_wide_supervision/code/phase_metrics.py`
- Reuse: `19_no_wide_supervision/code/leakage_guard.py`

- [ ] **Step 1: 编写失败的配置测试**

```python
from config import (
    CURRICULUM_STAGES,
    F3_MASK_TASKS,
    FINAL_PROJECTOR,
    TOTAL_EPOCHS,
)


def test_curriculum_configuration_is_locked():
    assert TOTAL_EPOCHS == 300
    assert [(s["start"], s["end"], s["f3_ratio"], s["synthetic_ratio"])
            for s in CURRICULUM_STAGES] == [
        (1, 60, 1, 0),
        (61, 180, 2, 1),
        (181, 300, 1, 1),
    ]
    assert set(F3_MASK_TASKS) == {"A", "B", "C", "D"}
    assert FINAL_PROJECTOR == {
        "low_stop": 32.0,
        "low_pass": 38.0,
        "high_pass": 85.0,
        "high_stop": 100.0,
    }
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```powershell
D:\Anaconda\python.exe -m pytest 20_curriculum_multiband\tests\test_config.py -q
```

Expected: FAIL，原因是 `config.py` 尚不存在。

- [ ] **Step 3: 创建配置和目录**

`config.py` 至少定义：

```python
ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
SOURCE_DATA_DIR = WORKSPACE_ROOT / "shared_data"
SEGY_PATH = WORKSPACE_ROOT / "Rawdata" / "Seismic_data.sgy"

DT = 0.004
PATCH_SIZE = 256
PATCH_STRIDE = 128
NARROW_BAND = (3.0, 6.0, 25.0, 35.0)
NOISE_LEVELS = (0.0, 0.01, 0.03)
TOTAL_EPOCHS = 300

F3_MASK_TASKS = {
    "A": {"kind": "lowpass", "stop": (11.0, 13.0), "pass": (15.0, 17.0),
          "target_high": 20.0},
    "B": {"kind": "lowpass", "stop": (17.0, 19.0), "pass": (21.0, 23.0),
          "target_high": 28.0},
    "C": {"kind": "lowpass", "stop": (21.0, 23.0), "pass": (25.0, 27.0),
          "target_high": 35.0},
    "D": {"kind": "bandstop", "width": (4.0, 10.0),
          "center": (12.0, 30.0)},
}

CURRICULUM_STAGES = (
    {"name": "f3_foundation", "start": 1, "end": 60,
     "f3_ratio": 1, "synthetic_ratio": 0, "lr": 5e-4},
    {"name": "f3_priority_joint", "start": 61, "end": 180,
     "f3_ratio": 2, "synthetic_ratio": 1, "lr": 3e-4},
    {"name": "balanced_joint", "start": 181, "end": 300,
     "f3_ratio": 1, "synthetic_ratio": 1, "lr": 1e-4},
)
```

`ensure_dirs()` 创建规格中的数据、图件和日志目录。将19号的三个稳定模块复制到20号，
只修改实验编号和路径，不读取19号checkpoint或训练数据。

- [ ] **Step 4: 运行配置测试**

Run:

```powershell
D:\Anaconda\python.exe -m pytest 20_curriculum_multiband\tests\test_config.py -q
```

Expected: `1 passed`。

- [ ] **Step 5: 提交**

```powershell
git add 20_curriculum_multiband
git commit -m "feat: scaffold curriculum experiment 20"
```

## Task 2: 实现F3动态多频带遮蔽

**Files:**
- Create: `20_curriculum_multiband/code/masking.py`
- Create: `20_curriculum_multiband/tests/test_masking.py`

- [ ] **Step 1: 编写遮蔽闭合与频带边界测试**

```python
def test_lowpass_task_closes_to_known_target():
    known = make_signal([8, 18, 27], dt=0.004, samples=256)
    pair = make_masked_pair(known, dt=0.004, task_name="C",
                            rng=np.random.default_rng(7))
    assert np.allclose(pair.input_norm + pair.label_norm,
                       pair.target_norm, atol=2e-5)
    assert spectral_energy(pair.label_norm, 36, 100, 0.004) < 1e-5


def test_bandstop_label_is_only_removed_known_band():
    known = make_signal([10, 18, 26, 33], dt=0.004, samples=256)
    pair = make_masked_pair(known, dt=0.004, task_name="D",
                            rng=np.random.default_rng(11))
    assert np.allclose(pair.input_norm + pair.label_norm,
                       pair.target_norm, atol=2e-5)
    assert pair.target_low < pair.target_high <= 35.0
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```powershell
D:\Anaconda\python.exe -m pytest 20_curriculum_multiband\tests\test_masking.py -q
```

Expected: FAIL，缺少 `make_masked_pair`。

- [ ] **Step 3: 实现动态任务**

定义：

```python
@dataclass(frozen=True)
class MaskedPair:
    input_norm: np.ndarray
    label_norm: np.ndarray
    target_norm: np.ndarray
    scale: float
    task_name: str
    target_low: float
    target_high: float
    projector: dict


def make_masked_pair(known_narrow, dt, task_name, rng, noise_level=0.0):
    ...
```

实现要求：

- scale 始终为原始 `known_narrow` 的绝对值P99；
- A/B/C动态采样截止频率并低通，标签为已知目标低通减输入；
- D使用余弦带阻掩码，标签为原窄频减带阻输入；
- 噪声只加入 `input_norm`；
- `target_norm` 保持无噪声；
- projector只覆盖本任务目标频带；
- 任何任务均不得在35 Hz以上产生标签目标。

- [ ] **Step 4: 运行测试**

Run:

```powershell
D:\Anaconda\python.exe -m pytest 20_curriculum_multiband\tests\test_masking.py -q
```

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add 20_curriculum_multiband/code/masking.py 20_curriculum_multiband/tests/test_masking.py
git commit -m "feat: add dynamic F3 multiband masking"
```

## Task 3: 缓存F3基础窄频patch并建立动态数据集

**Files:**
- Create: `20_curriculum_multiband/code/datasets.py`
- Create: `20_curriculum_multiband/code/01_prepare_f3_patches.py`
- Create: `20_curriculum_multiband/tests/test_datasets.py`

- [ ] **Step 1: 编写空间分组、动态重复和增强测试**

```python
def test_f3_dataset_changes_task_without_changing_clean_patch(tmp_path):
    save_patch_file(tmp_path, count=2)
    dataset = F3MaskedDataset(tmp_path, split="train", seed=42)
    first = dataset.sample_at(0, epoch=1)
    second = dataset.sample_at(0, epoch=2)
    assert first["clean_id"] == second["clean_id"]
    assert first["task_name"] != second["task_name"] or not np.allclose(
        first["input"], second["input"]
    )


def test_only_horizontal_flip_is_used(tmp_path):
    source = inspect.getsource(F3MaskedDataset)
    assert "axis=0" not in source
    assert "axis=1" in source
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```powershell
D:\Anaconda\python.exe -m pytest 20_curriculum_multiband\tests\test_datasets.py -q
```

Expected: FAIL，数据集尚不存在。

- [ ] **Step 3: 实现基础patch准备脚本**

`01_prepare_f3_patches.py`：

- 正确读取SEG-Y几何；
- 对完整剖面执行3-6-25-35 Hz低通；
- 使用 `256 x 256`、步长128、`cover_last=True`；
- 按inline/crossline平面分组划分训练和验证，禁止相邻patch随机拆分；
- 默认缓存训练4000个、验证600个基础patch；
- 保存 `clean_narrow.npy`、`metadata.npy` 和 `manifest.json`；
- manifest记录 `uses_f3_wide_target=false` 和空间分组。

- [ ] **Step 4: 实现数据集**

`F3MaskedDataset.__getitem__` 使用 `(seed, epoch, index)` 创建可复现RNG，动态选择
A/B/C/D、频率抖动和输入噪声。返回：

```python
{
    "input": tensor[1, 256, 256],
    "label": tensor[1, 256, 256],
    "target": tensor[1, 256, 256],
    "projector": tensor[4],
    "domain": "f3",
    "task_name": "A" | "B" | "C" | "D",
}
```

`SyntheticResidualDataset` 读取合成输入和标签，仅允许左右翻转。

- [ ] **Step 5: 运行测试和小规模样本烟雾测试**

Run:

```powershell
D:\Anaconda\python.exe -m pytest 20_curriculum_multiband\tests\test_datasets.py -q
D:\Anaconda\python.exe 20_curriculum_multiband\code\01_prepare_f3_patches.py --max-train 32 --max-val 8 --smoke
```

Expected: 测试PASS；生成32个训练和8个验证基础patch，manifest显示无宽频目标。

- [ ] **Step 6: 提交**

```powershell
git add 20_curriculum_multiband/code/datasets.py 20_curriculum_multiband/code/01_prepare_f3_patches.py 20_curriculum_multiband/tests/test_datasets.py
git commit -m "feat: prepare dynamic F3 self-supervision patches"
```

## Task 4: 生成带估计子波的测井合成样本

**Files:**
- Create: `20_curriculum_multiband/code/02_generate_well_synthetic.py`
- Create: `20_curriculum_multiband/tests/test_synthetic_samples.py`
- Reference: `16_geometry_realistic_samples/code/02_generate_synthetic_dataset.py`

- [ ] **Step 1: 编写合成闭合、尺度和分组测试**

```python
def test_synthetic_label_uses_clean_narrow_not_noisy_input():
    sample = build_test_sample(noise_level=0.03, seed=5)
    assert np.allclose(
        sample.clean_narrow_norm + sample.label_norm,
        sample.wide_norm,
        atol=2e-5,
    )
    assert not np.allclose(
        sample.input_norm + sample.label_norm,
        sample.wide_norm,
    )
    assert sample.scale_source == "p99_abs_clean_narrow"


def test_group_split_keeps_section_out_of_both_splits():
    train, val = grouped_split(fake_section_metadata(), seed=42)
    assert set(m["section_id"] for m in train).isdisjoint(
        set(m["section_id"] for m in val)
    )
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```powershell
D:\Anaconda\python.exe -m pytest 20_curriculum_multiband\tests\test_synthetic_samples.py -q
```

Expected: FAIL。

- [ ] **Step 3: 实现合成样本生成**

从16号脚本提取并保留：

- 真实井位置和2/3/4井线性投影；
- 线性时间重采样；
- 倾斜断层与中深层构造扰动；
- `uniform_filter1d(size=5, axis=1)`；
- 宽频低通生成窄频输入；
- 噪声仅加入归一化输入；
- 标签为无噪声宽频减无噪声窄频。

子波库由以下两部分组成：

```python
estimated_bank = normalize_each(np.load(SOURCE_DATA_DIR / "well_estimated_wavelets.npy"))
bspline_bank = build_bspline_wavelets(...)
wavelet_bank = phase_preserving_augment(estimated_bank, bspline_bank)
```

估计子波不得只用于长度统计。每个合成剖面的metadata记录子波来源、井组合、结构种子、
噪声水平和section_id。默认输出2000-3000个patch。

- [ ] **Step 4: 运行测试和66样本烟雾生成**

Run:

```powershell
D:\Anaconda\python.exe -m pytest 20_curriculum_multiband\tests\test_synthetic_samples.py -q
D:\Anaconda\python.exe 20_curriculum_multiband\code\02_generate_well_synthetic.py --max-patches 66 --smoke
```

Expected: PASS；标签闭合误差小于 `2e-5`；训练和验证section_id无交集。

- [ ] **Step 5: 提交**

```powershell
git add 20_curriculum_multiband/code/02_generate_well_synthetic.py 20_curriculum_multiband/tests/test_synthetic_samples.py
git commit -m "feat: generate well-constrained synthetic samples"
```

## Task 5: 实现分域损失、课程调度和门控选模

**Files:**
- Create: `20_curriculum_multiband/code/phase_loss.py`
- Create: `20_curriculum_multiband/code/curriculum.py`
- Create: `20_curriculum_multiband/tests/test_curriculum.py`
- Modify: `20_curriculum_multiband/tests/test_phase_loss.py`

- [ ] **Step 1: 编写课程比例和门控测试**

```python
def test_domain_schedule_matches_three_stages():
    assert domain_cycle(20) == ("f3",)
    assert domain_cycle(100) == ("f3", "f3", "synthetic")
    assert domain_cycle(240) == ("f3", "synthetic")


def test_gate_rejects_good_synthetic_but_bad_f3():
    selector = GatedCheckpointSelector()
    rejected = selector.consider(
        epoch=120,
        f3={"correlation": 0.84, "phase": 0.90, "leakage": 0.01},
        synthetic={"residual_correlation": 0.95},
    )
    assert rejected is False


def test_gate_prefers_synthetic_after_f3_thresholds():
    selector = GatedCheckpointSelector()
    assert selector.consider(
        120,
        {"correlation": 0.87, "phase": 0.83, "leakage": 0.02},
        {"residual_correlation": 0.70},
    )
    assert selector.consider(
        121,
        {"correlation": 0.86, "phase": 0.82, "leakage": 0.02},
        {"residual_correlation": 0.74},
    )
    assert selector.best_epoch == 121
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```powershell
D:\Anaconda\python.exe -m pytest 20_curriculum_multiband\tests\test_curriculum.py -q
```

Expected: FAIL。

- [ ] **Step 3: 实现课程调度**

`domain_cycle(epoch)` 严格返回：

```python
if epoch <= 60:
    return ("f3",)
if epoch <= 180:
    return ("f3", "f3", "synthetic")
return ("f3", "synthetic")
```

`GatedCheckpointSelector` 门槛：

```python
F3_MIN_CORRELATION = 0.85
F3_MIN_PHASE = 0.80
F3_MAX_LEAKAGE = 0.03
SYNTHETIC_TIE_TOLERANCE = 0.01
```

只有通过三个F3门槛后才比较合成残差相关性；差值不超过0.01时选择F3相关性更高者。

- [ ] **Step 4: 实现分域损失**

统一接口：

```python
loss, parts = criterion(
    residual_prediction,
    residual_target,
    input_data,
    target_wide,
    domain="f3" or "synthetic",
    projector=projector,
)
```

F3损失权重：

```text
residual=0.25, correlation=0.35, complex_stft=0.25,
lateral=0.10, leakage=0.05
```

合成损失权重：

```text
residual=0.20, correlation=0.30, complex_stft=0.25,
wide_waveform=0.15, lateral=0.05, leakage=0.05
```

所有相关性和STFT损失必须先应用当前样本的目标频带projector。

- [ ] **Step 5: 运行课程和损失测试**

Run:

```powershell
D:\Anaconda\python.exe -m pytest 20_curriculum_multiband\tests\test_curriculum.py 20_curriculum_multiband\tests\test_phase_loss.py -q
```

Expected: PASS。

- [ ] **Step 6: 提交**

```powershell
git add 20_curriculum_multiband/code/curriculum.py 20_curriculum_multiband/code/phase_loss.py 20_curriculum_multiband/tests
git commit -m "feat: add gated joint curriculum training logic"
```

## Task 6: 实现并烟雾验证300轮训练器

**Files:**
- Create: `20_curriculum_multiband/code/03_train_curriculum.py`
- Create: `20_curriculum_multiband/tests/test_training_smoke.py`

- [ ] **Step 1: 编写单epoch双域训练烟雾测试**

```python
def test_one_joint_epoch_records_both_domains(tmp_path):
    result = run_training(
        epochs=1,
        f3_dataset=tiny_f3_dataset(),
        synthetic_dataset=tiny_synthetic_dataset(),
        output_dir=tmp_path,
        device="cpu",
    )
    assert result.history[0]["f3_train"]["correlation"] >= -1.0
    assert "synthetic_val" in result.history[0]
    assert result.history[0]["uses_f3_wide_target"] is False
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```powershell
D:\Anaconda\python.exe -m pytest 20_curriculum_multiband\tests\test_training_smoke.py -q
```

Expected: FAIL。

- [ ] **Step 3: 实现训练器**

训练器必须：

- 每个epoch调用 `f3_dataset.set_epoch(epoch)`；
- 按 `domain_cycle(epoch)` 交替取batch；
- 不把两个域的不同projector混在同一batch；
- 每轮分别运行F3验证和合成验证；
- 记录相关性、相位、包络、泄漏、频谱和波形指标；
- 学习率在阶段边界重设，并使用 `ReduceLROnPlateau` 衰减；
- 仅由 `GatedCheckpointSelector` 决定 `best_model.pth`；
- 若300轮结束未通过门槛，写出 `training_failed_gate.json`，不生成模型锁；
- 通过门槛后生成包含SHA256、验证指标和 `uses_f3_wide_target=false` 的锁文件。

- [ ] **Step 4: 运行训练器烟雾测试**

Run:

```powershell
D:\Anaconda\python.exe -m pytest 20_curriculum_multiband\tests\test_training_smoke.py -q
D:\Anaconda\python.exe 20_curriculum_multiband\code\03_train_curriculum.py --epochs 3 --smoke
```

Expected: 三轮完成；阶段和域来源记录正确；不读取SEG-Y宽频标签。

- [ ] **Step 5: 提交**

```powershell
git add 20_curriculum_multiband/code/03_train_curriculum.py 20_curriculum_multiband/tests/test_training_smoke.py
git commit -m "feat: implement 300-epoch curriculum trainer"
```

## Task 7: 实现锁定推理、direct/highpass双结果和盲评

**Files:**
- Create: `20_curriculum_multiband/code/04_predict_locked_f3.py`
- Create: `20_curriculum_multiband/code/05_blind_evaluate.py`
- Create: `20_curriculum_multiband/tests/test_inference_lock.py`

- [ ] **Step 1: 编写锁验证和双结果测试**

```python
def test_prediction_refuses_unlocked_checkpoint(tmp_path):
    checkpoint = tmp_path / "best_model.pth"
    checkpoint.write_bytes(b"model")
    with pytest.raises((FileNotFoundError, ValueError)):
        validate_before_reference_read(checkpoint, tmp_path / "model_lock.json")


def test_direct_and_highpass_keep_narrow_body():
    direct, highpass = recombine(narrow, residual, dt=0.004)
    assert np.allclose(direct, narrow + residual)
    assert low_band_nrmse(highpass, narrow, high_hz=22) < 1e-5
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```powershell
D:\Anaconda\python.exe -m pytest 20_curriculum_multiband\tests\test_inference_lock.py -q
```

Expected: FAIL。

- [ ] **Step 3: 实现锁定推理**

从19号推理器复用正确几何和滑窗融合，并增加：

```text
direct_prediction = narrow_raw + projected_residual
highpass_prediction = narrow_raw + highpass(projected_residual, 35 Hz)
```

必须在 `read_segy()` 之前校验模型锁。保存窄频输入、残差、direct、highpass和参考；
metadata记录锁SHA256及 `reference_read_after_lock=true`。

- [ ] **Step 4: 实现盲评**

评价器同时比较：

- low-pass baseline；
- direct；
- highpass residual recombination；
- F3 wide reference。

剖面前三类使用共同P99色标。报告inline、crossline总指标和每个剖面指标，并明确：
频谱改善不能替代相关性、相位和误差改善。

- [ ] **Step 5: 运行测试**

Run:

```powershell
D:\Anaconda\python.exe -m pytest 20_curriculum_multiband\tests\test_inference_lock.py -q
```

Expected: PASS。

- [ ] **Step 6: 提交**

```powershell
git add 20_curriculum_multiband/code/04_predict_locked_f3.py 20_curriculum_multiband/code/05_blind_evaluate.py 20_curriculum_multiband/tests/test_inference_lock.py
git commit -m "feat: add locked dual-output blind evaluation"
```

## Task 8: 生成完整样本并执行300轮训练

**Files:**
- Generate: `20_curriculum_multiband/data/F3多频带自监督/*`
- Generate: `20_curriculum_multiband/data/测井合成样本/*`
- Generate: `20_curriculum_multiband/data/模型检查点/*`
- Generate: `20_curriculum_multiband/logs/train_full.log`

- [ ] **Step 1: 运行全部静态测试**

```powershell
D:\Anaconda\python.exe -m pytest 20_curriculum_multiband\tests -q
$files = Get-ChildItem 20_curriculum_multiband\code -Filter *.py | ForEach-Object FullName
D:\Anaconda\python.exe -m py_compile $files
```

Expected: 全部PASS且编译成功。

- [ ] **Step 2: 生成完整F3基础patch**

```powershell
D:\Anaconda\python.exe 20_curriculum_multiband\code\01_prepare_f3_patches.py --max-train 4000 --max-val 600
```

检查：

```text
shape = (N, 256, 256)
normalization = per_patch_p99_abs_clean_narrow
uses_f3_wide_target = false
train/val plane groups disjoint
```

- [ ] **Step 3: 生成完整测井合成样本**

```powershell
D:\Anaconda\python.exe 20_curriculum_multiband\code\02_generate_well_synthetic.py --target-patches 2600
```

检查闭合误差、子波来源比例、噪声水平、井组合和section分组。

- [ ] **Step 4: 启动300轮训练**

```powershell
D:\Anaconda\python.exe 20_curriculum_multiband\code\03_train_curriculum.py --epochs 300 --batch-size 4 *> 20_curriculum_multiband\logs\train_full.log
```

训练过程中每30分钟检查进程、stderr、F3验证相关性、相位和合成验证相关性。不得根据
F3宽频结果中途调参。

- [ ] **Step 5: 检查门控结果**

通过门槛时：

```powershell
D:\Anaconda\python.exe -c "from pathlib import Path; import sys; sys.path.insert(0,'20_curriculum_multiband/code'); from leakage_guard import verify_model_lock; from config import CHECKPOINT_DIR; print(verify_model_lock(CHECKPOINT_DIR/'best_model.pth', CHECKPOINT_DIR/'model_lock.json'))"
```

若未通过门槛，停止，不读取F3宽频参考，并根据 `training_failed_gate.json` 写训练失败分析。

## Task 9: 完成一次性四井盲评与结果归档

**Files:**
- Generate: `20_curriculum_multiband/figures/预测评价/*`
- Create: `20_curriculum_multiband/figures/预测评价/curriculum20_final_analysis.txt`
- Modify: `20_curriculum_multiband/README.md`
- Create: `docs/21_20号实验改进思路.txt`

- [ ] **Step 1: 模型锁通过后运行inline与crossline预测**

```powershell
D:\Anaconda\python.exe 20_curriculum_multiband\code\04_predict_locked_f3.py --section-axis inline --values 244,362,442,722 --output-prefix curriculum20_inline
D:\Anaconda\python.exe 20_curriculum_multiband\code\04_predict_locked_f3.py --section-axis crossline --values 336,387,848,1007 --output-prefix curriculum20_crossline
```

- [ ] **Step 2: 运行双结果盲评**

```powershell
D:\Anaconda\python.exe 20_curriculum_multiband\code\05_blind_evaluate.py --prefix curriculum20_inline
D:\Anaconda\python.exe 20_curriculum_multiband\code\05_blind_evaluate.py --prefix curriculum20_crossline
```

- [ ] **Step 3: 编写最终分析**

分析文件必须列出：

- 数据边界和锁SHA256；
- F3遮蔽验证相关性、相位、泄漏；
- 合成验证残差相关性；
- direct和highpass的inline/crossline指标；
- 与低通基线、16号和19号对比；
- 是否满足全频相关性不下降、残差相关性明显为正、相位稳定为正；
- 若失败，明确指出失败来自已知频带迁移、合成外推或重组策略中的哪一环。

- [ ] **Step 4: 最终验证**

```powershell
D:\Anaconda\python.exe -m pytest 20_curriculum_multiband\tests -q
git diff --check
git status --short
```

人工查看inline/crossline剖面和频谱图，确认使用共同色标、无剖面拼接错误、图题归属20号。

- [ ] **Step 5: 提交代码和小型报告**

```powershell
git add 20_curriculum_multiband/README.md 20_curriculum_multiband/code 20_curriculum_multiband/tests docs/21_20号实验改进思路.txt
git add -f 20_curriculum_multiband/figures/预测评价/*evaluation_report.txt 20_curriculum_multiband/figures/预测评价/curriculum20_final_analysis.txt
git commit -m "feat: complete curriculum experiment 20 evaluation"
```
