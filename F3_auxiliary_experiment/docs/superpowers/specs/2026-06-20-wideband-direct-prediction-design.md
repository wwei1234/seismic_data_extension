# 17号宽频直接预测实验设计

## 目标

创建 `17_wideband_direct_prediction` 实验，在保持 16 号实验样本几何、合成流程、噪声设置、归一化和网络主体不变的前提下，将训练标签从高频残差改为完整宽频数据。模型直接输出宽频剖面，训练 200 轮，并在 F3 实际数据上进行拓频预测和评价。

## 实验边界

- 复用根目录 `Rawdata/`、`shared_data/` 和 `shared_code/`，不复制共享数据和代码。
- 17 号文件夹只保存本实验的代码、样本、模型、日志、预测结果、评价结果和图件。
- 保持 `UNetCBAM` 单输入、单输出结构，不修改主干网络。
- 不使用 16 号模型继续训练。标签和输出含义发生变化，必须从随机初始化重新训练。
- 训练轮数固定为 200。

## 样本生成

样本的地质结构、井位投影和增强规则与 16 号保持一致：

- 使用 2 井、3 井、4 井共 11 种组合。
- 使用 3 个 B-spline 宽频子波。
- 使用噪声水平 `0.01` 和 `0.03`。
- 共生成 `11 x 3 x 2 = 66` 个合成剖面。
- 剖面宽度为 951 道，对应 crossline 300–1250。
- 使用井的实际 crossline 位置和线性横向插值。
- 使用线性时间插值、size=5 横向平滑和 16 号的构造扰动。
- `PATCH_SIZE=256`，`PATCH_STRIDE=128`。
- 仅允许左右翻转，不允许时间轴上下翻转。

每个剖面的数据闭合关系为：

```text
clean_wide = reflectivity * wide_wavelet
clean_narrow = lowpass(clean_wide)
shared_scale = P99(abs(clean_narrow))

wide_label = clip(clean_wide / shared_scale, -1, 1)
clean_input = clip(clean_narrow / shared_scale, -1, 1)
model_input = clip(clean_input + noise, -1, 1)
```

`train_labels.npy` 和 `val_labels.npy` 保存 `wide_label`，标签不含噪声。元数据明确记录：

```text
label_type = wide_band
normalization = per_section_p99_abs_clean_narrow
```

## 模型输出

模型直接预测归一化宽频数据：

```text
wide_pred_norm = model(narrow_input_norm)
```

不再使用 16 号的以下残差专用逻辑：

- 零均值输出约束。
- `wide_pred = input + residual_pred`。
- 残差均值损失。
- 残差能量不足惩罚。
- 推理阶段的残差高通重组。
- `R_MEDIAN` 或全局宽窄频振幅比。

实际数据推理时：

```text
narrow_scale = P99(abs(narrow_raw))
narrow_norm = clip(narrow_raw / narrow_scale, -1, 1)
wide_pred_norm = model(narrow_norm)
wide_pred = wide_pred_norm * narrow_scale
```

## 损失函数

总损失直接围绕宽频标签设计：

```text
L_total =
    1.0 * L_waveform
  + 0.5 * L_spectrum
  + 0.2 * L_phase
  + 0.2 * L_time_gradient
  + 0.5 * L_low_frequency
```

各项含义：

1. `L_waveform`
   - 预测宽频与标签之间的 L1 损失。
   - 负责整体振幅、波形和空间结构。

2. `L_spectrum`
   - 比较预测与标签在 25–80 Hz 范围内的 FFT 振幅。
   - 使用归一化频谱，降低绝对振幅对频谱形状损失的支配。

3. `L_phase`
   - 比较 25–80 Hz 范围内的复频谱相位。
   - 使用 `1-cos(phase_pred-phase_target)`，约束新增高频的相位位置。

4. `L_time_gradient`
   - 比较时间方向一阶差分。
   - 提高薄层边界和同相轴细节，同时避免仅增加高频噪声。

5. `L_low_frequency`
   - 对预测和标签分别做 3–35 Hz 零相位滤波，再计算 L1。
   - 防止直接预测模式破坏低频主体。

模型选择依据为验证集 `L_total` 最低值。保存 `best_model.pth` 和 `last_model.pth`。

## 训练配置

```text
epochs = 200
batch_size = 4
optimizer = Adam
learning_rate = 1e-3
ReduceLROnPlateau patience = 10
random_seed = 42
```

不因验证集暂时不改善而提前终止，确保完成 200 轮。训练日志记录总损失及五个分量，并生成训练曲线。

## 推理与评价

训练完成后首先评价四口井所在 inline：

```text
244, 362, 442, 722
```

同时评价四口井所在 crossline：

```text
336, 387, 848, 1007
```

每个方向输出：

- 低通输入。
- 模型直接预测宽频结果。
- 实际 F3 宽频参考。
- 预测误差剖面。
- 共用色标的逐剖面对比图。
- 平均振幅频谱对比图。
- 每条剖面及整体评价报告。

评价指标：

```text
MAE
RMSE
PSNR
Correlation
35–80 Hz energy ratio
Spectrum L1 distance
```

评价时同时报告低通输入基线和模型预测结果。只有频谱改善而空间指标下降时，必须明确说明新增高频的相位或振幅匹配仍不可靠。

## 验证要求

训练前验证：

- 66 个剖面数量正确。
- stride 为 128。
- 标签等于归一化宽频数据，而不是高频残差。
- 标签不含输入噪声。
- 归一化使用低通数据 P99。
- 不存在时间轴翻转。
- 模型输出不做零均值处理。

训练后验证：

- 训练历史包含完整 200 轮。
- 最佳和最后模型均可加载。
- inline 与 crossline 预测数组 shape、编号和 metadata 正确。
- 预测图、频谱图、指标文件和最终中文分析报告均存在。
