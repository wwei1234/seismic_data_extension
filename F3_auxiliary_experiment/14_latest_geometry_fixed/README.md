# 14_latest_geometry_fixed

本目录是 F3 辅助实验当前保留的最新整理版本。旧的 01-13 实验目录和根目录零散输出已清理，当前只保留最新代码、基础输入数据、原始 SEG-Y 数据和技术记录。

## 目录结构

- `code/`: 最新可运行代码，基于 `13_kriging` 整理，并修复了 F3 inline 不等长导致的剖面拼接问题。
- `data/`: 基础井数据、子波和井旁道缓存，用于重新生成训练样本。
- `Rawdata/`: 原始 F3 SEG-Y 数据。
- `figures/`: 重新运行后保存图件。
- `logs/`: 重新训练后保存日志和模型权重。
- `docs/`: 历史技术记录和方案说明。

## 关键修复

原始读取方式默认所有 inline 的 crossline 数量一致，会在 F3 工区这种 inline 长度不一致的数据上把相邻 inline 的部分道拼接到同一个剖面中。

当前 `code/segy_reader.py` 已改为按 SEG-Y header 中的 inline/crossline 建立规则网格：

- inline 范围：100-750，共 651 条。
- crossline 范围：300-1250，共 951 条。
- 每条 inline 的实际道数不固定，约 570-951。
- 缺失位置用 `0.0` 填充，输出 cube 尺寸为 `(651, 462, 951)`。

因此旧目录中基于错误读取方式生成的训练集、预测结果和图件没有继续保留，需要用当前目录重新生成。

## 建议重跑顺序

在本目录下执行：

```powershell
D:\Anaconda\python.exe code\02_generate_synthetic_dataset.py
D:\Anaconda\python.exe code\01_train.py
D:\Anaconda\python.exe code\02_predict_f3.py --output-prefix kriging_v1
```

之后按需要运行 `code/03_evaluate.py` 和 `code/04-10_*.py` 生成频谱、滤波和井 inline 对比图。
