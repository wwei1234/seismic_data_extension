# 15_bspline_wavelet_bank

15 号实验目录保留 B-spline 子波库方案的代码、结果和日志。

统一资源已经移到 `F3_auxiliary_experiment` 根目录：

- `../Rawdata/`：原始 F3 SEG-Y、测井和辅助数据。
- `../docs/`：历史方案文档和技术记录。

本目录结构：

- `code/`：15 号实验代码。
- `data/`：15 号实验生成的数据和检查点。
- `figures/`：15 号实验生成的图件。
- `logs/`：15 号实验日志。

代码中的 SEG-Y 路径已改为读取根目录 `../Rawdata/Seismic_data.sgy`，避免每个方案目录重复保存原始数据。

