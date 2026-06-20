# 16_geometry_realistic_samples

16 号实验目录只保存本轮方案相关的实验脚本、生成样本、预测结果、图件和日志。

根目录统一资源：

- `../Rawdata/`：原始 F3 SEG-Y 和辅助原始数据。
- `../shared_data/`：各轮实验重复使用的井数据、井旁道匹配结果和估计子波。
- `../shared_code/`：各轮实验重复使用的公共代码。
- `../docs/`：历史方案文档和技术记录。

本目录结构：

- `code/`：16 号实验专属脚本，不再包含 `model.py`、`segy_reader.py`、`signal_utils.py` 和 `01_rematch_wells_estimate_wavelets.py`。
- `data/`：本轮生成的数据。
  - `样本数据/`
  - `模型检查点/`
  - `预测结果/`
  - `评价结果/`
- `figures/`：本轮生成的图件。
  - `合成剖面/`
  - `训练样本/`
  - `频谱分析/`
  - `预测评价/`
- `logs/`：训练和运行日志。

新实验目录沿用这个结构即可，不要再复制根目录 `shared_data` 和 `shared_code` 中的文件。

