# 17_wideband_direct_prediction

本实验沿用 16 号的合成几何、宽频子波、噪声和切片方式，将监督标签改为无噪声宽频数据。

- 输入：加噪低通数据。
- 标签：无噪声宽频数据。
- 输出：模型直接预测的宽频数据。
- 训练：200 轮。
- 共享数据：`../shared_data/`。
- 共享代码：`../shared_code/`。
- 原始地震：`../Rawdata/Seismic_data.sgy`。

本目录不复制共享井数据、SEG-Y 读取器、信号工具或模型定义。

## 本轮结果

- 完成 200 轮训练，最佳模型位于第 197 轮。
- 四口井 inline 和 crossline 均完成预测评价。
- 预测结果的频谱距离明显改善，但空间域 MAE、RMSE 和相关性劣于低通基线。
- 详细结论见 `figures/预测评价/wide17_200_final_analysis.txt`。
