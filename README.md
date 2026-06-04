# seismic_data_extension

地震数据拓频相关代码，包括数据集处理、网络结构、损失函数、训练、预测和绘图脚本。

## Files

- `dataset.py`: 数据集加载与预处理
- `U_Net_CBAM.py`: U-Net/CBAM 网络结构
- `loss_function.py`: 训练损失函数
- `train.py`: 模型训练入口
- `predict.py`: 模型预测入口
- `project_115/train.py`: 训练脚本备份或变体
- `*.py`: 数据生成、剖面绘制、频谱分析等辅助脚本

## Notes

本仓库只跟踪代码和小型说明材料。训练数据、模型权重、预测结果、日志和 Marmousi 数据等大文件未纳入 Git 跟踪，需要在本地按脚本路径准备。
