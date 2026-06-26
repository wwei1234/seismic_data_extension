# 27 Self-Supervised IDR Spectral Extrapolation

文献来源：A Self-Supervised Learning Framework for Seismic Low-Frequency Extrapolation。

算法实现：该文献面向低频外推。这里将其warm-up/iterative data refinement思想改造成无标签频带外推基线：从F3低通数据构造更窄频输入，迭代生成伪宽频标签并更新谱补偿算子。

运行入口：`code/run_experiment.py`
