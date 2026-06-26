# 24 Low-Frequency Protection And Spectral Extrapolation

文献来源：地震数据低频信号保护与拓频方法研究。

算法实现：保留低频主体，利用低通有效频带的对数谱趋势外推高频目标谱，形成低频保护谱平衡基线。该实现对应文献中“保护有效低频，再进行拓频/重构”的处理思想。

运行入口：`code/run_experiment.py`
