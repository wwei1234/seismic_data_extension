# 23 Spectral Shaping SRI

文献来源：Seismic Resolution Enhancement by Spectral Shaping Using Shaping-Regularized Inversion。

算法实现：用F3低通数据的平均振幅谱构造对角正演核，目标谱设为宽频梯形谱，通过平滑约束的谱整形算子近似shaping-regularized inversion。低频0-25Hz由输入低通数据保护。

运行入口：`code/run_experiment.py`
