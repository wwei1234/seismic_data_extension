# 20_curriculum_multiband

本实验使用F3窄频动态多频带遮蔽和测井约束合成残差进行联合课程训练。

F3实际宽频数据禁止参与训练、验证、调参和checkpoint选择。只有模型通过F3已知
频带门控并生成哈希锁后，才允许读取F3宽频参考进行一次性盲评。

设计文档：
`../docs/superpowers/specs/2026-06-23-multiband-curriculum-design.md`

实施计划：
`../docs/superpowers/plans/2026-06-23-multiband-curriculum-implementation.md`

## 实验状态

已完成：

- F3窄频基础patch：训练4000、验证600；
- 测井约束合成patch：训练2075、验证525；
- 300轮联合课程训练；
- 29项单元和集成测试。

训练未通过预先锁定的checkpoint门控，因此未生成模型锁，也未读取F3宽频参考进行
盲评。

最佳训练指标：

- F3遮蔽相关性：0.787289，第175轮；
- F3遮蔽相位分数：0.789738，第169轮；
- 合成残差相关性：0.752786，第281轮；
- 最终F3遮蔽相关性/相位：0.775065/0.778485。

详细分析：
`figures/预测评价/curriculum20_training_gate_failure_analysis.txt`
