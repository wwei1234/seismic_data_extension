# 20_curriculum_multiband

本实验使用F3窄频动态多频带遮蔽和测井约束合成残差进行联合课程训练。

F3实际宽频数据禁止参与训练、验证、调参和checkpoint选择。只有模型通过F3已知
频带门控并生成哈希锁后，才允许读取F3宽频参考进行一次性盲评。

设计文档：
`../docs/superpowers/specs/2026-06-23-multiband-curriculum-design.md`

实施计划：
`../docs/superpowers/plans/2026-06-23-multiband-curriculum-implementation.md`
