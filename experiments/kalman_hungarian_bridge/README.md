# Kalman + Hungarian 桥接实验

本实验只验证 Kalman predicted bbox 是否比上一帧 last bbox 更适合作为 Hungarian
关联输入，不实现 SORT 或轨迹生命周期。

两条路线使用完全相同的 detections 和 IoU gating 阈值：

- 方案 A：last bbox → IoU matrix → `cost = 1 - IoU` → Hungarian → IoU gating；
- 方案 B：Kalman predicted bbox → IoU matrix → `cost = 1 - IoU` → Hungarian → IoU gating。

Kalman 状态保持为 `[x, y, vx, vy]`。它只预测 bbox 中心，宽高沿用 last bbox，
再将预测中心转换回 xyxy。每个合成场景固定 2 条预先存在的轨迹，不创建或删除轨迹。
漏检一帧和三帧场景分别在当前检测到达前累计执行 2 次和 4 次 `predict`。

## 运行

```powershell
python -m experiments.kalman_hungarian_bridge.run_experiment
python -m unittest discover -s experiments/kalman_hungarian_bridge/tests -v
```

实验会在终端输出两路矩阵、Hungarian 原始匹配、gating 结果及未匹配集合，并将逐对
比较保存到 `results/association_comparison.csv`。
