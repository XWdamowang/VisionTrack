# 匈牙利算法与 IoU 代价矩阵实验

本实验只学习预测轨迹框与当前检测框之间的一对一关联，不实现 SORT 生命周期、
多 Kalman Track 管理、ByteTrack 或 DeepSORT。

## 核心流程

1. 计算形状为 `Tracks × Detections` 的 IoU 矩阵；
2. 使用 `cost = 1 - IoU` 把“越大越好”的相似度转换为“越小越好”的代价；
3. 将代价矩阵传给 `scipy.optimize.linear_sum_assignment`；
4. 对原始分配执行 IoU gating，默认阈值为 `0.30`；
5. 低于阈值的配对被拒绝，双方分别进入未匹配集合。

`linear_sum_assignment` 输入二维代价矩阵，输出两个等长索引数组：第一个是被选中的
行索引（track），第二个是对应列索引（detection）。组合相同位置的两个索引即可得到
一对一匹配。

## SciPy 依赖

项目增加 `scipy`，原因是本阶段明确使用其经过验证的 `linear_sum_assignment` 实现
矩形线性分配问题，避免把学习重点转移到手写求解器的边界细节。

## 运行

```powershell
python -m experiments.hungarian_assignment.run_experiment
python -m unittest discover -s experiments/hungarian_assignment/tests -v
```
