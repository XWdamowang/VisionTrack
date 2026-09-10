# 简化 SORT 实验

该实验把项目已有组件串成一个可运行的标准 SORT 核心：

1. 使用 `[cx, cy, area, aspect_ratio, vx, vy, v_area]` 七维 Kalman 状态；
2. 同时预测检测框中心与面积，宽高比在状态中按常量建模；
3. 使用现有 IoU 代价矩阵、SciPy Hungarian 和 IoU gating 完成一对一关联；
4. 用匹配检测的 `[cx, cy, area, aspect_ratio]` 修正 Kalman；
5. 为未匹配检测创建单调递增的 Track ID；
6. 维护 `age`、`hits`、`hit_streak` 和 `time_since_update`；
7. `hits >= min_hits` 时标记为已确认，`time_since_update > max_age` 时删除。

`update()` 为兼容现有实验返回全部活动轨迹；可通过结果的 `is_confirmed` 或跟踪器的
`confirmed_track_ids` 获取达到 `min_hits` 的轨迹。旧参数 `max_missed_frames` 仍可使用，
显式设置时优先于 `max_age`。

运行确定性实验：

```bash
python -m experiments.sort_tracker.run_experiment
```

运行全量测试：

```bash
python -m pytest -q
```

实验输出逐帧展示确认状态、四个生命周期计数、`area`、`v_area`、关联前 predicted
bbox 以及关联后的 filtered bbox，可直接观察尺度预测。

当前版本只使用运动与 IoU，不含类别约束、ReID、DeepSORT 或 ByteTrack。宽高比没有
速度项，密集交叉、形变、高速运动和长时间遮挡仍可能造成 ID 切换。
