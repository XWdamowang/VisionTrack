# 项目交接

更新日期：2026-09-01

## 当前阶段

正式工程已具备单视频 YOLO11 检测基线。轻量 IoU Tracker、IoU 典型场景演示和
二维匀速卡尔曼滤波均属于学习实验，不作为正式工程模块依赖。

## 代码边界

- `src/`：正式工程代码，当前只包含检测及其共享数据结构；
- `experiments/`：学习实验，可以导入 `src/`，但 `src/` 不得导入实验模块；
- `tests/`：正式工程测试；
- `experiments/*/tests/`：各实验自己的测试；
- `outputs/`：生成结果，不提交视频、图片和轨迹 CSV。

根目录的 `track.py`、`demo_iou_tracker.py` 和 `kalman_experiment.py` 仅用于兼容旧命令，
具体实现分别位于 `experiments/iou_tracker/` 和 `experiments/kalman_cv_2d/`。

## 已完成

- YOLO11 单视频逐帧检测及检测 CSV 导出；
- 实验性的轻量、类别感知 IoU Tracker 及轨迹 CSV 导出；
- 遮挡、交叉、高速运动的合成 IoU 演示；
- 状态 `[x, y, vx, vy]`、观测 `[x, y]` 的二维匀速卡尔曼学习实验；
- 卡尔曼预测、更新、短暂漏检模拟、结果 CSV、对比图和实验记录；
- IoU 代价矩阵、Hungarian 全局分配、IoU gating 和七种关联场景；
- 正式检测工程与 IoU、Kalman 学习实验的目录和导入方向隔离。

## 尚未实现

- 最小 SORT 学习实验；
- 正式工程中的 SORT、ByteTrack 或其他高级跟踪器；
- 运动状态分析、多源融合、模型训练和前端界面。

## 常用命令

正式检测：

```powershell
python main.py
```

学习实验：

```powershell
python -m experiments.iou_tracker.run_video
python -m experiments.iou_tracker.run_scenarios
python -m experiments.kalman_cv_2d.run_experiment
```

测试：

```powershell
python -m unittest discover -s tests -v
python -m unittest discover -s experiments/iou_tracker/tests -v
python -m unittest discover -s experiments/kalman_cv_2d/tests -v
```

## 下一步建议

当前先复核 `experiments/hungarian_assignment/` 中的代价矩阵、矩形分配、空输入、
低 IoU 门控和贪心反例。只有收到明确指令后，才进入最小 SORT 学习阶段；届时仍应
放在 `experiments/`，不得直接修改正式检测流程。
