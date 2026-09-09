# 学习实验

本目录保存用于理解算法原理、观察边界条件和记录实验结果的代码。

依赖方向固定为：实验代码可以导入 `src/` 中的正式工程模块，`src/` 不得导入
`experiments/`。实验成熟后，需要重新确认接口、测试和错误处理，再决定是否将相关
能力迁入正式工程，不能直接把实验模块作为生产依赖。

## 当前实验

- `detection_thresholds/`：同一视频在多组 YOLO 置信度阈值下的检测结果对比；
- `iou_tracker/`：IoU Tracker、真实视频入口、典型场景演示及其测试；
- `kalman_cv_2d/`：状态为 `[x, y, vx, vy]` 的二维匀速卡尔曼滤波实验。
- `hungarian_assignment/`：IoU 代价矩阵、Hungarian 全局分配和 IoU gating 实验。
- `kalman_hungarian_bridge/`：last bbox 与 Kalman predicted bbox 的双路 Hungarian 关联对照。
