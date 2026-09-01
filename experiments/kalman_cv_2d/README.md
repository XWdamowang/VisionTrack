# 二维匀速卡尔曼滤波实验

该实验从 IoU Tracker 的轨迹 CSV 中选择最长的完整连续轨迹，使用视频 FPS 计算
`dt`，分别输出观测位置、更新前预测位置和更新后滤波位置。实验会在轨迹中段人为
隐藏 1–3 帧观测，用于观察只执行预测时的表现。

状态为 `[x, y, vx, vy]`，观测为 `[x, y]`。这里的实现用于学习原理，不是完整
SORT，也不是正式工程跟踪器。

运行：

```powershell
python -m experiments.kalman_cv_2d.run_experiment
```

实验测试：

```powershell
python -m unittest discover -s experiments/kalman_cv_2d/tests -v
```

为兼容原有使用方式，项目根目录的 `kalman_experiment.py` 仍可执行，但只负责转发。
