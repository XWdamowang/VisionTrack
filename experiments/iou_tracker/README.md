# IoU Tracker 学习实验

本目录包含仅依赖边界框 IoU 的轻量多目标跟踪实验，不属于正式工程跟踪模块。

内容包括：

- `iou_tracker.py`：IoU 计算、贪心一对一匹配和简单轨迹生命周期；
- `video_tracker.py`：消费正式检测接口并导出带 ID 的视频和轨迹 CSV；
- `run_video.py`：真实视频实验入口；
- `run_scenarios.py`：遮挡、交叉和高速运动的合成场景；
- `tests/`：算法与输出契约测试；
- `NOTES.md`：输入输出契约、参数和验收记录。

运行真实视频实验：

```powershell
python -m experiments.iou_tracker.run_video
```

运行合成场景：

```powershell
python -m experiments.iou_tracker.run_scenarios
```

运行测试：

```powershell
python -m unittest discover -s experiments/iou_tracker/tests -v
```

根目录的 `track.py` 和 `demo_iou_tracker.py` 是兼容旧命令的转发入口。
