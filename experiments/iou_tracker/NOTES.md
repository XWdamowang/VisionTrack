# 轻量 IoU 多目标跟踪实验记录

## 当前边界

当前已实现基于边界框 IoU 的贪心目标关联和简单轨迹生命周期管理，不包含
卡尔曼滤波、外观特征、运动状态分析或多源融合。

## 检测输入契约

跟踪模块应消费 `src.detection.FrameDetections`：

- `frame_number`：从 1 开始的帧号；
- `timestamp_seconds`：根据原视频 FPS 计算的时间戳；
- `source_fps`：原视频帧率；
- `frame`：未经绘制的原始 BGR 帧；
- `annotated_frame`：仅包含检测可视化的 BGR 帧；
- `detections`：本帧全部 `Detection` 对象。

每个 `Detection` 包含类别 ID、类别名称、置信度和 `xyxy` 像素坐标。跟踪实现应使用
这些稳定字段，不应直接依赖 Ultralytics `Results` 或 `Boxes` 对象。

逐帧入口为：

```python
from contextlib import closing
from src.detection import DetectionConfig, VideoDetector, iter_video_detections

detector = VideoDetector(DetectionConfig())
with closing(iter_video_detections(input_path, detector)) as frames:
    for frame_result in frames:
        # 后续在这里调用跟踪器。
        pass
```

迭代器正常结束或提前关闭时都会释放视频读取资源。

## 跟踪策略

- 暂定检测基线：`conf=0.35`、`iou=0.50`；
- 默认关联阈值为 `0.30`，默认保留丢失轨迹 2 帧；
- 只关联相同检测类别，并按 IoU 从高到低进行一对一匹配；
- 首轮保留全部检测类别，不预先缩小目标范围；
- 跟踪视频与轨迹 CSV 单独保存，不覆盖检测阶段结果；
- 跟踪算法封装在 `experiments/iou_tracker/` 中，只依赖正式检测接口。

真实视频入口为 `python -m experiments.iou_tracker.run_video`。无需模型推理的
典型场景演示入口为 `python -m experiments.iou_tracker.run_scenarios`。

## 跟踪输出契约

轨迹 CSV 保存到 `outputs/csv/test_tracks_*.csv`，每行表示一帧中的一个已跟踪目标，
字段顺序由 `experiments.iou_tracker.TRACK_CSV_FIELDS` 统一定义：

```text
frame_number,timestamp_seconds,track_id,class_id,class_name,
confidence,x1,y1,x2,y2
```

跟踪视频保存到 `outputs/videos/test_tracked_*.mp4`。

## 验收标准

1. 跟踪视频帧数、分辨率和 FPS 与输入视频一致；
2. 每条轨迹记录都包含合法的帧号、非负整数 `track_id` 和有效边界框；
3. 同一目标在连续帧中尽量保持相同 ID；
4. 不同目标不应长期共享同一 ID；
5. 单条轨迹的类别不应无故变化；
6. 目标交叉、短时遮挡、进入和离开画面的片段需要人工抽查；
7. 跟踪视频首帧和末帧均可读取；
8. 轨迹 CSV 与视频中的帧号和 ID 能够对应。

没有人工标注轨迹时，只能验证结构、连续性和典型片段，不能声称获得定量跟踪精度。
