# 多目标跟踪接入准备

## 当前边界

本阶段只准备检测与跟踪之间的数据接口、输出契约和验收标准，不实现目标关联、
轨迹管理、卡尔曼滤波、运动状态分析或多源融合。

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

## 首轮跟踪策略

- 暂定检测基线：`conf=0.35`、`iou=0.50`；
- 暂定跟踪器配置：Ultralytics `bytetrack.yaml`；
- 首轮保留全部检测类别，不预先缩小目标范围；
- 跟踪视频与轨迹 CSV 单独保存，不覆盖检测阶段结果；
- ByteTrack 与其他算法的差异应封装在 `src/tracking/` 中，不反向侵入检测模块。

## 跟踪输出契约

轨迹 CSV 保存到 `outputs/csv/test_tracks_*.csv`，每行表示一帧中的一个已跟踪目标，
字段顺序由 `src.tracking.TRACK_CSV_FIELDS` 统一定义：

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
