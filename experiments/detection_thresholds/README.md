# YOLO 检测置信度阈值对比实验

该实验对同一输入视频依次使用多组 `conf` 参数运行正式单视频检测入口，所有组共用
同一个 `iou` 参数。检测实现仍位于 `src/detection/`，本目录只负责组织实验参数，
不复制检测或视频编码逻辑。

默认运行 `conf=0.20`、`0.35`、`0.60` 和 `iou=0.50`：

```bash
python -m experiments.detection_thresholds.run_experiment
```

自定义输入、IoU 和置信度组：

```bash
python -m experiments.detection_thresholds.run_experiment \
  --input data/videos/test.mp4 \
  --iou 0.50 \
  --confidences 0.20 0.35 0.60
```

每组结果沿用正式入口的命名规则，分别写入 `outputs/videos/` 和 `outputs/csv/`。
已有实验记录位于 `outputs/logs/experiment_01.md`。
