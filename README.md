# 多源视觉目标识别与运动状态分析系统

## 项目目标

本项目面向多源视觉场景，计划逐步实现目标检测、目标跟踪、运动状态估计与多源信息融合。本阶段建立一个可复现的单视频 YOLO 目标检测基线，为后续模块提供输入和实验依据。

## 当前已完成功能

- 从固定位置读取单个测试视频；
- 使用 Ultralytics YOLO11n 预训练模型逐帧检测目标；
- 自动选择 CUDA GPU 或 CPU；
- 使用 `conf=0.35`、`iou=0.50` 作为基线参数；
- 将带类别、置信度和检测框的结果保存为视频；
- 将每个检测框的帧号、类别、置信度和坐标导出为 CSV；
- 输出处理帧数和结果路径，并为输入缺失等情况提供中文提示。

本阶段不包含目标跟踪、卡尔曼滤波、多源融合、模型训练或前端界面。

## 环境安装

建议使用 Python 3.10 或更高版本，并在虚拟环境中安装依赖：

```bash
python -m venv .venv
```

Windows PowerShell 激活虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如需使用 NVIDIA GPU，请确保显卡驱动及 PyTorch CUDA 环境可用；否则程序会自动使用 CPU。首次运行时，Ultralytics 会按需获取 `yolo11n.pt` 预训练权重。

## 准备测试视频

将测试视频命名为 `test.mp4`，放置到：

```text
data/videos/test.mp4
```

项目不会自动下载测试视频。

## 运行方法

在项目根目录执行：

```bash
python main.py
```

## 输出位置

检测结果视频保存到：

```text
outputs/videos/test_detected.mp4
```

逐帧检测结果保存到：

```text
outputs/csv/test_detections.csv
```

CSV 每个检测框占一行，字段为 `frame_number`、`class_id`、`class_name`、
`confidence`、`x1`、`y1`、`x2`、`y2`。帧号从 1 开始，框坐标采用像素单位的
左上角 `(x1, y1)` 和右下角 `(x2, y2)`。

实验记录模板位于 `outputs/logs/experiment_01.md`，图像类结果可放入 `outputs/images/`。

## 后续开发计划

1. 在不同置信度阈值下评估检测效果并记录实验结果；
2. 封装检测模块和配置加载逻辑；
3. 增加多目标跟踪及轨迹管理；
4. 引入卡尔曼滤波并分析速度、方向等运动状态；
5. 扩展到多视频或多传感器信息融合；
6. 根据需要开发可视化前端和评估工具。
