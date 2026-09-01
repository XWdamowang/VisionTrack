# 多源视觉目标识别与运动状态分析系统

## 项目目标

本项目面向多源视觉场景，计划逐步实现目标检测、目标跟踪、运动状态估计与多源信息融合。正式工程当前提供单视频 YOLO 目标检测基线；IoU Tracker 和二维匀速卡尔曼滤波位于独立学习实验区。

## 当前已完成功能

- 从固定位置读取单个测试视频；
- 使用 Ultralytics YOLO11n 预训练模型逐帧检测目标；
- 自动选择 CUDA GPU 或 CPU；
- 使用 `conf=0.35`、`iou=0.50` 作为基线参数；
- 将带类别、置信度和检测框的结果保存为视频；
- 将每个检测框的帧号、类别、置信度和坐标导出为 CSV；
- 检测配置和视频处理逻辑封装在 `src/detection/` 模块中；
- 输出处理帧数和结果路径，并为输入缺失等情况提供中文提示；
- 在实验区通过逐帧 IoU 贪心关联生成轨迹 ID，可设置关联阈值和最大丢失帧数；
- 在实验区提供遮挡、交叉和高速运动的合成演示，不需要下载额外视频。

正式工程当前不包含目标跟踪、卡尔曼滤波、外观特征、多源融合、模型训练或前端界面。

此外提供独立的二维匀速卡尔曼滤波学习实验。该实验读取 IoU Tracker 的轨迹 CSV，
不参与多目标关联，也不构成完整 SORT。

匈牙利算法学习实验使用 SciPy 的 `linear_sum_assignment` 完成矩形代价矩阵的全局
最优一对一分配，因此项目依赖中增加了 `scipy`。

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

可通过命令行设置输入、检测阈值和输出路径：

```bash
python main.py --input data/videos/test.mp4 --conf 0.35 --iou 0.50
python main.py --conf 0.20 --output-video outputs/videos/custom.mp4 --output-csv outputs/csv/custom.csv
```

执行 `python main.py --help` 可查看全部参数。未指定输出路径时，程序会将参数写入
文件名，例如 `test_detected_conf_0.35_iou_0.50.mp4`，避免不同实验互相覆盖。

运行真实视频的 IoU 跟踪实验：

```bash
python -m experiments.iou_tracker.run_video
python -m experiments.iou_tracker.run_video --track-iou 0.30 --max-missed 2
```

生成遮挡、交叉和高速运动的可复现学习演示：

```bash
python -m experiments.iou_tracker.run_scenarios
```

演示结果保存为 `outputs/videos/iou_tracker_scenarios.mp4`，逐帧的真实对象标签和
跟踪 ID 保存为 `outputs/csv/iou_tracker_scenarios.csv`。短时遮挡通常能依靠轨迹保留
维持 ID；目标交叉会产生关联歧义；高速运动在相邻帧完全不重叠时会不断创建新 ID。

运行独立卡尔曼滤波实验：

```bash
python -m experiments.kalman_cv_2d.run_experiment
```

程序会自动选择最长连续轨迹，通过视频 FPS 计算 `dt`，在轨迹中段模拟连续 3 帧
漏检，并生成 `outputs/tracking/kalman_test.csv`、轨迹对比图和实验记录。

运行匈牙利算法与 IoU 代价矩阵实验：

```bash
python -m experiments.hungarian_assignment.run_experiment
```

## 输出位置

检测结果视频保存到：

```text
outputs/videos/test_detected_conf_0.35_iou_0.50.mp4
```

逐帧检测结果保存到：

```text
outputs/csv/test_detections_conf_0.35_iou_0.50.csv
```

CSV 每个检测框占一行，字段为 `frame_number`、`class_id`、`class_name`、
`confidence`、`x1`、`y1`、`x2`、`y2`。帧号从 1 开始，框坐标采用像素单位的
左上角 `(x1, y1)` 和右下角 `(x2, y2)`。

## 代码结构

- `main.py`：解析命令行参数、组织输入输出路径并展示运行状态；
- `src/detection/models.py`：定义检测框和帧级检测结果等共享数据结构；
- `src/detection/video_detector.py`：加载模型、选择设备、逐帧检测并写出视频和 CSV；
- `src/detection/__init__.py`：提供检测配置、运行摘要和检测函数的统一导入入口。
- `experiments/iou_tracker/`：IoU 算法、真实视频入口、场景演示、契约和测试；
- `experiments/kalman_cv_2d/`：二维匀速卡尔曼实现、实验入口、说明和测试；
- `experiments/hungarian_assignment/`：IoU 代价矩阵、Hungarian 分配、门控和测试；
- `track.py`、`demo_iou_tracker.py`、`kalman_experiment.py`：兼容旧命令的转发入口。

实验记录模板位于 `outputs/logs/experiment_01.md`，图像类结果可放入 `outputs/images/`。

## 后续工作约束

请遵守以下规则：

1. 不要把实验逻辑直接堆入 `main.py`；
2. 学习性、可视化、对比实验代码放到 `experiments/` 目录；
3. 可复用算法模块才放入 `src/tracking/`；
4. 不修改已有 YOLO 检测模块的行为；
5. 不修改已有 CSV 格式，除非确有必要并先说明原因；
6. 不删除已有实验结果；
7. 不重构与当前任务无关的代码；
8. 新增依赖时说明原因；
9. 修改前先列出准备修改和新增的文件。

## 后续开发计划

1. 人工核验三组置信度实验，确认后续使用的检测基线；
2. 复核匈牙利分配、IoU gating 和贪心反例；
3. 收到明确指令后再在 `experiments/` 中进入最小 SORT 学习；
4. 根据明确需求分析速度、方向等运动状态；
5. 扩展到多视频或多传感器信息融合；
6. 根据需要开发可视化前端和评估工具。
