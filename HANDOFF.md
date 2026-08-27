# 项目交接

更新日期：2026-08-27

## 当前阶段

项目名称为“多源视觉目标识别与运动状态分析系统”。当前处于第一阶段，只实现单视频 YOLO11n 目标检测基线。

## 已完成

- 已建立数据、输出、配置和源码目录骨架。
- `main.py` 从 `data/videos/test.mp4` 读取视频。
- 使用 Ultralytics `yolo11n.pt` 预训练模型逐帧检测。
- 自动选择 CUDA 或 CPU。
- 基线参数为 `conf=0.35`、`iou=0.50`。
- 将带检测框的视频保存到 `outputs/videos/test_detected.mp4`。
- 输入视频不存在时输出中文错误提示。
- 处理成功时输出处理帧数和结果保存位置。
- 已创建依赖清单、项目说明和三组置信度实验记录模板。

## 尚未实现

- 目标跟踪
- 卡尔曼滤波
- 运动状态分析
- 多源融合
- 模型训练
- 前端界面

## 当前环境状态

- 操作系统：Windows。
- Conda 环境 `visual_analysis` 已存在。
- 当前检查终端仍使用 `base` 环境和 Python 3.11.5，运行项目前需要激活 `visual_analysis`。
- `visual_analysis` 内的 Python、PyTorch、CUDA、Ultralytics 和 OpenCV 状态尚未在本次交接中重新验证。
- `data/videos/test.mp4` 当前不存在。
- `outputs/videos/test_detected.mp4` 当前不存在。

## 关键文件

- `main.py`：单视频检测入口。
- `requirements.txt`：项目 Python 依赖。
- `README.md`：安装、输入、运行和输出说明。
- `outputs/logs/experiment_01.md`：`conf=0.20`、`0.35`、`0.60` 的实验记录模板，三组 `iou` 均为 `0.50`。
- `AGENTS.md`：后续修改必须遵守的要求。

## 恢复工作

在项目根目录执行：

```powershell
conda activate visual_analysis
python --version
```

检查运行依赖和 CUDA：

```powershell
python -c "import torch, ultralytics, cv2; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Ultralytics:', ultralytics.__version__); print('OpenCV:', cv2.__version__)"
```

如果依赖尚未安装：

```powershell
python -m pip install -r requirements.txt
```

将测试视频放置为：

```text
data/videos/test.mp4
```

运行基线：

```powershell
python main.py
```

## 建议的下一步

1. 验证 `visual_analysis` 环境中的依赖和 CUDA 可用性。
2. 用户自行准备 `data/videos/test.mp4`。
3. 运行 `main.py`，确认输出视频可播放并记录处理帧数。
4. 分别以 `conf=0.20`、`0.35`、`0.60` 运行检测，将结果填写到 `outputs/logs/experiment_01.md`。
5. 第一阶段验证完成后，再根据明确需求规划后续模块。
