# InfantVision

**InfantVision** 是婴儿多模态哭泣检测系统的视觉特征提取子模块，负责从婴儿视频中提取身体运动特征与面部关键点特征，为下游的多模态机器学习分析提供视觉模态输入。

---

## 功能概述

本模块包含两条独立的视频处理流水线：

### 1. 人体部位分割与运动分析 (`body_seg_5.py`)

- **人体部位分割**：基于 [ModelScope `iic/cv_resnet101_image-multiple-human-parsing`](https://modelscope.cn/models/iic/cv_resnet101_image-multiple-human-parsing) 模型，对视频帧进行像素级人体部位解析，提取以下部位掩码：
  - 左臂 (Left-arm) / 右臂 (Right-arm)
  - 左腿 (Left-leg) / 右腿 (Right-leg)
  - 面部 (Face)
  - 躯干皮肤 (Torso-skin)
- **光流运动估计**：使用 Farneback 光流算法计算各部位及全局画面的运动向量。
- **输出**：
  - 带分割掩码叠加的演示视频 (`*_masked.avi`)
  - 首帧分割可视化图 (`*_seg_result.png`)
  - 逐帧运动特征 JSON (`*_motion_features.json`)

### 2. 面部 68 点关键点检测 (`facial_landmark_68.py`)

- **人脸检测与关键点回归**：基于 [Infant Facial Landmark Detection and Tracking](https://github.com/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking) 项目中的 YOLOv5-face 检测器与 HRNet 关键点模型，对婴儿面部进行边界框定位与 68 点关键点回归。
- **输出**：
  - 带关键点与边界框的演示视频 (`*_face_landmarks.avi`)
  - 首帧检测结果图 (`*_face_landmarks.png`)
  - 逐帧关键点与置信度 JSON (`*_face_landmarks.json`)

---

## 目录结构

```
InfantVision/
├── body_seg_5.py              # 身体部位分割与运动特征提取
├── facial_landmark_68.py      # 面部68点关键点检测
├── face_detector.py           # YOLOv5-face 人脸检测器封装
├── reproducibility_utils.py   # 随机种子与可复现性工具
│
├── ckpt/                      # 预训练模型权重（需自行准备）
│   ├── face.pt                # YOLOv5-face 检测器权重
│   └── hrnet-r90jt.pth        # HRNet 68点关键点权重
│
├── config/
│   └── hrnet-r90jt.yaml       # HRNet 模型配置文件
│
├── models/                    # YOLOv5-face 模型定义
│   ├── common.py
│   ├── experimental.py
│   └── yolo.py
│
├── utils/                     # YOLOv5-face 通用工具
│   ├── autoanchor.py
│   ├── datasets.py
│   ├── general.py
│   ├── google_utils.py
│   ├── metrics.py
│   ├── plots.py
│   └── torch_utils.py
│
├── lib/                       # HRNet 模型与配置库
│   ├── config/
│   ├── core/
│   ├── models/
│   └── utils/
│
└── README.md                  # 本文件
```

> **注意**：上游主项目 (`code/`) 中的 `Feature_Extraction_Body.py` 与 `Feature_Extraction_Face.py` 会读取本模块生成的 JSON 文件，进一步提取滑动窗口统计特征，供多模态分类器使用。

---

## 环境依赖

### 基础环境

- Python >= 3.8
- CUDA >= 11.0（推荐用于 GPU 加速）

### Python 依赖

```bash
pip install torch torchvision
pip install opencv-python
pip install numpy
pip install pillow
pip install tqdm
pip install matplotlib
pip install scipy
pip install modelscope
```

> `modelscope` 用于自动下载和加载 `cv_resnet101_image-multiple-human-parsing` 分割模型。

---

## 模型权重准备

运行脚本前，请将以下权重文件放置于 `ckpt/` 目录：

### 1. face.pt（YOLOv5-face 人脸检测器）

从 YOLOv5-face 官方项目下载预训练权重，重命名为 `face.pt` 后放入 `ckpt/`。

### 2. hrnet-r90jt.pth（HRNet 68点关键点模型）

从 [Infant Facial Landmark Detection and Tracking 项目](https://github.com/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking) 下载 `hrnet-r90jt.pth`，放入 `ckpt/` 目录。

---

## 使用方法

### 身体部位分割与运动分析

```bash
cd InfantVision
python body_seg_5.py
```

**默认配置**（可在脚本内修改）：
- 输入视频：`../demo/data/50cm3kg1.mp4`
- 输出目录：`../demo/Body/`
- 分割置信度阈值：`0.7`
- 最小掩码面积：`3000` 像素

### 面部关键点检测

```bash
cd InfantVision
python facial_landmark_68.py
```

**默认配置**（可在脚本内修改）：
- 输入视频：`../demo/data/50cm3kg1.mp4`
- 输出目录：`../demo/Face/`
- HRNet 配置：`./config/hrnet-r90jt.yaml`
- 检测器权重：`./ckpt/face.pt`
- 关键点权重：`./ckpt/hrnet-r90jt.pth`

---

## 输入与输出格式

### 输入

标准婴儿视频文件（`.mp4`、`.avi` 等），建议分辨率不低于 480p。

### 输出 1：身体运动特征 (`*_motion_features.json`)

```json
{
  "video_info": {
    "path": "...",
    "fps": 30.0,
    "frame_count": 900,
    "resolution": [1280, 720]
  },
  "features": [
    {
      "Frame": 0,
      "head": [width, height],
      "Face": [[shift_x, shift_y, shift_r, shift_a, score], ...],
      "Left-arm": [...],
      "Right-arm": [...],
      "Left-leg": [...],
      "Right-leg": [...],
      "Torso-skin": [...],
      "WholeBody": [[shift_x, shift_y, shift_r, shift_a, avg_score]],
      "WholeFrameMotion": [[shift_x, shift_y, shift_r, shift_a]]
    }
  ]
}
```

- `shift_x`, `shift_y`：光流向量均值
- `shift_r`：光流幅度均值
- `shift_a`：光流角度（度）
- `score`：模型分割置信度

### 输出 2：面部关键点 (`*_face_landmarks.json`)

```json
{
  "video_info": {
    "path": "...",
    "fps": 30.0,
    "frame_count": 900,
    "resolution": "1280x720"
  },
  "frames": [
    {
      "frame_number": 0,
      "timestamp": 0.0,
      "landmarks": [[x1, y1], [x2, y2], ...],
      "face_confidence": 0.95,
      "landmark_confidences": [c1, c2, ...]
    }
  ]
}
```

- `landmarks`：68 个面部关键点坐标 `(x, y)`
- `face_confidence`：人脸检测置信度
- `landmark_confidences`：每个关键点的热力图峰值置信度

---

## 与主项目的衔接

本模块属于数据预处理阶段。典型工作流如下：

1. **原始视频** → `InfantVision/body_seg_5.py` → `../demo/Body/*_motion_features.json`
2. **原始视频** → `InfantVision/facial_landmark_68.py` → `../demo/Face/*_face_landmarks.json`
3. **JSON 特征** → `code/features/Feature_Extraction_Body.py` → 运动模态特征矩阵
4. **JSON 特征** → `code/features/Feature_Extraction_Face.py` → 面部模态特征矩阵（MAR、EAR 等）
5. **多模态特征** → `code/main.py` → 音频 / 运动 / 面部 / 早期融合 / 晚期融合 评估

---

## 注意事项

1. **GPU 显存**：`body_seg_5.py` 默认使用 `cuda`，`facial_landmark_68.py` 同样默认使用 GPU。若显存不足，请修改脚本中的 `device` 参数为 `cpu`。
2. **跳过已处理文件**：两个脚本均会自动检测输出文件是否存在，若存在则跳过，方便批量重跑。
3. **Seaborn 依赖**：`utils/plots.py` 依赖 `seaborn`，若运行时提示缺少该包，请执行 `pip install seaborn`。
4. **模型权重路径**：脚本中的权重路径为相对路径（`./ckpt/...`），请确保在 `InfantVision/` 目录内运行。

---

## 引用

本模块使用了以下开源模型与代码库：

- **人体部位分割**：[
ModelScope / `iic/cv_resnet101_image-multiple-human-parsing`
](https://modelscope.cn/models/iic/cv_resnet101_image-multiple-human-parsing)
- **婴儿人脸检测与关键点检测**：[
Infant Facial Landmark Detection and Tracking (Ostadabbas Lab)
](https://github.com/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking)
  - 包含 YOLOv5-face 婴儿人脸检测器
  - 包含基于 InfantFace 数据集训练的 HRNet 68 点关键点模型
