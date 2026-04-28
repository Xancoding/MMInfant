# Multimodal Infant Cry Detection

本模块是婴儿多模态哭泣检测系统的**主分析管道**，负责从原始音频和 InfantVision 预处理得到的视觉 JSON 特征中提取高阶统计特征，并基于 SVM / Random Forest / LightGBM 进行单模态与多模态（早期融合 / 晚期融合）分类评估。

---

## 功能概述

### 1. 三模态特征提取 (`features/`)

| 模态 | 文件 | 输入 | 说明 |
|---|---|---|---|
| **Audio** | `Feature_Extraction_Audio.py` | `.wav` 原始音频 | 短时音频特征（ZCR、能量、MFCC、Chroma 等），按滑动窗口计算统计量 |
| **Motion** | `Feature_Extraction_Body.py` | `*_motion_features.json` | 读取 InfantVision 输出的身体部位光流幅度，按滑动窗口提取统计量 |
| **Face** | `Feature_Extraction_Face.py` | `*_face_landmarks.json` | 基于 68 点面部关键点计算 MAR / EAR，按滑动窗口提取统计特征 |

### 2. 数据加载与缓存 (`data_loader.py`)

- 读取同目录下的 `sample_manifest.txt` 获取样本列表
- 提取三模态特征（音频、运动、面部），并按样本分别缓存为 4 个 `.npz` 文件（metadata / acoustic / motion / face）
- 缓存在 `{dataDir父目录}/Features/` 下，后续运行直接加载缓存（缓存损坏时自动重新提取）

### 3. 模型评估 (`evaluator.py`)

- **分类器**：SVM (RBF)、Random Forest、LightGBM（通过 `config.py` 切换）
- **交叉验证**：StratifiedGroupKFold（10 折，按婴儿 ID 分组）
- **融合策略**：单模态评估、Early Fusion（特征拼接）、Late Fusion（Stacking）

---

## 配置说明 (`config.py`)

| 参数 | 默认值 | 说明 |
|---|---|---|
| `seed` | `42` | 全局随机种子 |
| `audioSampleRate` | `16000` | 音频重采样率 |
| `n_splits` | `10` | 交叉验证折数 |
| `slidingWindows` | `2.5` | 滑动窗口时长（秒） |
| `step` | `1.5` | 滑动窗口步长（秒） |
| `FFTwindow` | `256` | 音频 FFT 窗长（样本数） |
| `FFTOverlap` | `128` | 音频 FFT 步长（样本数） |
| `MFCCFiliterNum` | `26` | MFCC 滤波器组数量 |
| `model_type` | `'lgbm'` | 分类器类型：`svm`、`rf`、`lgbm` |
| `dataDir` | `'{PROJECT_ROOT}/dataset/NICU50/data/'` | 数据目录（`.wav` 所在路径） |

修改 `model_type` 即可切换分类器；修改 `dataDir` 即可切换数据集。

---

## 使用方法

编辑 `main.py` 中的 `tasks` 列表，选择要运行的评估任务：

```python
tasks = ['audio', 'motion', 'face', 'early_fusion', 'late_fusion']
```

编辑 `main.py` 顶部的开关，控制附加信息的打印：

| 开关 | 默认值 | 设为 `True` 后的效果 |
|---|---|---|
| `enable_scene_printing` | `False` | 打印数据集**场景分布统计**，以及每折评估的**按场景准确率** |
| `enable_feature_printing` | `False` | 树模型（RF / LightGBM）运行 Early Fusion 后，打印 **Top N 特征重要性** |

```python
enable_feature_printing = False
enable_scene_printing = False   
```

> 场景分析需要数据集包含 `Scene/` 标签文件；若不存在，所有片段场景标记为 `0`，此时场景分析无区分意义。

### 运行

```bash
cd code
python main.py
```

---