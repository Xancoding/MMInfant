# Multimodal Infant Cry Detection

本模块是婴儿多模态哭泣检测系统的**主分析管道**，负责从原始音频和 InfantVision 预处理得到的视觉 JSON 特征中提取高阶统计特征，并基于 SVM / Random Forest / LightGBM 进行单模态与多模态（早期融合 / 晚期融合）分类评估。

---

## 功能概述

### 1. 三模态特征提取 (`features/`)

| 模态 | 文件 | 输入 | 说明 |
|---|---|---|---|
| **Audio** | `Feature_Extraction_Audio.py` | `.wav` 原始音频 | 基于 pyAudioAnalysis 的短时音频特征：ZCR、能量、频谱特征、MFCC、Chroma 等，并计算每个滑动窗口的 mean / median / std |
| **Motion** | `Feature_Extraction_Body.py` | `*_motion_features.json` | 读取 InfantVision 输出的身体部位光流幅度，按滑动窗口提取 std / median / mean / max / min |
| **Face** | `Feature_Extraction_Face.py` | `*_face_landmarks.json` | 基于 68 点面部关键点计算 MAR（嘴部纵横比）、左右 EAR（眼部纵横比），按滑动窗口提取统计特征 |

### 2. 数据加载与缓存 (`data_loader.py`)

- 自动扫描数据目录下的 `.wav` 文件
- 首次运行时提取三模态特征并缓存为 `.npz` 文件（存储在 `../Features/` 目录）
- 后续运行直接加载缓存，大幅提升效率
- 支持标签（Label）与场景（Scene）的同步分割

### 3. 模型评估 (`evaluator.py`)

- **分类器**：SVM (RBF)、Random Forest、LightGBM（通过 `config.py` 切换）
- **交叉验证**：StratifiedGroupKFold（10 折，按婴儿 ID 分组，避免信息泄漏）
- **数据平衡**：训练集内使用 RandomOverSampler 过采样
- **缺失值处理**：按折内训练集拟合 `SimpleImputer(median)`，并统一 indicator 维度
- **评估指标**：Accuracy、Precision、Recall、Specificity、F1-Score
- **场景分析**：支持按场景（Scene）细分统计准确率（可选开启）
- **特征重要性**：树模型（RF / LightGBM）支持输出归一化特征重要性

### 4. 多模态融合策略

| 策略 | 方法 | 说明 |
|---|---|---|
| **单模态** | 独立评估 | Audio / Motion / Face 分别输入单一分类器 |
| **Early Fusion** | 特征级拼接 | 三模态特征向量直接拼接后输入单一分类器 |
| **Late Fusion** | 决策级 Stacking | 每个模态训练独立基分类器，Meta-Classifier（LogisticRegression）融合概率输出 |

### 5. 主控入口 (`main.py`)

通过修改 `tasks` 列表灵活启用/禁用评估任务：

```python
tasks = [
    # 'audio', 
    # 'motion', 
    # 'face', 
    'early_fusion', 
    # 'late_fusion',
]
```

---

## 目录结构

```
code/
├── main.py                          # 主控入口
├── config.py                        # 全局配置（模型、路径、窗口参数等）
├── data_loader.py                   # 数据加载、特征提取、缓存管理
├── evaluator.py                     # 模型评估器（单模态 / 融合 / 交叉验证）
├── utils.py                         # 通用工具（随机种子、路径构造、异常值过滤等）
│
├── features/
│   ├── Feature_Extraction_Audio.py  # 音频短时特征提取
│   ├── Feature_Extraction_Body.py   # 身体运动滑动窗口特征
│   └── Feature_Extraction_Face.py   # 面部关键点（MAR / EAR）滑动窗口特征
│
└── README.md                        # 本文件
```

---

## 环境依赖

### 基础环境

- Python >= 3.8

### Python 依赖

```bash
pip install numpy scipy matplotlib tqdm
pip install scikit-learn lightgbm imbalanced-learn
pip install librosa
pip install torch  # 仅用于 evaluator 中的随机种子控制
```

> `imbalanced-learn` 提供 `RandomOverSampler` 用于类别平衡。

---

## 数据集准备

本模块支持 `NICU50` 与 `NEWBORN200` 两个数据集，目录结构如下（以 `NICU50` 为例）：

```
dataset/NICU50/
├── data/
│   ├── subject_001.wav
│   ├── subject_002.wav
│   └── ...
│
├── Label/                           # 哭声标签（与 wav 同名）
│   ├── subject_001.txt              # 格式: <start_time>\t<end_time>\t<label(0/1)>
│   └── ...
│
├── Scene/                           # 场景标签（可选）
│   ├── subject_001.txt              # 格式: <start_time>\t<end_time>\t<scene_type>
│   └── ...
│
├── Body/                            # InfantVision 身体运动输出
│   ├── subject_001_motion_features.json
│   └── ...
│
├── Face/                            # InfantVision 面部关键点输出
│   ├── subject_001_face_landmarks.json
│   └── ...
│
└── Features/                        # 自动生成的特征缓存（首次运行后生成）
    ├── subject_001_metadata.npz
    ├── subject_001_acoustic.npz
    ├── subject_001_motion.npz
    └── subject_001_face.npz
```

### 标签文件格式

每行一条记录，制表符分隔：

```
0.00	3.50	1
5.20	8.00	0
```

- `label = 1`：该时间段存在哭声
- `label = 0`：该时间段无哭声

### 场景文件格式（可选）

```
0.00	10.00	3
10.00	20.00	1
```

场景类型优先级（数值越小优先级越高）：`{3:0, 1:1, 4:2, 2:3, 8:4, 7:5, 6:6, 5:7}`。若不存在 Scene 目录，则所有片段场景标记为 `0`。

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
| `dataDir` | `'/data/.../NICU50/data/'` | 数据目录（`.wav` 所在路径） |

修改 `model_type` 即可切换分类器；修改 `dataDir` 即可切换数据集。

---

## 使用方法

### 1. 配置数据路径

编辑 `config.py`，将 `dataDir` 指向实际数据目录：

```python
dataDir = '/path/to/your/dataset/NICU50/data/'
```

### 2. 选择评估任务

编辑 `main.py` 中的 `tasks` 列表，启用需要的任务：

```python
tasks = ['audio', 'motion', 'face', 'early_fusion', 'late_fusion']
```

### 3. 配置输出选项

编辑 `main.py` 顶部的两个开关，控制附加信息的打印：

| 开关 | 默认值 | 设为 `True` 后的效果 |
|---|---|---|
| `enable_scene_printing` | `False` | 打印数据集**场景分布统计**，以及每折评估的**按场景准确率** |
| `enable_feature_printing` | `False` | 树模型（RF / LightGBM）运行 Early Fusion 后，打印 **Top N 特征重要性** |

```python
enable_feature_printing = False
enable_scene_printing = False   # 设为 True 可查看场景分布与按场景准确率
```

> 场景分析需要数据集包含 `Scene/` 标签文件；若不存在，所有片段场景标记为 `0`，此时场景分析无区分意义。

### 4. 运行

```bash
cd code
python main.py
```

---

## 特征维度说明

### 音频特征 (`Feature_Extraction_Audio.py`)

对每个音频片段提取以下短时特征（34 维），然后计算每个滑动窗口的 **mean / median / std**：

| 特征类别 | 具体特征 | 维度 |
|---|---|---|
| 时域 | ZCR、Energy、Energy Entropy | 3 |
| 频域 | Spectral Centroid、Spread、Entropy、Flux、Rolloff | 5 |
| MFCC | MFCC 1~13 | 13 |
| Chroma | Chroma 1~12 + Chroma Std | 13 |

**总计**：34 × 3 = **102 维**

### 身体运动特征 (`Feature_Extraction_Body.py`)

基于 InfantVision 输出的各部位光流幅度 `shift_r`，提取以下统计量：

| 部位 | 统计量 | 维度 |
|---|---|---|
| Face、Left-arm、Right-arm、Left-leg、Right-leg | std、median、mean、max、min | 5 × 5 = **25 维** |

> 当前默认提取 5 个身体部位。代码中可通过修改 `body_parts` 列表切换为 `WholeBody` 或 `WholeFrameMotion`。

### 面部特征 (`Feature_Extraction_Face.py`)

基于 68 点面部关键点，逐帧计算：

- **MAR**（Mouth Aspect Ratio）：嘴部张开程度
- **Left EAR** / **Right EAR**（Eye Aspect Ratio）：左右眼部张开程度

然后通过置信度阈值过滤低质量帧，对每类指标提取统计量：

| 指标 | 统计量 | 维度 |
|---|---|---|
| MAR、Left EAR、Right EAR | mean、median、std、min、max | 3 × 5 = **15 维** |

> 当前 `data_loader.py` 中默认仅保留 MAR 的 5 维特征（`face_feature_names[:5]`），如需使用完整 15 维，请注释掉对应代码行。

---

## 评估策略详解

### 分组策略

采用 `StratifiedGroupKFold`，按**婴儿基础 ID** 分组（而非单个片段），确保同一婴儿的所有片段不会同时出现在训练集和测试集中，避免信息泄漏。

### 缺失值处理

- 在交叉验证的每一折内，基于**训练集**拟合 `SimpleImputer(strategy='median', add_indicator=True)`
- 测试集仅做 `transform`，严格遵守不泄漏原则
- 强制对齐 indicator 维度，确保多折间特征维度一致

### 类别不平衡

训练阶段使用 `RandomOverSampler` 对少数类进行过采样，使训练集类别均衡。

### Late Fusion (Stacking)

1. 将三模态特征拼接为联合特征矩阵
2. 通过 `ColumnTransformer` 为每个模态构建独立 Pipeline（Imputer → StandardScaler → Classifier）
3. 以各基分类器的预测概率作为 Meta-Feature
4. Meta-Classifier 采用 `LogisticRegression(class_weight='balanced')`

---

## 与 InfantVision 的衔接

```
原始视频 ──→ InfantVision/body_seg_5.py ──→ Body/*_motion_features.json
         ──→ InfantVision/facial_landmark_68.py ──→ Face/*_face_landmarks.json
                                                          ↓
原始音频 ─────────────────────────────────────────────────→ code/main.py
                                                          ↓
                                              三模态特征 → 分类评估
```

运行本模块前，请确保：
1. InfantVision 已正确运行，生成 `Body/` 和 `Face/` 下的 JSON 文件
2. `config.py` 中的 `dataDir` 指向包含 `.wav` 的目录
3. 同级目录下存在对应的 `Label/`、`Scene/`、`Body/`、`Face/` 文件夹

---

## 注意事项

1. **首次运行较慢**：需要逐文件提取音频、运动、面部特征并保存 `.npz` 缓存；后续运行直接读取缓存。
2. **缓存失效**：若修改了特征提取逻辑或换了数据集，请手动删除 `Features/` 目录下的 `.npz` 文件以强制重新提取。
3. **特征子集选择**：`data_loader.py` 中硬编码了仅保留 MAR 面部特征，如需完整面部特征请注释掉对应切片代码。
4. **GPU 需求**：本模块训练阶段仅使用 CPU（sklearn / LightGBM），`torch` 仅用于 evaluator 中的随机种子同步。
5. **路径问题**：所有路径通过 `config.py` 和 `utils.py` 动态构造，请确保目录层级与上文「数据集准备」一致。
