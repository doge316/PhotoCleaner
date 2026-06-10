# PhotoCleaner — 路人自动识别与消除系统

面向景区、展馆、活动现场等真实拍照场景。自动判断主体与路人，完成分割、消除与背景修复。

## 快速开始

### 一键部署（推荐）

```bash
# Linux / macOS
bash setup.sh

# Windows（双击运行或在终端执行）
setup.bat
```

完成后启动：

```bash
# Linux / macOS
source venv/bin/activate && streamlit run app.py

# Windows
venv\Scripts\activate.bat && streamlit run app.py
```

### 手动部署

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python setup_models.py
streamlit run app.py
```

## 模型文件

所有模型通过 `python setup_models.py` 下载到 `models/` 目录：

| 文件 | 大小 | 用途 |
|------|------|------|
| `yolov8s-seg.pt` | 23 MB | 人物检测 + 实例分割 |
| `mobile_sam.pt` | 39 MB | SAM 高精度分割 mask |
| `big-lama.pt` | 197 MB | LaMa 背景修复 |
| `midas_v21_small_256.pt` | 82 MB | MiDaS 单目深度估算 |
| `tf_efficientnet_lite3-b733e338.pth` | 32 MB | 深度模型骨架权重 |
| `blaze_face_short_range.tflite` | 225 KB | BlazeFace 人脸检测 |

## 系统要求

- Python 3.10+
- 16 GB+ 内存（CPU 模式，无需 GPU）
- Linux / macOS / Windows

## 项目结构

```
PhotoCleaner/
├── app.py                  # Streamlit Web 界面
├── photo_cleaner_core.py   # 核心管线（YOLO + SAM + LaMa）
├── photo_cleaner_qwen.py   # Qwen API 云端消除管线
├── llm_subject_selector.py # 主体/路人判断（多维度特征融合）
├── subject_features.py     # 深度 + 人脸特征提取
├── db.py                   # SQLite 处理日志
├── setup_models.py         # 模型文件下载脚本
├── requirements.txt        # Python 依赖
├── vendor/                 # 离线依赖源码（无需联网）
│   ├── midas/              #   MiDaS 深度模型
│   └── geffnet/            #   EfficientNet 骨架
└── models/                 # 模型权重（setup_models.py 下载）
```

## 处理流程

1. **YOLOv8-seg** 检测图中所有人物 → bbox + 粗 mask
2. **SAM (MobileSAM)** 根据 bbox 生成高精度 mask
3. **MiDaS + BlazeFace + 几何特征** 多维度打分，区分主体/路人
4. **LaMa** 消除所有人 → 干净背景
5. 主体人物从原图**羽化回贴**到干净背景

## 命令行使用

```bash
python photo_cleaner_core.py <输入路径> <输出目录>
```