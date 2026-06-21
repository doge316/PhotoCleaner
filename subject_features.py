"""
subject_features.py
-------------------
路人/主体判断的辅助特征模块。
提供两个能力：
  1. 单目深度估算（MiDaS v3.1 Small，离镜头越近 disparity 越大）
  2. 人脸检测（OpenCV YuNet，用于评估"人脸完整度"）

设计原则：
  * 模型懒加载 + @lru_cache，进程内只加载一次
  * 任何模型加载/推理失败都只记日志，不抛异常，整体流程必须能继续跑
  * 关闭开关或模型不可用时，特征值返回中性（None），调用方按规则降权
"""

from __future__ import annotations

import urllib.request
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from device_utils import get_device

# ---------------------------------------------------------------------------
# 配置与全局状态
# ---------------------------------------------------------------------------

# 注释：模型缓存目录放项目根下 ./models/，方便甲方打包交付
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models"

# 注释：MiDaS 本地权重文件名（如果有内网环境，手动放这个文件进 models/）
_MIDAS_MODEL_FILENAME = "midas_v21_small_256.pt"

# BlazeFace 模型，Google 官方 CDN，~440KB
_BLAZEFACE_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
_BLAZEFACE_FILENAME = "blaze_face_short_range.tflite"

def get_model_dir() -> Path:
    """获取模型目录，如果不存在就创建。"""
    _DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return _DEFAULT_MODEL_DIR


# ---------------------------------------------------------------------------
# 单目深度估算：MiDaS Small
# ---------------------------------------------------------------------------


def _load_midas_model():
    """
    加载 MiDaS v3.1 Small 深度模型（vendored 源码 + 本地权重）。
    返回 (model, transform, device) 三元组。
    失败返回 (None, None, cpu)。
    """
    device = get_device()
    try:
        import torch
    except ImportError:
        print("[depth] torch 未安装，跳过深度估算。")
        return None, None, device

    local_weight_path = get_model_dir() / _MIDAS_MODEL_FILENAME
    if not local_weight_path.exists():
        print(f"[depth] 未找到 MiDaS 权重: {local_weight_path}，跳过深度估算。")
        print("[depth] 请运行 python setup_models.py 下载模型文件。")
        return None, None, device

    from torchvision.transforms import Compose, Resize, ToTensor, Normalize

    transform = Compose([
        Resize((256, 256)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        _project_root = Path(__file__).resolve().parent
        os.environ.setdefault("TORCH_HOME", str(_project_root / "models"))

        import sys
        _vendor_root = str(_project_root / "vendor")
        if _vendor_root not in sys.path:
            sys.path.insert(0, _vendor_root)

        from midas.midas_net_custom import MidasNet_small

        print(f"[depth] 从本地权重加载 MiDaS 到 {device}……")
        state = torch.load(str(local_weight_path), map_location=str(device),
                           weights_only=True)

        model = MidasNet_small(
            str(local_weight_path),
            features=64,
            backbone="efficientnet_lite3",
            exportable=True,
            non_negative=True,
            blocks={"expand": True},
        )

        model.eval()
        model = model.to(device)
        for p in model.parameters():
            p.requires_grad = False

        print(f"[depth] MiDaS 模型加载完成（{device}，vendored 源码）。")
        return model, transform, device
    except Exception as exc:
        print(f"[depth] MiDaS 加载失败: {type(exc).__name__}: {exc}")
        print("[depth] 将跳过深度特征。")
        return None, None, device


@lru_cache(maxsize=1)
def get_midas():
    """进程内只加载一次的 MiDaS 模型。"""
    return _load_midas_model()


def compute_depth_map(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    计算整张图像的深度图（disparity，越大表示离镜头越近）。
    返回与原图同尺寸的 float32 数组，缺失时返回 None。
    """
    model, transform, device = get_midas()
    if model is None or transform is None:
        return None
    try:
        import torch
        from PIL import Image

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        input_tensor = transform(image_pil).unsqueeze(0).to(device)  # (1, 3, 256, 256)

        with torch.no_grad():
            prediction = model(input_tensor)
            # 注释：MiDaS 输出是 256x256，插值回原图尺寸
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image_bgr.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy().astype(np.float32)
        # 注释：归一化到 0..1，不同图之间才能比较
        d_min, d_max = float(depth.min()), float(depth.max())
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)
        return depth
    except Exception as exc:
        print(f"[depth] 深度图推理失败: {type(exc).__name__}: {exc}")
        return None


def compute_box_depth_score(box: list[float], depth_map: Optional[np.ndarray]) -> Optional[float]:
    """
    给定一个人物检测框，计算框内区域的平均 disparity。
    返回 0..1 的相对分数（>0 表示离镜头更近）。
    如果 depth_map 为空，返回 None。
    """
    if depth_map is None:
        return None
    image_h, image_w = depth_map.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    # 注释：clamp 到图像范围内，避免越界
    x1 = max(0, min(image_w, x1))
    x2 = max(0, min(image_w, x2))
    y1 = max(0, min(image_h, y1))
    y2 = max(0, min(image_h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    roi = depth_map[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return float(roi.mean())


# ---------------------------------------------------------------------------
# 人脸检测：MediaPipe Face Detection（Google 出品，pip 装完即用，零配置）
# ---------------------------------------------------------------------------

def _download_blazeface_if_missing() -> Optional[Path]:
    """
    从 Google 官方 CDN 下载 BlazeFace 模型。
    本地已有且大小正常就直接返回，断了网也能用本地缓存。
    """
    target = get_model_dir() / _BLAZEFACE_FILENAME
    if target.exists() and target.stat().st_size > 100_000:
        return target
    try:
        print(f"[face] 正在下载 BlazeFace 模型到 {target} ……")
        urllib.request.urlretrieve(_BLAZEFACE_URL, str(target))
        if target.exists() and target.stat().st_size > 100_000:
            print(f"[face] BlazeFace 模型下载完成，{target.stat().st_size} 字节。")
            return target
        else:
            print("[face] BlazeFace 模型下载失败（文件太小，可能被拦截）")
    except Exception as exc:
        print(f"[face] BlazeFace 模型下载失败: {type(exc).__name__}: {exc}")
    return None


def _load_face_detector():
    """
    使用 mediapipe Tasks API（适用于 mediapipe >= 0.10.30）加载人脸检测器。
    失败返回 None。
    """
    # WSL/Docker 补丁：注入项目 lib/ 和 ~/.local/lib 到库搜索路径
    # 注意：需要同时用 ctypes 预加载，因为 os.environ 修改对当前进程的动态链接器不一定生效
    _project_lib = str(Path(__file__).resolve().parent / "lib")
    _lib_dirs = []
    if os.path.isdir(_project_lib):
        _lib_dirs.append(_project_lib)
    _local_lib = os.path.expanduser("~/.local/lib")
    if os.path.isdir(_local_lib):
        _lib_dirs.append(_local_lib)
    if _lib_dirs and "LD_LIBRARY_PATH" not in os.environ.get("_GLES_FIXED", ""):
        # 预加载 libGLESv2，避免 mediapipe import 时找不到
        for _d in _lib_dirs:
            _gles = os.path.join(_d, "libGLESv2.so.2")
            if os.path.isfile(_gles):
                try:
                    import ctypes
                    ctypes.CDLL(_gles, mode=ctypes.RTLD_GLOBAL)
                except Exception:
                    pass
        os.environ["LD_LIBRARY_PATH"] = ":".join(_lib_dirs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["_GLES_FIXED"] = "1"

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError as exc:
        print(f"[face] mediapipe Tasks API 导入失败: {exc}")
        return None

    model_path = _download_blazeface_if_missing()
    if model_path is None:
        return None

    try:
        with open(model_path, "rb") as f:
            model_bytes = f.read()
        base_options = mp_python.BaseOptions(
            model_asset_buffer=model_bytes,
            delegate=mp_python.BaseOptions.Delegate.CPU,
        )
        options = mp_vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.5,
        )
        detector = mp_vision.FaceDetector.create_from_options(options)
        print("[face] mediapipe Tasks 人脸检测器加载完成。")
        return detector
    except Exception as exc:
        print(f"[face] mediapipe Tasks 检测器创建失败: {type(exc).__name__}: {exc}")
        return None


@lru_cache(maxsize=1)
def get_face_detector():
    """进程内只加载一次的人脸检测器。"""
    return _load_face_detector()


def detect_faces(image_bgr: np.ndarray) -> list[dict]:
    """
    检测图中所有人脸。
    返回 list[dict]，每个元素包含 {box: (x, y, w, h), conf: float}。
    失败或检测器未加载时返回空列表。
    """
    detector = get_face_detector()
    if detector is None:
        return []
    try:
        import mediapipe as mp
        # 注释：Tasks API 用 mp.Image 包装图像
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = detector.detect(mp_image)
        faces: list[dict] = []
        for det in result.detections:
            # 注释：Tasks API 直接返回绝对像素坐标
            bbox = det.bounding_box
            x = float(bbox.origin_x)
            y = float(bbox.origin_y)
            bw = float(bbox.width)
            bh = float(bbox.height)
            conf = 0.0
            if det.categories:
                conf = float(det.categories[0].score)
            faces.append({
                "box": (x, y, bw, bh),
                "conf": conf,
            })
        return faces
    except Exception as exc:
        print(f"[face] 人脸推理失败: {type(exc).__name__}: {exc}")
        return []


def compute_face_completeness_score(person_box: list[float], faces: list[dict]) -> float:
    """
    评估一个人物框内"人脸完整度"，返回 0..1 之间的分数。
    计算逻辑：
      1. 找到中心点落在 person_box 内的最大面积人脸
      2. score = 是否有人脸 (0.4) + 人脸面积/人物框面积 (0..0.4) + 脸中心居中度 (0..0.2)
    若没有脸或脸太小，得分为 0。
    """
    if not faces:
        return 0.0
    px1, py1, px2, py2 = [float(v) for v in person_box]
    pw, ph = max(1.0, px2 - px1), max(1.0, py2 - py1)
    person_area = pw * ph
    pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0

    best = None  # 注释：记录 (face_area, face_cx, face_cy, face_h_ratio)
    for f in faces:
        fx, fy, fw, fh = f["box"]
        face_cx, face_cy = fx + fw / 2.0, fy + fh / 2.0
        # 注释：脸中心必须落在 person_box 里，否则不算这个人
        if not (px1 <= face_cx <= px2 and py1 <= face_cy <= py2):
            continue
        face_area = fw * fh
        face_h_ratio = fh / ph
        if best is None or face_area > best[0]:
            best = (face_area, face_cx, face_cy, face_h_ratio)

    if best is None:
        return 0.0

    face_area, face_cx, face_cy, face_h_ratio = best
    area_ratio = min(face_area / person_area, 1.0)

    # 注释：居中度 = 脸中心相对人物框中心的归一化距离
    dx = abs(face_cx - pcx) / pw
    dy = abs(face_cy - pcy) / ph
    centered = max(0.0, 1.0 - (dx + dy))
    centered = min(centered, 1.0)

    # 注释：脸相对人物框高度 < 8% → 视为不可用，直接给 0
    if face_h_ratio < 0.08:
        return 0.0

    return 0.4 * 1.0 + 0.4 * area_ratio + 0.2 * centered


# ---------------------------------------------------------------------------
# 人物完整度（不依赖外部模型，纯几何判断）
# ---------------------------------------------------------------------------


def compute_completeness_score(box: list[float], image_h: int, image_w: int) -> float:
    """
    评估人物被截断/部分可见的程度。
    评估方式：
      1. 框是否贴到图像边缘 → 贴边说明可能被截断（扣分）
      2. 框的宽高比 → 极扁的可能是仅露出肢体（扣分）
      3. 框面积占图比 → 太小（远处且小）的不太可能是主体（轻度扣分）
    返回 0..1 之间的分数。
    """
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)

    score = 1.0

    # 1) 边缘裁切检测：留 1 像素余量
    edge_threshold = 1
    if x1 <= edge_threshold or y1 <= edge_threshold:
        score -= 0.4
    if (image_w - x2) <= edge_threshold or (image_h - y2) <= edge_threshold:
        score -= 0.4

    # 2) 宽高比：正常站立人物 h/w ≈ 1.5~3.5
    aspect = h / w
    if aspect < 0.8 or aspect > 5.0:
        score -= 0.3

    # 3) 占图面积：过小（远处路人）适度降权
    area_ratio = (w * h) / (image_h * image_w)
    if area_ratio < 0.005:  # 0.5% 以下
        score -= 0.2

    return float(max(0.0, min(1.0, score)))


# ---------------------------------------------------------------------------
# 一站式：给一堆检测框批量算深度+人脸特征
# ---------------------------------------------------------------------------


@dataclass
class BoxFeatures:
    """每个人物检测框的扩展特征汇总。"""
    depth_mean: Optional[float] = None  # 0..1，越大越近
    face_score: float = 0.0            # 0..1
    completeness: float = 1.0          # 0..1
    mask_fill_ratio: Optional[float] = None  # mask 像素 / 框面积


def extract_features_for_detections(
    image_bgr: np.ndarray,
    detections: list[dict],
    masks: Optional[np.ndarray] = None,
    enable_depth: bool = True,
    enable_face: bool = True,
) -> list[BoxFeatures]:
    """
    对所有检测框一次性算齐三种特征。
    `detections` 元素需包含 'index' 和 'box' 字段。
    `masks` 形状 (N, H, W)，可选，用于计算 mask 填充比。
    """
    image_h, image_w = image_bgr.shape[:2]

    # 1) 深度图（只算一次）
    depth_map = None
    if enable_depth:
        depth_map = compute_depth_map(image_bgr)

    # 2) 人脸检测（只跑一次）
    faces: list[dict] = []
    if enable_face:
        faces = detect_faces(image_bgr)

    # 3) 对每个检测框求特征
    features_list: list[BoxFeatures] = []
    for det in detections:
        box = det["box"]
        idx = det.get("index", -1)

        # 深度均值
        depth_score = None
        if depth_map is not None:
            depth_score = compute_box_depth_score(box, depth_map)

        # mask 填充比
        mask_ratio = None
        if masks is not None and 0 <= idx < len(masks):
            m = masks[idx] > 0.5
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1, x2 = max(0, x1), min(image_w, x2)
            y1, y2 = max(0, y1), min(image_h, y2)
            box_area = max(1, (x2 - x1) * (y2 - y1))
            mask_in_box = int(m[y1:y2, x1:x2].sum())
            mask_ratio = mask_in_box / box_area

        features_list.append(BoxFeatures(
            depth_mean=depth_score,
            face_score=compute_face_completeness_score(box, faces),
            completeness=compute_completeness_score(box, image_h, image_w),
            mask_fill_ratio=mask_ratio,
        ))
    return features_list