"""
Qwen 云端全图人物消除 + 主体回贴算法

新管线（适配 MiDaS / BlazeFace 分支）：
1. YOLO 检测所有人 + 多尺度补充
2. select_subject_indices（LLM + 规则 fallback）识别主体
3. 调用云端 Qwen API 做全图人物消除（获取干净背景），失败则回退 LaMa
4. 将主体人物从原图提取，羽化后贴回干净背景
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Iterable
from urllib import error as urllib_error
from urllib import request as urllib_request

import cv2
import numpy as np
from PIL import Image

# 强制 CPU 模式
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

torch.cuda.is_available = lambda: False

_original_jit_load = torch.jit.load


def _patched_jit_load(*args, **kwargs):
    kwargs.setdefault("map_location", "cpu")
    return _original_jit_load(*args, **kwargs)


torch.jit.load = _patched_jit_load

from ultralytics import YOLO

from db import init_db, insert_record
from llm_subject_selector import (
    LLMSelectionConfig,
    image_to_data_url,
    select_subject_indices,
)

# 复用 photo_cleaner_core 的数据结构和工具函数
from photo_cleaner_core import (
    SUPPORTED_EXTENSIONS,
    DetectionSummary,
    ProcessResult,
    build_union_mask,
    collect_images,
    format_detection_lines,
    is_supported_image,
    load_model,
    load_sam_model,
    generate_sam_masks,
    refine_sam_masks_adaptive,
    save_result,
)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

DEFAULT_FEATHER_RADIUS = 5

QWEN_REMOVE_PROMPT = (
    "Remove all people from this image. "
    "Fill in the background naturally and seamlessly where people were removed. "
    "Return the complete edited image without any people. "
    "Make the background look as if the people were never there."
)

QWEN_REFINE_EDGES_PROMPT = (
    "Clean up the edges of this person cutout. "
    "Remove any background artifacts, halos, or bleeding at the boundaries. "
    "Make the silhouette edges crisp and accurate. "
    "Keep the person's appearance, pose, clothing, and colors exactly the same. "
    "Return the cleaned cutout with transparent background."
)

# DashScope 原生图像编辑 API
DASHSCOPE_IMAGE_EDIT_ENDPOINT = (
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/"
    "multimodal-generation/generation"
)
DASHSCOPE_IMAGE_EDIT_MODEL = "qwen-image-edit"


# ---------------------------------------------------------------------------
# LaMa 模型加载
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_lama_model():
    try:
        simple_lama_module = import_module("simple_lama_inpainting")
    except ImportError as exc:
        raise ImportError(
            "LaMa 后端不可用，请先安装 simple-lama-inpainting"
        ) from exc
    return simple_lama_module.SimpleLama()


def _lama_inpaint(image_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """用 LaMa 对 mask 区域做单次修复。自动对齐 mask 与图像尺寸。"""
    if mask_u8.sum() == 0:
        return image_bgr.copy()

    image_h, image_w = image_bgr.shape[:2]
    if mask_u8.shape[:2] != (image_h, image_w):
        mask_u8 = cv2.resize(mask_u8, (image_w, image_h), interpolation=cv2.INTER_NEAREST)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    mask_pil = Image.fromarray(mask_u8, mode="L")

    lama = _load_lama_model()
    result_pil = lama(image_pil, mask_pil)

    result_rgb = np.array(result_pil.convert("RGB"), dtype=np.uint8)
    if result_rgb.shape[:2] != (image_h, image_w):
        result_rgb = cv2.resize(result_rgb, (image_w, image_h))

    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)


def _lama_inpaint_twice(image_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """LaMa 双次修复。"""
    first = _lama_inpaint(image_bgr, mask_u8)
    return _lama_inpaint(first, mask_u8)


# ---------------------------------------------------------------------------
# Qwen 云端全图人物消除 API
# ---------------------------------------------------------------------------

def _is_dashscope(config: LLMSelectionConfig) -> bool:
    haystack = config.base_url.lower()
    return "dashscope" in haystack or "aliyun" in haystack or "aliyuncs" in haystack


def _download_image_from_url(url: str, timeout: int = 60) -> np.ndarray | None:
    try:
        req = urllib_request.Request(url, headers={"User-Agent": "PhotoCleaner/1.0"})
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            img_bytes = resp.read()
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img_bgr if img_bgr is not None else None
    except Exception:
        return None


def _extract_images_from_content_list(content: list) -> list[np.ndarray]:
    """从 DashScope / OpenAI 返回的 content 数组中提取所有图片。"""
    images: list[np.ndarray] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        image_ref = item.get("image", "")
        if isinstance(image_ref, str) and image_ref:
            if image_ref.startswith("data:image"):
                try:
                    b64_data = image_ref.split(",", 1)[1]
                    img_bytes = base64.b64decode(b64_data)
                    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img_bgr is not None:
                        images.append(img_bgr)
                except Exception:
                    pass
            elif image_ref.startswith("http"):
                img_bgr = _download_image_from_url(image_ref)
                if img_bgr is not None:
                    images.append(img_bgr)
        if item.get("type") == "image_url":
            url = item.get("image_url", {}).get("url", "")
            if url.startswith("data:image"):
                try:
                    b64_data = url.split(",", 1)[1]
                    img_bytes = base64.b64decode(b64_data)
                    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img_bgr is not None:
                        images.append(img_bgr)
                except Exception:
                    pass
            elif url.startswith("http"):
                img_bgr = _download_image_from_url(url)
                if img_bgr is not None:
                    images.append(img_bgr)
    return images


def _call_dashscope_image_edit(
    image_bgr: np.ndarray,
    config: LLMSelectionConfig,
) -> np.ndarray | None:
    """通过 DashScope 原生 API 调用 qwen-image-edit 消除人物。"""
    image_data_url = image_to_data_url(image_bgr)

    payload = {
        "model": DASHSCOPE_IMAGE_EDIT_MODEL,
        "input": {
            "messages": [{
                "role": "user",
                "content": [
                    {"image": image_data_url},
                    {"text": QWEN_REMOVE_PROMPT},
                ],
            }]
        },
    }

    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    request = urllib_request.Request(
        url=DASHSCOPE_IMAGE_EDIT_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib_request.urlopen(request, timeout=config.timeout_seconds) as response:
            response_text = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise RuntimeError(
            f"DashScope 图像编辑 API 失败: HTTP {exc.code} {exc.reason}: {error_text}"
        ) from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"DashScope 图像编辑 API 失败: {exc}") from exc

    response_json = json.loads(response_text)
    output = response_json.get("output", {})
    choices = output.get("choices", []) or response_json.get("choices", [])
    if not choices:
        return None

    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        images = _extract_images_from_content_list(content)
        if images:
            return images[-1]
    if isinstance(content, str):
        data_url_pattern = r"data:image/(?:png|jpeg|jpg|webp);base64,([A-Za-z0-9+/=]+)"
        matches = re.findall(data_url_pattern, content)
        if matches:
            try:
                img_bytes = base64.b64decode(matches[-1])
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    return img_bgr
            except Exception:
                pass
    return None


def _call_openai_compatible_image_edit(
    image_bgr: np.ndarray,
    config: LLMSelectionConfig,
) -> np.ndarray | None:
    """通过 OpenAI 兼容接口调用图像编辑模型。"""
    image_data_url = image_to_data_url(image_bgr)

    payload = {
        "model": config.model,
        "temperature": 0,
        "messages": [{
            "role": "system",
            "content": "你是一个图像编辑助手。收到照片后，移除图中的所有人，输出编辑后的完整图片。",
        }, {
            "role": "user",
            "content": [
                {"type": "text", "text": QWEN_REMOVE_PROMPT},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }],
    }

    url = config.base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    request = urllib_request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib_request.urlopen(request, timeout=config.timeout_seconds) as response:
            response_text = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise RuntimeError(
            f"图像编辑 API 调用失败: HTTP {exc.code} {exc.reason}: {error_text}"
        ) from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"图像编辑 API 调用失败: {exc}") from exc

    response_json = json.loads(response_text)
    choices = response_json.get("choices", [])
    if not choices:
        return None

    message = choices[0].get("message", {})
    content = message.get("content", "")

    if isinstance(content, list):
        images = _extract_images_from_content_list(content)
        if images:
            return images[-1]

    if isinstance(content, str):
        data_url_pattern = r"data:image/(?:png|jpeg|jpg|webp);base64,([A-Za-z0-9+/=]+)"
        matches = re.findall(data_url_pattern, content)
        if matches:
            try:
                img_bytes = base64.b64decode(matches[-1])
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    return img_bgr
            except Exception:
                pass

    images_field = message.get("images", [])
    if images_field:
        for img_entry in images_field:
            img_url = img_entry if isinstance(img_entry, str) else img_entry.get("url", "")
            if img_url.startswith("data:image"):
                try:
                    b64_data = img_url.split(",", 1)[1]
                    img_bytes = base64.b64decode(b64_data)
                    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img_bgr is not None:
                        return img_bgr
                except Exception:
                    pass
            elif img_url.startswith("http"):
                img_bgr = _download_image_from_url(img_url)
                if img_bgr is not None:
                    return img_bgr

    return None


def call_qwen_remove_all_people(
    image_bgr: np.ndarray,
    config: LLMSelectionConfig,
) -> np.ndarray | None:
    """调用云端 Qwen API 做全图人物消除。

    Returns:
        消除人物后的图像 (BGR)，API 不支持图像输出时返回 None 触发 LaMa fallback。
    """
    if _is_dashscope(config):
        return _call_dashscope_image_edit(image_bgr, config)
    else:
        return _call_openai_compatible_image_edit(image_bgr, config)


# ---------------------------------------------------------------------------
# 人物消除（Qwen 优先，LaMa 回退）
# ---------------------------------------------------------------------------

def remove_all_people_with_fallback(
    image_bgr: np.ndarray,
    all_people_mask: np.ndarray,
    config: LLMSelectionConfig,
) -> tuple[np.ndarray, str]:
    """尝试 Qwen API 做全图人物消除，失败则回退 LaMa。"""
    if all_people_mask is None or not all_people_mask.any():
        return image_bgr.copy(), "no_people"

    try:
        qwen_result = call_qwen_remove_all_people(image_bgr, config)
        if qwen_result is not None:
            if qwen_result.shape[:2] != image_bgr.shape[:2]:
                qwen_result = cv2.resize(qwen_result, (image_bgr.shape[1], image_bgr.shape[0]))
            return qwen_result, "qwen"
    except Exception:
        pass

    mask_u8 = (all_people_mask > 0).astype(np.uint8) * 255
    result = _lama_inpaint_twice(image_bgr, mask_u8)
    return result, "lama"


# ---------------------------------------------------------------------------
# 主体回贴
# ---------------------------------------------------------------------------

def _refine_single_mask_roi(
    roi_bgr: np.ndarray,
    roi_mask: np.ndarray,
    iterations: int = 3,
) -> np.ndarray:
    """对单个主体 ROI 做 GrabCut 精化，返回精化后的二值 mask。"""
    h, w = roi_bgr.shape[:2]
    if h < 10 or w < 10:
        return roi_mask

    original_area = int(roi_mask.sum())
    if original_area < 50:
        return roi_mask  # 太小，不精化

    # 根据原始 mask 面积动态调整腐蚀迭代次数
    # 小物体少腐蚀，避免把前景全消掉
    if original_area < 500:
        erode_iters = 1
    elif original_area < 2000:
        erode_iters = 1
    else:
        erode_iters = 1  # 全分辨率下 1 次就够

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    definite_fg = cv2.erode(roi_mask, kernel, iterations=erode_iters)
    if not definite_fg.any():
        return roi_mask

    # 构建 GrabCut mask: 确定前景 / 可能前景 / 其余为背景
    gc_mask = np.zeros((h, w), dtype=np.uint8)
    gc_mask[definite_fg > 0] = cv2.GC_FGD
    gc_mask[(roi_mask > 0) & (definite_fg == 0)] = cv2.GC_PR_FGD

    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    try:
        gc_mask, _, _ = cv2.grabCut(
            roi_bgr, gc_mask, None, bgd_model, fgd_model,
            iterCount=iterations, mode=cv2.GC_INIT_WITH_MASK,
        )
    except Exception:
        return roi_mask

    refined = ((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)).astype(np.uint8)

    # 安全校验：精化后的面积不能跟原始差太远
    refined_area = int(refined.sum())
    if refined_area < original_area * 0.2 or refined_area > original_area * 5:
        return roi_mask  # GrabCut 跑飞了，退回原始 mask

    return refined


def refine_subject_mask(
    image_bgr: np.ndarray,
    subject_mask: np.ndarray,
    padding: int = 15,
) -> np.ndarray:
    """用 GrabCut 对主体 mask 做边缘精化。

    对每个连通分量（即每个主体人物）单独做 GrabCut，
    避免多人物混合导致 GMM 混淆。
    返回精化后的二值 mask (uint8, 0/1)，与原图同尺寸。
    """
    h, w = image_bgr.shape[:2]

    # upscale 到全分辨率
    mask_u8 = (subject_mask > 0).astype(np.uint8)
    if mask_u8.shape[:2] != (h, w):
        mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_LINEAR)

    if not mask_u8.any():
        return mask_u8

    # 连通分量分析
    num_labels, labels = cv2.connectedComponents(mask_u8)
    refined = np.zeros((h, w), dtype=np.uint8)

    for label_id in range(1, num_labels):
        ys, xs = np.where(labels == label_id)
        if len(xs) < 20:   # 太小（< 20px），不精化，直接保留
            refined[ys, xs] = 1
            continue

        x1 = max(0, int(xs.min()) - padding)
        x2 = min(w, int(xs.max()) + padding)
        y1 = max(0, int(ys.min()) - padding)
        y2 = min(h, int(ys.max()) + padding)

        roi_bgr = image_bgr[y1:y2, x1:x2]
        roi_mask = (labels[y1:y2, x1:x2] == label_id).astype(np.uint8)

        refined_roi = _refine_single_mask_roi(roi_bgr, roi_mask)

        # 回填到全图
        existing = refined[y1:y2, x1:x2]
        refined[y1:y2, x1:x2] = np.maximum(existing, refined_roi)

    return refined


def extract_subject_colored(
    image_bgr: np.ndarray,
    subject_mask: np.ndarray,
    padding: int = 10,
) -> np.ndarray | None:
    """从原图中抠出主体人物，返回带透明通道的 RGBA 图像 (BGR + Alpha)。

    用于 LLM 精修：把抠出的人物发给 LLM 让它修正边缘。
    如果 mask 为空则返回 None。
    """
    h, w = image_bgr.shape[:2]
    mask_u8 = (subject_mask > 0).astype(np.uint8)
    if mask_u8.shape[:2] != (h, w):
        mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_LINEAR)

    if not mask_u8.any():
        return None

    # 找到 mask 的 bounding box，留 padding
    ys, xs = np.where(mask_u8)
    x1 = max(0, int(xs.min()) - padding)
    x2 = min(w, int(xs.max()) + padding)
    y1 = max(0, int(ys.min()) - padding)
    y2 = min(h, int(ys.max()) + padding)

    # 抠出 BGR 区域
    roi_bgr = image_bgr[y1:y2, x1:x2].copy()
    roi_alpha = mask_u8[y1:y2, x1:x2] * 255

    # 合成 BGRA
    roi_bgra = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2BGRA)
    roi_bgra[:, :, 3] = roi_alpha

    return roi_bgra


def refine_subject_with_llm(
    image_bgr: np.ndarray,
    subject_mask: np.ndarray,
    config: LLMSelectionConfig,
) -> np.ndarray | None:
    """将抠出的主体人物发给 LLM 做轮廓精修，提取精修后的 mask。

    流程：
      1. 从原图抠出主体（带 alpha 通道的 RGBA）
      2. 编码为 PNG data URL，发送给 Qwen/DashScope API
      3. 从返回图中提取新的 alpha 通道
      4. 失败返回 None，调用方 fallback 到 GrabCut 或原始 mask

    注意：此函数会消耗 API 调用次数，且比本地精修慢很多。
    """
    if not subject_mask.any():
        return None

    # 1) 抠出主体 RGBA
    subject_bgra = extract_subject_colored(image_bgr, subject_mask)
    if subject_bgra is None:
        return None

    # 2) 编码为 PNG data URL（保留 alpha 通道）
    import base64
    success, buffer = cv2.imencode(".png", subject_bgra)
    if not success:
        return None
    subject_data_url = "data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")

    # 3) 发送给 LLM
    try:
        if _is_dashscope(config):
            result_bgra = _call_dashscope_for_refine(subject_data_url, config)
        else:
            result_bgra = _call_openai_compatible_for_refine(subject_data_url, config)
    except Exception:
        return None

    if result_bgra is None:
        return None

    # 4) 从返回图提取 alpha → 构建精化 mask
    if result_bgra.shape[2] == 4:
        # LLM 返回了带 alpha 的图，直接用
        refined_alpha = result_bgra[:, :, 3]
        refined_mask = (refined_alpha > 128).astype(np.uint8)
    else:
        # LLM 没保留透明通道，用差分法估算 mask
        # 把返回的 BGR 与原始背景做差异比较
        gray_result = cv2.cvtColor(result_bgra, cv2.COLOR_BGR2GRAY)
        # 用一个低阈值做前景检测（LLM 修饰后的人物与纯背景不同）
        _, refined_mask = cv2.threshold(gray_result, 10, 1, cv2.THRESH_BINARY)

    h, w = image_bgr.shape[:2]
    if refined_mask.shape[:2] != (h, w):
        refined_mask = cv2.resize(refined_mask, (w, h), interpolation=cv2.INTER_LINEAR)

    return refined_mask.astype(np.uint8)


def _call_dashscope_for_refine(subject_data_url: str, config: LLMSelectionConfig) -> np.ndarray | None:
    """DashScope 专用：发送人物抠图请求精修。"""
    payload = {
        "model": "qwen-image-edit",
        "input": {
            "messages": [{
                "role": "user",
                "content": [
                    {"image": subject_data_url},
                    {"text": QWEN_REFINE_EDGES_PROMPT},
                ],
            }]
        },
    }
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    import json
    from urllib import error as urllib_error
    from urllib import request as urllib_request

    req = urllib_request.Request(
        url=DASHSCOPE_IMAGE_EDIT_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=config.timeout_seconds) as resp:
            response_text = resp.read().decode("utf-8")
    except Exception:
        return None

    response_json = json.loads(response_text)
    choices = (response_json.get("output", {}).get("choices", [])
               or response_json.get("choices", []))
    if not choices:
        return None

    content = choices[0].get("message", {}).get("content", "")
    images = _extract_images_from_content_list(content) if isinstance(content, list) else []
    if images:
        return images[-1]
    return None


def _call_openai_compatible_for_refine(subject_data_url: str, config: LLMSelectionConfig) -> np.ndarray | None:
    """OpenAI 兼容接口：发送人物抠图请求精修。"""
    payload = {
        "model": config.model,
        "temperature": 0,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": QWEN_REFINE_EDGES_PROMPT},
                {"type": "image_url", "image_url": {"url": subject_data_url}},
            ],
        }],
    }
    url = config.base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    import json
    from urllib import error as urllib_error
    from urllib import request as urllib_request

    req = urllib_request.Request(
        url=url, data=json.dumps(payload).encode("utf-8"),
        headers=headers, method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=config.timeout_seconds) as resp:
            response_text = resp.read().decode("utf-8")
    except Exception:
        return None

    response_json = json.loads(response_text)
    choices = response_json.get("choices", [])
    if not choices:
        return None

    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        images = _extract_images_from_content_list(content)
        if images:
            return images[-1]
    return None


def composite_subjects_back(
    background_bgr: np.ndarray,
    original_bgr: np.ndarray,
    subject_mask: np.ndarray,
    feather_radius: int = DEFAULT_FEATHER_RADIUS,
) -> np.ndarray:
    """将主体人物从原图提取，羽化边缘后贴回干净背景。

    改进点：
      1. mask 先 upscale 到全图分辨率再做处理
      2. 用 guided filter 以原图为引导做边缘保持的 alpha 羽化
    """
    if subject_mask is None or not subject_mask.any():
        return background_bgr.copy()

    h, w = background_bgr.shape[:2]
    if original_bgr.shape[:2] != (h, w):
        original_bgr = cv2.resize(original_bgr, (w, h))

    # 1) 粗 mask → 全分辨率 uint8
    mask_u8 = (subject_mask > 0).astype(np.uint8)
    if mask_u8.shape[:2] != (h, w):
        mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_LINEAR)

    # 2) 轻微膨胀，只在全分辨率下扩 2-3px
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(mask_u8, kernel, iterations=1)

    # 3) 以原图灰度为引导，做边缘保持的 alpha 羽化
    #    guided filter 会让 alpha 过渡沿着原图的真实边缘走
    guidance = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    alpha_input = dilated.astype(np.float32)  # mask 值本身就是 0/1，不用除 255

    soft_alpha = cv2.ximgproc.guidedFilter(
        guide=guidance,
        src=alpha_input,
        radius=feather_radius,       # 羽化半径（全分辨率像素）
        eps=1e-3,                    # 正则化，稍大避免过度贴合噪声
    )
    soft_alpha = np.clip(soft_alpha, 0.0, 1.0)

    # 4) 3 通道 alpha 混合
    alpha_3ch = np.dstack([soft_alpha] * 3)
    result = original_bgr.astype(np.float32) * alpha_3ch + background_bgr.astype(np.float32) * (1.0 - alpha_3ch)
    return result.astype(np.uint8)


# ---------------------------------------------------------------------------
# 主分析管线
# ---------------------------------------------------------------------------

def analyze_image_qwen(
    image_path: str | Path,
    model_path: str = "yolov8s-seg.pt",
    subject_score_ratio: float = 0.75,
    min_area_ratio: float = 0.01,
    llm_config: LLMSelectionConfig | None = None,
) -> ProcessResult:
    """新管线：YOLO → 主体识别 → Qwen/LaMa 全图消人 → 主体回贴。"""
    init_db()
    start_time = time.time()
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")
    if not is_supported_image(image_path):
        raise ValueError(f"不支持的图片格式: {image_path.suffix}")

    original_bgr = cv2.imread(str(image_path))
    if original_bgr is None:
        raise ValueError(f"无法读取图片: {image_path}")

    if llm_config is None:
        llm_config = LLMSelectionConfig()

    # ---- 1. YOLO 检测 ----
    model = load_model(model_path)
    results = model(str(image_path), classes=[0], conf=0.02, verbose=False)
    res = results[0]
    image_h, image_w = res.orig_shape

    boxes = res.boxes
    xyxy = boxes.xyxy.cpu().numpy() if boxes is not None else np.empty((0, 4), dtype=np.float32)
    conf = boxes.conf.cpu().numpy() if boxes is not None else np.empty((0,), dtype=np.float32)
    masks = res.masks.data.cpu().numpy() if res.masks is not None else np.array([])

    # ---- 2. 多尺度检测（同原逻辑） ----
    if len(xyxy) < 50 and image_h * image_w > 50000:
        try:
            original_count = len(xyxy)
            large_h, large_w = int(image_h * 2), int(image_w * 2)
            large_image = cv2.resize(original_bgr, (large_w, large_h))
            large_results = model(large_image, classes=[0], conf=0.15, verbose=False)
            if large_results and large_results[0].masks is not None:
                large_boxes = large_results[0].boxes
                large_xyxy = large_boxes.xyxy.cpu().numpy()
                large_conf = large_boxes.conf.cpu().numpy()
                large_masks = large_results[0].masks.data.cpu().numpy()

                scale_x = image_w / large_w
                scale_y = image_h / large_h
                large_xyxy_scaled = large_xyxy.copy()
                large_xyxy_scaled[:, [0, 2]] *= scale_x
                large_xyxy_scaled[:, [1, 3]] *= scale_y

                resized_masks = np.array([
                    cv2.resize(m.astype(np.float32), (image_w, image_h))
                    for m in large_masks
                ])

                # IoU 去重
                from photo_cleaner_core import compute_iou
                iou_threshold = 0.1
                new_indices = []
                for i, new_box in enumerate(large_xyxy_scaled):
                    duplicate = False
                    for old_box in xyxy[:original_count]:
                        if compute_iou(new_box, old_box) > iou_threshold:
                            duplicate = True
                            break
                    if not duplicate:
                        new_indices.append(i)

                if new_indices:
                    xyxy = np.concatenate([xyxy, large_xyxy_scaled[new_indices]], axis=0)
                    conf = np.concatenate([conf, large_conf[new_indices]], axis=0)
                    masks = np.concatenate([masks, resized_masks[new_indices]], axis=0)
        except Exception:
            pass

    # ---- SAM 高精度 mask 精化（可选，边界贴合远优于 YOLO 原生 mask） ----
    _sam_log = ""
    if getattr(llm_config, 'enable_sam', False) and len(xyxy) > 0:
        try:
            sam_model_name = getattr(llm_config, 'sam_model', 'mobile_sam.pt')
            sam_masks = generate_sam_masks(
                original_bgr, xyxy, image_h, image_w,
                model_name=sam_model_name,
            )
            if sam_masks is not None and len(sam_masks) == len(xyxy):
                # 自适应边界精化：大目标 GrabCut + 中目标比例腐蚀
                masks = refine_sam_masks_adaptive(original_bgr, sam_masks, xyxy)
                _sam_log = " + SAM 高精度 mask + 边界精化"
            else:
                _sam_log = " (SAM 失败，降级 YOLO mask)"
        except Exception as exc:
            _sam_log = f" (SAM 异常:{type(exc).__name__}, 降级 YOLO mask)"

    # ---- 3. 无检测 ----
    if len(xyxy) == 0:
        elapsed = time.time() - start_time
        return ProcessResult(
            input_path=str(image_path), output_path="", subject_count=0, stray_count=0,
            elapsed_seconds=elapsed, original_bgr=original_bgr, cleaned_bgr=original_bgr.copy(),
            subject_mask=np.zeros((image_h, image_w), dtype=bool),
            stray_mask=np.zeros((image_h, image_w), dtype=bool),
            detections=[], logs=["未检测到人物，已原样输出。"],
        )

    # ---- 4. 主体识别 ----
    detections = [{
        "index": i, "box": xyxy[i].tolist(), "conf": float(conf[i]),
        "area": float((xyxy[i, 2] - xyxy[i, 0]) * (xyxy[i, 3] - xyxy[i, 1])),
    } for i in range(len(xyxy))]

    try:
        subject_indices, selector_logs = select_subject_indices(
            image_bgr=original_bgr, detections=detections,
            masks=masks if masks.size > 0 else None,
            llm_config=llm_config,
        )
    except Exception as exc:
        selector_logs = [f"主体识别失败 ({exc})，回退为消除所有人物。"]
        all_indices = np.arange(len(xyxy))
        all_people_mask = build_union_mask(masks, all_indices)
        subject_mask = np.zeros((image_h, image_w), dtype=bool)
        stray_mask = all_people_mask.copy()
        removal_method = "lama"
        try:
            background_bgr, removal_method = remove_all_people_with_fallback(
                original_bgr, all_people_mask, llm_config)
        except Exception:
            background_bgr = original_bgr.copy()
            removal_method = "failed"
            selector_logs.append("人物消除失败，已保留原图。")

        elapsed = time.time() - start_time
        return ProcessResult(
            input_path=str(image_path), output_path="", subject_count=0,
            stray_count=len(xyxy), elapsed_seconds=elapsed,
            original_bgr=original_bgr, cleaned_bgr=background_bgr,
            subject_mask=subject_mask, stray_mask=stray_mask,
            detections=[DetectionSummary(
                index=i, box=xyxy[i].tolist(), conf=float(conf[i]), score=0.0,
                area=float((xyxy[i, 2] - xyxy[i, 0]) * (xyxy[i, 3] - xyxy[i, 1])),
                label="background",
            ) for i in range(len(xyxy))],
            logs=selector_logs + [
                f"检测到 {len(xyxy)} 人 | 全部消除 | 消除方式: {removal_method} | 耗时 {elapsed:.2f}s"
            ],
        )

    # ---- 5. 构建 mask ----
    all_indices = np.arange(len(xyxy))
    subject_index_set = set(subject_indices.tolist())
    stray_indices = np.setdiff1d(all_indices, subject_indices)

    all_people_mask = build_union_mask(masks, all_indices)
    subject_mask = build_union_mask(masks, subject_indices)
    stray_mask = build_union_mask(masks, stray_indices) if len(stray_indices) > 0 else np.zeros((image_h, image_w), dtype=bool)

    # ---- 6. 全图人物消除 ----
    removal_method = "lama"
    try:
        background_bgr, removal_method = remove_all_people_with_fallback(
            original_bgr, all_people_mask, llm_config)
    except Exception:
        background_bgr = original_bgr.copy()
        removal_method = "failed"
        selector_logs.append("人物消除失败，已保留原图。")

    # ---- 7. 主体 mask 精化 ----
    refine_method = "none"
    if subject_mask.any():
        # 7a. GrabCut 边缘精化（实验性，需显式开启）
        if getattr(llm_config, "enable_grabcut_refine", False):
            try:
                refined = refine_subject_mask(original_bgr, subject_mask)
                if refined.any():
                    subject_mask = refined
                    refine_method = "grabcut"
                    selector_logs.append("[refine] GrabCut 边缘精化完成")
                else:
                    selector_logs.append("[refine] GrabCut 返回空 mask，使用原始 mask")
            except Exception as exc:
                selector_logs.append(f"[refine] GrabCut 失败 ({exc})，使用原始 mask")

        # 7b. LLM 轮廓精修（可选，需显式开启）
        if getattr(llm_config, "enable_llm_refine", False) and llm_config.api_key:
            try:
                llm_mask = refine_subject_with_llm(original_bgr, subject_mask, llm_config)
                if llm_mask is not None and llm_mask.any():
                    subject_mask = llm_mask
                    refine_method = "llm"
                    selector_logs.append("[refine] LLM 轮廓精修完成")
                else:
                    selector_logs.append("[refine] LLM 未返回有效 mask，跳过")
            except Exception as exc:
                selector_logs.append(f"[refine] LLM 精修失败 ({exc})，跳过")

    # ---- 8. 主体回贴 ----
    if subject_mask.any():
        cleaned_bgr = composite_subjects_back(
            background_bgr=background_bgr,
            original_bgr=original_bgr,
            subject_mask=subject_mask,
        )
    else:
        cleaned_bgr = background_bgr

    # ---- 9. 构建结果 ----
    elapsed = time.time() - start_time
    subject_count = len(subject_indices)
    stray_count = len(stray_indices)

    detections_summary = [
        DetectionSummary(
            index=i, box=xyxy[i].tolist(), conf=float(conf[i]),
            score=1.0 if i in subject_index_set else 0.0,
            area=float((xyxy[i, 2] - xyxy[i, 0]) * (xyxy[i, 3] - xyxy[i, 1])),
            label="subject" if i in subject_index_set else "stray",
        )
        for i in range(len(xyxy))
    ]

    logs = selector_logs + [
        f"检测到 {len(xyxy)} 人 | 主体 {subject_count} 人保留 | 路人 {stray_count} 人消除{_sam_log}",
        f"消除方式: {removal_method} | 精化方式: {refine_method} | 耗时 {elapsed:.2f}s",
    ]

    return ProcessResult(
        input_path=str(image_path), output_path="",
        subject_count=subject_count, stray_count=stray_count,
        elapsed_seconds=elapsed, original_bgr=original_bgr, cleaned_bgr=cleaned_bgr,
        subject_mask=subject_mask, stray_mask=stray_mask,
        detections=detections_summary, logs=logs,
    )


# ---------------------------------------------------------------------------
# 封装函数
# ---------------------------------------------------------------------------

def process_image_qwen(
    image_path: str | Path,
    output_dir: str | Path,
    model_path: str = "yolov8s-seg.pt",
    llm_config: LLMSelectionConfig | None = None,
    save_masks: bool = True,
) -> tuple[ProcessResult, dict[str, Path]]:
    """处理单张图片（Qwen 管线），保存结果并写入数据库。"""
    start_time = time.time()
    try:
        result = analyze_image_qwen(
            image_path=image_path, model_path=model_path, llm_config=llm_config,
        )
        saved_paths = save_result(result, output_dir=output_dir, save_masks=save_masks)
        try:
            insert_record(
                input_path=result.input_path, output_path=str(saved_paths["output"]),
                subject_count=result.subject_count, stray_count=result.stray_count,
                status="success", error_message="", elapsed=result.elapsed_seconds,
            )
        except Exception:
            pass
        return result, saved_paths
    except Exception as exc:
        try:
            insert_record(
                input_path=str(image_path), output_path="",
                subject_count=0, stray_count=0, status="failed",
                error_message=str(exc), elapsed=time.time() - start_time,
            )
        except Exception:
            pass
        raise


def process_batch_qwen(
    image_paths: Iterable[str | Path],
    output_dir: str | Path,
    model_path: str = "yolov8s-seg.pt",
    llm_config: LLMSelectionConfig | None = None,
    save_masks: bool = True,
) -> list[tuple[Path, ProcessResult | None, str | None]]:
    """批量处理（Qwen 管线）。"""
    results: list[tuple[Path, ProcessResult | None, str | None]] = []
    for image_path in image_paths:
        path = Path(image_path)
        try:
            result, _ = process_image_qwen(
                image_path=path, output_dir=output_dir, model_path=model_path,
                llm_config=llm_config, save_masks=save_masks,
            )
            results.append((path, result, None))
        except Exception as exc:
            results.append((path, None, str(exc)))
    return results


def run_cli_qwen(
    input_path: str,
    output_dir: str,
    model_path: str = "yolov8s-seg.pt",
    llm_config: LLMSelectionConfig | None = None,
) -> None:
    """CLI 入口（Qwen 管线）。"""
    path = Path(input_path)
    if path.is_dir():
        image_paths = collect_images(path)
        batch_results = process_batch_qwen(
            image_paths, output_dir=output_dir, model_path=model_path, llm_config=llm_config,
        )
        print(f"共处理 {len(batch_results)} 张图片")
        for item_path, result, error in batch_results:
            if result is None:
                print(f"失败: {item_path} -> {error}")
                continue
            print(f"成功: {item_path} -> {result.output_path}")
    else:
        result, saved_paths = process_image_qwen(
            input_path, output_dir=output_dir, model_path=model_path, llm_config=llm_config,
        )
        print(f"处理完成: {input_path}")
        print(f"输出路径: {saved_paths['output']}")
        print(f"主体数量: {result.subject_count} | 路人数量: {result.stray_count}")
        for line in format_detection_lines(result.detections):
            print(line)
