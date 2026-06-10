from __future__ import annotations

# 强制使用 CPU，必须在导入 torch 相关库之前设置
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import time
from importlib import import_module
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

# 在导入 ultralytics 之前，先强制 torch 使用 CPU
import torch
torch.cuda.is_available = lambda: False  # 强制覆盖，让所有 GPU 检查返回 False

# 补丁 torch.jit.load，强制使用 CPU
_original_jit_load = torch.jit.load
def _patched_jit_load(*args, **kwargs):
    kwargs.setdefault('map_location', 'cpu')
    return _original_jit_load(*args, **kwargs)
torch.jit.load = _patched_jit_load

from ultralytics import YOLO

from db import init_db, insert_record
from llm_subject_selector import LLMSelectionConfig, compute_iou, select_subject_indices

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass
class DetectionSummary:
    index: int
    box: list[float]
    conf: float
    score: float
    area: float
    label: str


@dataclass
class ProcessResult:
    input_path: str
    output_path: str
    subject_count: int
    stray_count: int
    elapsed_seconds: float
    original_bgr: np.ndarray
    cleaned_bgr: np.ndarray
    subject_mask: np.ndarray
    stray_mask: np.ndarray
    detections: list[DetectionSummary]
    logs: list[str]
    status: str = "success"
    error_message: str = ""


def is_supported_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def build_union_mask(masks: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if masks.size == 0 or len(indices) == 0:
        return np.zeros((0, 0), dtype=bool)
    
    # 过滤掉超出 masks 范围的索引
    valid_mask_count = masks.shape[0]
    indices = indices[indices < valid_mask_count]

    union_mask = np.zeros(masks.shape[1:], dtype=bool)
    for idx in indices:
        union_mask |= masks[idx] > 0.5
    return union_mask


@lru_cache(maxsize=1)
def load_lama_model():
    try:
        simple_lama_module = import_module("simple_lama_inpainting")
    except ImportError as exc:
        raise ImportError(
            "LaMa 后端不可用，请先安装 simple-lama-inpainting"
        ) from exc

    return simple_lama_module.SimpleLama()


def remove_stray_people_lama(image_bgr: np.ndarray, stray_mask: np.ndarray) -> np.ndarray:
    if image_bgr.size == 0:
        return image_bgr

    if stray_mask.size == 0:
        return image_bgr.copy()

    image_h, image_w = image_bgr.shape[:2]
    if stray_mask.shape[:2] != (image_h, image_w):
        stray_mask = cv2.resize(
            stray_mask.astype(np.float32),
            (image_w, image_h),
            interpolation=cv2.INTER_NEAREST,
        )

    mask_u8 = (stray_mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(mask_u8) == 0:
        return image_bgr.copy()

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    mask_pil = Image.fromarray(mask_u8, mode="L")

    lama = load_lama_model()
    result_pil = lama(image_pil, mask_pil)
    result_rgb = np.array(result_pil.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)


def remove_stray_people(
    image_bgr: np.ndarray,
    stray_mask: np.ndarray,
) -> np.ndarray:
    if image_bgr.size == 0:
        return image_bgr

    if stray_mask.size == 0:
        return image_bgr.copy()

    image_h, image_w = image_bgr.shape[:2]
    if stray_mask.shape[:2] != (image_h, image_w):
        stray_mask = cv2.resize(
            stray_mask.astype(np.float32),
            (image_w, image_h),
            interpolation=cv2.INTER_NEAREST,
        )

    mask_u8 = (stray_mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(mask_u8) == 0:
        return image_bgr.copy()

    return remove_stray_people_lama(image_bgr, stray_mask)


def remove_all_people_twice(image_bgr: np.ndarray, all_people_mask: np.ndarray) -> np.ndarray:
    """对全部人物区域做 LaMa 双次修复，返回干净的背景图。"""
    if all_people_mask.size == 0 or not all_people_mask.any():
        return image_bgr.copy()
    mask_u8 = (all_people_mask > 0).astype(np.uint8) * 255
    first = remove_stray_people_lama(image_bgr, mask_u8)
    return remove_stray_people_lama(first, mask_u8)


def composite_subjects_back(
    background_bgr: np.ndarray,
    original_bgr: np.ndarray,
    subject_mask: np.ndarray,
    feather_radius: int = 5,
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



@lru_cache(maxsize=1)
def load_model(model_path: str = "yolov8s-seg.pt") -> YOLO:
    return YOLO(model_path)


# ---------------------------------------------------------------------------
# SAM (Segment Anything Model) 高精度 mask 生成
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_sam_model(model_name: str = "mobile_sam.pt"):
    """加载 SAM 模型用于高精度分割 mask。进程内只加载一次，CPU 模式。

    支持传入文件名（自动查 models/ 目录）或完整路径。
    支持的模型（按大小排序）：
      - mobile_sam.pt     (~38 MB, 推荐，CPU 可用)
      - sam2_t.pt         (~38 MB, SAM 2 tiny)
      - FastSAM-s.pt      (~12 MB, 基于 YOLOv8，最快但精度略低)
      - sam_b.pt          (~358 MB, SAM base)
      - sam2_b.pt         (~80 MB, SAM 2 base)
      - sam2_s.pt         (~46 MB, SAM 2 small)

    失败返回 None，调用方需做降级回退到 YOLO mask。
    """
    try:
        from ultralytics import SAM

        # 注释：优先使用项目 models/ 目录的本地模型（纯离线），
        #       没有的话再走 ultralytics 自动下载逻辑。
        _models_dir = Path(__file__).resolve().parent / "models"
        _local_path = _models_dir / model_name
        if _local_path.exists():
            model_path = str(_local_path)
            print(f"[sam] 使用本地模型: {model_path}")
        else:
            model_path = model_name

        print(f"[sam] 正在加载 {model_name} 到 CPU……")
        model = SAM(model_path)
        # 强制 CPU
        model.to("cpu")
        print(f"[sam] {model_name} 加载完成。")
        return model
    except Exception as exc:
        print(f"[sam] SAM 模型加载失败: {type(exc).__name__}: {exc}")
        print("[sam] 将回退到 YOLOv8-seg 原生 mask。")
        return None


def generate_sam_masks(
    image_bgr: np.ndarray,
    xyxy_boxes: np.ndarray,
    image_h: int,
    image_w: int,
    model_name: str = "mobile_sam.pt",
) -> np.ndarray | None:
    """使用 SAM 根据 YOLO 检测框生成高精度分割 mask。

    Args:
        image_bgr: 输入图像 (H, W, 3)，BGR 格式
        xyxy_boxes: (N, 4) 检测框数组，坐标在原图空间
        image_h, image_w: 原图高度和宽度
        model_name: SAM 模型文件名

    Returns:
        masks: (N, H, W) float32 概率 mask 数组，失败时返回 None
    """
    sam = load_sam_model(model_name)
    if sam is None:
        return None

    if len(xyxy_boxes) == 0:
        return np.array([])

    try:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # 注释：SAM 能一次处理多个 bbox，但过多会 OOM；
        #       上限 40 个 / 批，一般照片人物数远少于此。
        MAX_PER_CALL = 40
        all_masks: list[np.ndarray] = []

        boxes_list: list[list[float]] = xyxy_boxes.tolist()

        for start in range(0, len(boxes_list), MAX_PER_CALL):
            batch = boxes_list[start:start + MAX_PER_CALL]
            results = sam.predict(
                image_rgb,
                bboxes=batch,
                verbose=False,
                save=False,
                show=False,
            )

            if not results or not hasattr(results[0], "masks") or results[0].masks is None:
                print(f"[sam] 批次 {start // MAX_PER_CALL + 1} 无 mask 输出，回退 YOLO mask。")
                return None

            batch_masks = results[0].masks.data.cpu().numpy()

            # 注释：SAM 输出的 mask 尺寸可能不同于输入尺寸，统一缩放到原图
            if batch_masks.shape[1:] != (image_h, image_w):
                batch_masks = np.array([
                    cv2.resize(m.astype(np.float32), (image_w, image_h))
                    for m in batch_masks
                ])

            all_masks.append(batch_masks)

        if not all_masks:
            return None

        result = np.concatenate(all_masks, axis=0)

        # 注释：安全校验 — SAM 返回的 mask 数量必须与输入 bbox 数量一致
        if len(result) != len(xyxy_boxes):
            print(f"[sam] mask 数量不匹配: 期望 {len(xyxy_boxes)}, 实际 {len(result)}, 回退 YOLO。")
            return None

        print(f"[sam] 高精度 mask 生成完成，{len(result)} 个实例。")
        return result

    except Exception as exc:
        print(f"[sam] mask 生成异常: {type(exc).__name__}: {exc}，回退 YOLO mask。")
        return None


def refine_single_mask_with_grabcut(
    roi_bgr: np.ndarray,
    roi_mask: np.ndarray,
    iterations: int = 5,
) -> np.ndarray:
    """对单个人物 ROI 做 GrabCut 边界精化。

    SAM mask 虽锐利但过分割（bbox 内的背景像素也被判为前景）。
    GrabCut 利用图像颜色/纹理找到真实边界，把 mask 向内收缩到人物轮廓。

    Args:
        roi_bgr: 人物 ROI 区域 (H, W, 3) BGR
        roi_mask: SAM 输出的二值 mask (H, W) uint8 0/1
        iterations: GrabCut 迭代次数

    Returns:
        精化后的二值 mask (H, W) uint8 0/1
    """
    h, w = roi_bgr.shape[:2]
    if h < 10 or w < 10:
        return roi_mask

    original_area = int(roi_mask.sum())
    if original_area < 100:
        return roi_mask  # 太小，GrabCut 必然跑飞

    # 确定前景 = SAM mask 腐蚀后仍然保留的核心区域
    erode_kernel_size = max(3, int(min(h, w) * 0.03))  # 自适应：大目标多腐蚀
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_kernel_size, erode_kernel_size))
    definite_fg = cv2.erode(roi_mask, kernel, iterations=1)
    if not definite_fg.any():
        return roi_mask

    # 确定背景 = SAM mask 膨胀后之外的部分
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated = cv2.dilate(roi_mask, dilate_kernel, iterations=2)

    # 构建 GrabCut mask
    gc_mask = np.zeros((h, w), dtype=np.uint8)
    gc_mask[definite_fg > 0] = cv2.GC_FGD
    gc_mask[(roi_mask > 0) & (definite_fg == 0)] = cv2.GC_PR_FGD  # SAM 的边缘区域 = 可能前景
    gc_mask[dilated == 0] = cv2.GC_BGD                              # 远离 SAM mask = 确定背景

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
    refined_area = int(refined.sum())

    # 安全校验：GrabCut 跑飞了就退回原始 mask
    if refined_area < original_area * 0.15 or refined_area > original_area * 3:
        return roi_mask  # 退回

    return refined


def refine_sam_masks_adaptive(
    image_bgr: np.ndarray,
    sam_masks: np.ndarray,
    xyxy_boxes: np.ndarray,
) -> np.ndarray:
    """对 SAM 生成的 mask 做自适应边界精化。

    策略：
      - 大目标 (>2000px, 约 45x45): GrabCut 颜色感知精化
      - 中目标 (500-2000px): 比例腐蚀 (mask 面积的 2%)
      - 小目标 (<500px): 不做处理（精化风险 > 收益）
      - **与他人交叠的目标**: 跳过 GrabCut（避免交界处产生裂隙）

    Args:
        image_bgr: 原图 (H, W, 3) BGR
        sam_masks: SAM 输出的 mask (N, H, W) 0/1 或 float
        xyxy_boxes: 检测框 (N, 4)

    Returns:
        精化后的 mask (N, H, W) uint8 0/1
    """
    h, w = image_bgr.shape[:2]
    n = len(sam_masks)
    refined = np.zeros((n, h, w), dtype=np.uint8)

    # ---- 预检：标记哪些大目标跟其他人 mask 有交叠 ----
    #       只检测 GrabCut 候选（area >= 2000），小目标不受影响
    has_overlap = np.zeros(n, dtype=bool)

    # 先用 bbox IoU 快速过滤，再对疑似交叠的做像素级 mask overlap
    grabcut_candidates = [i for i in range(n) if (sam_masks[i] > 0.5).sum() >= 2000]
    for idx_i, i in enumerate(grabcut_candidates):
        mask_i = (sam_masks[i] > 0.5).astype(np.uint8)
        area_i = int(mask_i.sum())
        box_i = xyxy_boxes[i]
        for j in grabcut_candidates[idx_i + 1:]:
            # 快速过滤：bbox 不相交则 mask 也不可能交叠
            if (box_i[2] < xyxy_boxes[j][0] or xyxy_boxes[j][2] < box_i[0] or
                box_i[3] < xyxy_boxes[j][1] or xyxy_boxes[j][3] < box_i[1]):
                continue

            mask_j = (sam_masks[j] > 0.5).astype(np.uint8)
            area_j = int(mask_j.sum())
            overlap = int((mask_i & mask_j).sum())
            min_area = min(area_i, area_j)
            if overlap > min_area * 0.05:
                has_overlap[i] = True
                has_overlap[j] = True

    for i in range(n):
        mask = (sam_masks[i] > 0.5).astype(np.uint8)
        area = int(mask.sum())

        if area >= 2000 and not has_overlap[i]:
            # 大目标且不跟别人交叠 → GrabCut
            box = xyxy_boxes[i].astype(int)
            x1 = max(0, box[0] - 5)
            y1 = max(0, box[1] - 5)
            x2 = min(w, box[2] + 5)
            y2 = min(h, box[3] + 5)

            roi_bgr = image_bgr[y1:y2, x1:x2]
            roi_mask = mask[y1:y2, x1:x2]
            refined_roi = refine_single_mask_with_grabcut(roi_bgr, roi_mask)

            if refined_roi.any():
                refined[i, y1:y2, x1:x2] = np.maximum(
                    refined[i, y1:y2, x1:x2], refined_roi
                )
            else:
                refined[i] = mask
                print(f"[refine] 人物#{i} GrabCut 返回空，保留 SAM 原始 mask")

        elif area >= 2000 and has_overlap[i]:
            # 大目标但跟别人挨着 → 跳过 GrabCut，保留 SAM 原始 mask
            # （SAM 过分割反而填充了人物间的间隙，避免裂隙）
            refined[i] = mask

        elif area >= 500:
            # 中目标 → 比例腐蚀
            erode_px = max(1, int(np.sqrt(area) * 0.015))  # 约 1.5% 线性尺寸
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
            eroded = cv2.erode(mask, kernel, iterations=1)
            if eroded.sum() > area * 0.5:  # 安全校验：不能蚀掉一半以上
                refined[i] = eroded
            else:
                refined[i] = mask
                print(f"[refine] 人物#{i} 腐蚀过度，保留原始 mask")
        else:
            # 小目标 → 不做处理
            refined[i] = mask

    if n > 0:
        area_per_mask = np.array([(sam_masks[j] > 0.5).sum() for j in range(n)])
        grabcut_count = int(((area_per_mask >= 2000) & ~has_overlap).sum())
        skipped_overlap = int(((area_per_mask >= 2000) & has_overlap).sum())
        eroded_count = int(((area_per_mask >= 500) & (area_per_mask < 2000)).sum())
        skipped_small = n - grabcut_count - skipped_overlap - eroded_count
        parts = []
        if grabcut_count:
            parts.append(f"{grabcut_count} GrabCut")
        if skipped_overlap:
            parts.append(f"{skipped_overlap} 跳过(交叠)")
        if eroded_count:
            parts.append(f"{eroded_count} 腐蚀")
        if skipped_small:
            parts.append(f"{skipped_small} 跳过(太小)")
        if parts:
            print(f"[refine] 边界精化完成: {', '.join(parts)}")

    return refined


def analyze_image(
    image_path: str | Path,
    model_path: str = "yolov8s-seg.pt",
    subject_score_ratio: float = 0.75,
    min_area_ratio: float = 0.01,#降低最小面积阈值 - 检测更小的远处人物
    llm_config: LLMSelectionConfig | None = None,
) -> ProcessResult:
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

    model = load_model(model_path)
    results = model(str(image_path), classes=[0],conf=0.02,verbose=False)#conf降低置信度阈值 - 检出更多模糊人物

    if not results:
        raise RuntimeError("模型没有返回任何结果")

    res = results[0]
    image_h, image_w = res.orig_shape
    boxes = res.boxes
    xyxy = boxes.xyxy.cpu().numpy() if boxes is not None else np.empty((0, 4), dtype=np.float32)
    conf = boxes.conf.cpu().numpy() if boxes is not None else np.empty((0,), dtype=np.float32)
    masks = res.masks.data.cpu().numpy() if res.masks is not None else np.array([])

    # 多尺度检测：放大图片检测小人物
    if len(xyxy) < 50 and image_h * image_w > 50000:
        try:
            large_h, large_w = int(image_h * 2), int(image_w * 2)
            large_image = cv2.resize(original_bgr, (large_w, large_h))
            large_results = model(large_image, classes=[0], conf=0.15, verbose=False)
            if large_results and large_results[0].masks is not None:
                large_boxes = large_results[0].boxes
                large_xyxy = large_boxes.xyxy.cpu().numpy()
                large_conf = large_boxes.conf.cpu().numpy()
                large_masks = large_results[0].masks.data.cpu().numpy()
                # 缩放坐标映射回原图尺寸
                scale_x = image_w / large_w
                scale_y = image_h / large_h
                large_xyxy_scaled = large_xyxy.copy()
                large_xyxy_scaled[:, [0, 2]] *= scale_x
                large_xyxy_scaled[:, [1, 3]] *= scale_y
                # 缩放 masks 回原图尺寸
                resized_masks = np.array([
                    cv2.resize(m.astype(np.float32), (image_w, image_h))
                    for m in large_masks
                ])
                # 去重：只保留多尺度中新发现的人（与原结果 IoU < 0.1）
                iou_threshold = 0.1
                new_mask_indices = []
                for i, new_box in enumerate(large_xyxy_scaled):
                    is_duplicate = False
                    for old_box in xyxy[:original_count]:
                        if compute_iou(new_box, old_box) > iou_threshold:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        new_mask_indices.append(i)

                # 用新索引从多尺度结果中提取
                if new_mask_indices:
                    new_xyxy = large_xyxy_scaled[new_mask_indices]
                    new_conf = large_conf[new_mask_indices]
                    new_masks = resized_masks[new_mask_indices]

                    xyxy = np.concatenate([xyxy, new_xyxy], axis=0)
                    conf = np.concatenate([conf, new_conf], axis=0)
                    masks = np.concatenate([masks, new_masks], axis=0)
        except Exception:
            pass  # 多尺度失败不影响主流程
            

    # ---- SAM 高精度 mask 精化（可选，边界贴合远优于 YOLO 原生 mask） ----
    _sam_log = ""
    if llm_config and getattr(llm_config, 'enable_sam', False) and len(xyxy) > 0:
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



    if len(xyxy) == 0:
        cleaned_image = original_bgr.copy()
        subject_mask = np.zeros((image_h, image_w), dtype=bool)
        stray_mask = np.zeros((image_h, image_w), dtype=bool)
        elapsed_seconds = time.time() - start_time
        result = ProcessResult(
            input_path=str(image_path),
            output_path="",
            subject_count=0,
            stray_count=0,
            elapsed_seconds=elapsed_seconds,
            original_bgr=original_bgr,
            cleaned_bgr=cleaned_image,
            subject_mask=subject_mask,
            stray_mask=stray_mask,
            detections=[],
            logs=["未检测到人物，已原样输出。"],
        )
        return result

    detections = [
        {
            "index": i,
            "box": xyxy[i].tolist(),
            "conf": float(conf[i]),
            "area": float((xyxy[i, 2] - xyxy[i, 0]) * (xyxy[i, 3] - xyxy[i, 1])),
        }
        for i in range(len(xyxy))
    ]

    subject_indices, selector_logs = select_subject_indices(
        image_bgr=original_bgr,
        detections=detections,
        masks=masks if masks is not None and masks.size > 0 else None,
        llm_config=llm_config,
    )

    all_indices = np.arange(len(xyxy))
    stray_indices = np.setdiff1d(all_indices, subject_indices)

    subject_mask = build_union_mask(masks, subject_indices)
    stray_mask = build_union_mask(masks, stray_indices)
    all_people_mask = build_union_mask(masks, all_indices)

    # 注释：全图消人（LaMa 双次修复）得到干净背景，再本地羽化回贴主体
    background_bgr = remove_all_people_twice(original_bgr, all_people_mask)
    cleaned_image = composite_subjects_back(background_bgr, original_bgr, subject_mask)

    elapsed_seconds = time.time() - start_time
    subject_index_set = set(subject_indices.tolist())
    detections_summary = [
        DetectionSummary(
            index=i,
            box=xyxy[i].tolist(),
            conf=float(conf[i]),
            score=1.0 if i in subject_index_set else 0.0,
            area=float((xyxy[i, 2] - xyxy[i, 0]) * (xyxy[i, 3] - xyxy[i, 1])),
            label="subject" if i in subject_indices else "stray",
        )
        for i in range(len(xyxy))
    ]

    logs = selector_logs + [f"检测到 {len(xyxy)} 个人物，主体 {len(subject_indices)} 个，路人 {len(stray_indices)} 个，已全图修复两次并回贴主体{_sam_log}。"]

    return ProcessResult(
        input_path=str(image_path),
        output_path="",
        subject_count=len(subject_indices),
        stray_count=len(stray_indices),
        elapsed_seconds=elapsed_seconds,
        original_bgr=original_bgr,
        cleaned_bgr=cleaned_image,
        subject_mask=subject_mask,
        stray_mask=stray_mask,
        detections=detections_summary,
        logs=logs,
    )


def save_result(
    result: ProcessResult,
    output_dir: str | Path,
    save_masks: bool = True,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(result.input_path)
    base_name = source_path.stem

    output_image_path = output_dir / f"{base_name}_removed.png"
    cv2.imwrite(str(output_image_path), result.cleaned_bgr)

    saved_paths: dict[str, Path] = {"output": output_image_path}

    if save_masks:
        subject_mask_path = output_dir / f"{base_name}_subject_mask.png"
        stray_mask_path = output_dir / f"{base_name}_stray_mask.png"
        if result.subject_mask.size > 0:
            cv2.imwrite(str(subject_mask_path), result.subject_mask.astype(np.uint8) * 255)
        if result.stray_mask.size > 0:
            cv2.imwrite(str(stray_mask_path), result.stray_mask.astype(np.uint8) * 255)
        saved_paths["subject_mask"] = subject_mask_path
        saved_paths["stray_mask"] = stray_mask_path

    result.output_path = str(output_image_path)
    return saved_paths


def process_image(
    image_path: str | Path,
    output_dir: str | Path,
    model_path: str = "yolov8s-seg.pt",
    llm_config: LLMSelectionConfig | None = None,
    save_masks: bool = True,
) -> tuple[ProcessResult, dict[str, Path]]:
    start_time = time.time()
    try:
        result = analyze_image(
            image_path=image_path,
            model_path=model_path,
            llm_config=llm_config,
        )
        saved_paths = save_result(result, output_dir=output_dir, save_masks=save_masks)

        try:
            insert_record(
                input_path=result.input_path,
                output_path=str(saved_paths["output"]),
                subject_count=result.subject_count,
                stray_count=result.stray_count,
                status="success",
                error_message="",
                elapsed=result.elapsed_seconds,
            )
        except Exception:
            pass

        return result, saved_paths
    except Exception as exc:
        try:
            insert_record(
                input_path=str(image_path),
                output_path="",
                subject_count=0,
                stray_count=0,
                status="failed",
                error_message=str(exc),
                elapsed=time.time() - start_time,
            )
        except Exception:
            pass
        raise


def collect_images(path: str | Path) -> list[Path]:
    path = Path(path)
    if not path.exists():
        return []
    if path.is_file():
        return [path] if is_supported_image(path) else []
    return [item for item in sorted(path.iterdir()) if item.is_file() and is_supported_image(item)]


def process_batch(
    image_paths: Iterable[str | Path],
    output_dir: str | Path,
    model_path: str = "yolov8s-seg.pt",
    llm_config: LLMSelectionConfig | None = None,
    save_masks: bool = True,
) -> list[tuple[Path, ProcessResult | None, str | None]]:
    results: list[tuple[Path, ProcessResult | None, str | None]] = []
    for image_path in image_paths:
        path = Path(image_path)
        try:
            result, _ = process_image(
                image_path=path,
                output_dir=output_dir,
                model_path=model_path,
                llm_config=llm_config,
                save_masks=save_masks,
            )
            results.append((path, result, None))
        except Exception as exc:
            results.append((path, None, str(exc)))
    return results


def format_detection_lines(detections: list[DetectionSummary]) -> list[str]:
    lines: list[str] = []
    for item in detections:
        lines.append(
            f"#{item.index} {item.label} | conf={item.conf:.3f} | keep={int(item.score > 0)} | area={item.area:.0f}"
        )
    return lines


def run_cli(
    input_path: str,
    output_dir: str,
    model_path: str = "yolov8s-seg.pt",
    llm_config: LLMSelectionConfig | None = None,
) -> None:
    path = Path(input_path)
    if path.is_dir():
        image_paths = collect_images(path)
        batch_results = process_batch(
            image_paths,
            output_dir=output_dir,
            model_path=model_path,
            llm_config=llm_config,
        )
        print(f"共处理 {len(batch_results)} 张图片")
        for item_path, result, error in batch_results:
            if result is None:
                print(f"失败: {item_path} -> {error}")
                continue
            print(f"成功: {item_path} -> {result.output_path}")
    else:
        result, saved_paths = process_image(
            input_path,
            output_dir=output_dir,
            model_path=model_path,
            llm_config=llm_config,
        )
        print(f"处理完成: {input_path}")
        print(f"输出路径: {saved_paths['output']}")
        print(f"主体数量: {result.subject_count} | 路人数量: {result.stray_count}")
        for line in format_detection_lines(result.detections):
            print(line)
