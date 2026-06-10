"""
llm_subject_selector.py
-----------------------
主体选择模块：决定图中哪些人物是"主体"、哪些是"路人"。

使用多维度特征融合（面积、中心、深度、人脸、完整度、置信度）进行规则打分，
由 MiDaS 单目深度估算和 BlazeFace 人脸检测提供辅助特征。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from subject_features import BoxFeatures, extract_features_for_detections


@dataclass
class LLMSelectionConfig:
    base_url: str = "https://dashscope.aliyuncs.com"
    model: str = "qwen-image-edit"
    api_key: str = ""
    timeout_seconds: int = 90
    # 注释：多维度特征开关，UI 侧边栏会动态绑定
    enable_depth: bool = True
    enable_face: bool = True
    # 注释：GrabCut 边缘精化（实验性，默认关闭，可能不稳定）
    enable_grabcut_refine: bool = False
    # 注释：LLM 精修主体轮廓（费钱、慢，默认关闭）
    enable_llm_refine: bool = False
    # 注释：SAM 高精度 mask（MobileSAM，CPU 较慢但边界极精准，默认关闭）
    enable_sam: bool = False
    sam_model: str = "mobile_sam.pt"

def image_to_data_url(image_bgr: np.ndarray) -> str:
    """将 BGR 图像编码为 data:image/png;base64,... 格式的 data URL。"""
    import base64
    import cv2
    success, buffer = cv2.imencode(".png", image_bgr)
    if not success:
        raise ValueError("无法编码图片")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def compute_iou(box_a, box_b) -> float:
    x1 = max(box_a[0], box_b[0]); y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2]); y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
    b = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
    return inter / (a + b - inter) if (a + b - inter) > 0 else 0


# ---------------------------------------------------------------------------
# 规则打分：5 维度特征融合
# ---------------------------------------------------------------------------

# 注释：满配权重 = 100，关闭某维度时权重会重分配
_WEIGHTS_FULL = {
    "area": 18.0,
    "center": 12.0,
    "depth": 25.0,
    "face": 25.0,
    "completeness": 10.0,
    "conf": 10.0,
}


def _depth_relative_scores(depths: list) -> list[float]:
    """
    把绝对 disparity 值映射到 0..1 的相对分数。
    同图里最近的人 = 1.0，最远 = 0.0，缺失值 = 0.5（中性）。
    """
    valid = [d for d in depths if d is not None]
    if not valid:
        return [0.5] * len(depths)
    d_min, d_max = min(valid), max(valid)
    span = d_max - d_min
    if span < 1e-6:
        return [0.5] * len(depths)
    return [(d - d_min) / span if d is not None else 0.5 for d in depths]


def rule_based_select(
    detections, image_h, image_w,
    features: list[BoxFeatures] | None = None,
    llm_config: LLMSelectionConfig | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    多维度融合规则打分，返回主体检测框的索引数组。
    """
    if not detections:
        return np.array([], dtype=int), []

    image_area = image_h * image_w
    center_x, center_y = image_w / 2.0, image_h / 2.0
    max_dist = float(np.sqrt(center_x**2 + center_y**2)) or 1.0

    area_scores, center_scores, conf_scores = [], [], []
    completeness_scores, face_scores, depth_raw = [], [], []

    for i, det in enumerate(detections):
        box = det["box"]
        conf = float(det.get("conf", 0.0))
        area = float(det.get("area", 0.0))

        # 1) 面积比
        area_ratio = area / max(1.0, image_area)
        area_scores.append(min(area_ratio / 0.05, 1.0))

        # 2) 中心接近度
        x1, y1, x2, y2 = [float(v) for v in box]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dist = float(np.sqrt((cx - center_x)**2 + (cy - center_y)**2))
        center_scores.append(max(0.0, 1.0 - dist / max_dist))

        # 3) 置信度
        conf_scores.append(min(max(conf, 0.0), 1.0))

        # 4) 完整度 / 人脸 / 深度
        if features is not None and i < len(features):
            completeness_scores.append(features[i].completeness)
            face_scores.append(features[i].face_score)
            depth_raw.append(features[i].depth_mean)
        else:
            from subject_features import compute_completeness_score
            completeness_scores.append(compute_completeness_score(box, image_h, image_w))
            face_scores.append(0.0)
            depth_raw.append(None)

    depth_scores = _depth_relative_scores(depth_raw)

    # 注释：是否启用某维度 = 配置开关 + 实际有数据（双重判断）
    use_depth = bool(llm_config and llm_config.enable_depth) and any(d is not None for d in depth_raw)
    use_face = bool(llm_config and llm_config.enable_face) and any(fs > 0.0 for fs in face_scores)

    # 注释：自适应权重降级 —— 关闭的维度权重分给 area/center
    weights = dict(_WEIGHTS_FULL)
    if not use_depth:
        weights["area"] += weights["depth"] * 0.6
        weights["center"] += weights["depth"] * 0.4
        weights["depth"] = 0.0
    if not use_face:
        weights["area"] += weights["face"] * 0.5
        weights["center"] += weights["face"] * 0.5
        weights["face"] = 0.0

    total = np.zeros(len(detections), dtype=np.float32)
    total += np.array(area_scores) * weights["area"]
    total += np.array(center_scores) * weights["center"]
    total += np.array(depth_scores) * weights["depth"]
    total += np.array(face_scores) * weights["face"]
    total += np.array(completeness_scores) * weights["completeness"]
    total += np.array(conf_scores) * weights["conf"]

    # 注释：按总分从高到低排序
    order = np.argsort(total)[::-1]
    top1_idx = int(order[0])
    top1_score = float(total[top1_idx])
    top1_area = float(detections[top1_idx]["area"])

    # 注释：贪心扩展主体集合
    # 从 top1 开始，把"明确属于同一组"的人逐步加进来
    # 加入条件（同时满足）：分数 >= 80% AND 面积 >= 45% AND 跟任一已选主体 < 30% 对角线
    subject_indices = [int(top1_idx)]
    diag = float(np.sqrt(image_h ** 2 + image_w ** 2))

    SCORE_RATIO = 0.80      # 注释：分数 >= top1 的 80%
    AREA_RATIO = 0.45       # 注释：面积 >= top1 的 45%
    MAX_SUBJECTS = 10        # 注释：最多 10 个主体（防误判兜底）

    for i in range(1, min(len(order), 6)):
        if len(subject_indices) >= MAX_SUBJECTS:
            break
        idx = int(order[i])
        score = float(total[idx])
        area = float(detections[idx]["area"])

        if score < top1_score * SCORE_RATIO:
            break    # 注释：分数太低，再往后更不可能
        if area < top1_area * AREA_RATIO:
            continue # 注释：太小，跳过

        box = detections[idx]["box"]
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2

        # 注释：检查"接近"——两种方式任一通过即可
        #   A) bbox 水平相邻 + 垂直对齐（合影最典型信号）
        #   B) 中心距离 < 25% 对角线（兜底，处理非典型站位）
        is_close = False
        for subj_idx in subject_indices:
            subj_box = detections[subj_idx]["box"]
            subj_cx = (subj_box[0] + subj_box[2]) / 2
            subj_cy = (subj_box[1] + subj_box[3]) / 2

            # 方式 A: bbox 水平相邻
            h_gap = max(0.0, max(box[0], subj_box[0]) - min(box[2], subj_box[2]))
            v_overlap = min(box[3], subj_box[3]) - max(box[1], subj_box[1])
            v_min = min(box[3] - box[1], subj_box[3] - subj_box[1])
            v_ratio = v_overlap / max(1, v_min) if v_min > 0 else 0
            if h_gap < image_w * 0.08 and v_ratio > 0.30:
                is_close = True
                break

            # 方式 B: 中心距离 < 25% 对角线
            dist = float(np.sqrt((cx - subj_cx) ** 2 + (cy - subj_cy) ** 2))
            if dist < diag * 0.25:
                is_close = True
                break

        if is_close:
            subject_indices.append(idx)
        else:
            break    # 注释：与所有已选主体都不接近 → 不是同一组


    # 注释：贪心代码用的是 Python list，转成 numpy 才能用 .astype(int)
    subject_indices = np.array(subject_indices, dtype=int)

    logs = []
    logs.append(
        f"[select] top1=#{top1_idx} score={top1_score:.1f}, "
        f"subject_count={len(subject_indices)}"
    )

    weights_str = ", ".join(f"{k}={v:.1f}" for k, v in weights.items() if v > 0)
    logs.append(f"规则打分完成: 启用维度={weights_str}; depth={use_depth}, face={use_face}")
    for i in subject_indices:
        logs.append(
            f"  #{i}: area={area_scores[i]:.2f} center={center_scores[i]:.2f} "
            f"depth={depth_scores[i]:.2f} face={face_scores[i]:.2f} "
            f"completeness={completeness_scores[i]:.2f} conf={conf_scores[i]:.2f} "
            f"→ total={total[i]:.1f}"
        )
    return subject_indices.astype(int), logs




def select_subject_indices(
    image_bgr: np.ndarray,
    detections: list[dict],
    masks: np.ndarray | None = None,
    llm_config: LLMSelectionConfig | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    对外入口：
      1) 先算扩展特征（深度 + 人脸 + 完整度）
      2) 多维度规则打分，返回主体检测框索引
    """
    if not detections:
        return np.array([], dtype=int), []
    if llm_config is None:
        llm_config = LLMSelectionConfig()

    image_h, image_w = image_bgr.shape[:2]
    features = extract_features_for_detections(
        image_bgr=image_bgr, detections=detections, masks=masks,
        enable_depth=llm_config.enable_depth, enable_face=llm_config.enable_face,
    )

    idx, logs = rule_based_select(detections, image_h, image_w,
                                   features=features, llm_config=llm_config)
    logs.append("[mode] 规则打分（MiDaS + BlazeFace 多维度特征）")
    return idx, logs