"""
llm_subject_selector.py
-----------------------
主体选择模块：决定图中哪些人物是"主体"、哪些是"路人"。

两条路径：
  1. LLM（OpenAI 兼容视觉模型，Ollama / vLLM / 任何 chat completions 接口）
  2. 规则打分（多维度特征融合：面积、中心、深度、人脸、完整度、置信度）

LLM 不可用时自动 fallback 到规则版本，并保留 LLM 的返回作为日志参考。
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import cv2
import numpy as np

from subject_features import BoxFeatures, extract_features_for_detections


@dataclass
class LLMSelectionConfig:
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen3.5:9b"
    api_key: str = ""
    timeout_seconds: int = 90
    # 注释：新增开关，UI 侧边栏会动态绑定
    enable_depth: bool = True
    enable_face: bool = True
    # 注释：主体判断模式
    #   auto   - 自动（默认）
    #   single - 强制单人主体
    #   multi  - 允许合影等多主体
    subject_mode: str = "auto"

def image_to_data_url(image_bgr: np.ndarray) -> str:
    success, buffer = cv2.imencode(".png", image_bgr)
    if not success:
        raise ValueError("无法编码图片")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_detection_overlay(image_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    overlay = image_bgr.copy()
    for detection in detections:
        box = detection["box"]
        index = detection["index"]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 180, 255), 3)
        label = f"#{index}"
        text_origin = (x1, max(22, y1 - 8))
        cv2.putText(overlay, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 180, 255), 2, cv2.LINE_AA)
    return overlay


def parse_llm_indices(text: str, total_count: int) -> list[int]:
    text = text.strip()
    if not text:
        raise ValueError("模型没有返回内容")
    candidate_texts = [text]
    if "```" in text:
        candidate_texts.append(text.split("```", 2)[1])
    for candidate in candidate_texts:
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                subject_indices = payload.get("subject_indices", payload.get("keep_indices", []))
            elif isinstance(payload, list):
                subject_indices = payload
            else:
                continue
            if not isinstance(subject_indices, list):
                continue
            cleaned = sorted({int(item) for item in subject_indices
                              if isinstance(item, (int, float, str))})
            return [item for item in cleaned if 0 <= item < total_count]
        except Exception:
            continue
    raise ValueError(f"无法解析模型返回的主体编号: {text}")


def call_openai_compatible_vision_model(
    image_bgr, overlay_bgr, detections, config: LLMSelectionConfig
) -> tuple[list[int], str]:
    prompt = (
        "你是照片主体选择器。图中每个检测框都标了编号。\n"
        "请根据原图和编号框图判断哪些人物应该保留。\n"
        "只返回 JSON，格式: {\"subject_indices\":[0,2]}"
    )
    payload: dict[str, Any] = {
        "model": config.model, "temperature": 0,
        "messages": [
            {"role": "system", "content": "你是一个严格输出 JSON 的视觉判断助手。"},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_to_data_url(image_bgr)}},
                {"type": "image_url", "image_url": {"url": image_to_data_url(overlay_bgr)}},
                {"type": "text", "text": "候选人物：\n" + "\n".join(
                    [f'#{d["index"]}: box={d["box"]}, conf={d["conf"]:.3f}, area={d["area"]:.0f}'
                     for d in detections]
                )},
            ]},
        ],
    }
    url = config.base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    req = urllib_request.Request(url=url, data=json.dumps(payload).encode("utf-8"),
                                  headers=headers, method="POST")
    try:
        with urllib_request.urlopen(req, timeout=config.timeout_seconds) as resp:
            response_text = resp.read().decode("utf-8")
    except urllib_error.URLError as exc:
        raise RuntimeError(f"调用大模型失败: {exc}") from exc
    response_json = json.loads(response_text)
    choices = response_json.get("choices", [])
    if not choices:
        raise RuntimeError("大模型没有返回 choices")
    content = choices[0].get("message", {}).get("content", "")
    return parse_llm_indices(content, total_count=len(detections)), content


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


    # 注释：subject_mode="single" 兜底（手动指定时强制单人）

    if llm_config is not None and getattr(llm_config, "subject_mode", "auto") == "single":

        if len(subject_indices) > 1:

            subject_indices = subject_indices[:1]

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


# ---------------------------------------------------------------------------
# LLM 主体选择
# ---------------------------------------------------------------------------


def llm_subject_select(image_bgr, detections, llm_config=None):
    if not detections:
        return np.array([], dtype=int), []
    if llm_config is None:
        llm_config = LLMSelectionConfig()
    if not llm_config.model.strip() or not llm_config.base_url.strip():
        image_h, image_w = image_bgr.shape[:2]
        return rule_based_select(detections, image_h, image_w, llm_config=llm_config)
    overlay = render_detection_overlay(image_bgr, detections)
    subject_indices, raw = call_openai_compatible_vision_model(
        image_bgr, overlay, detections, llm_config
    )
    if not subject_indices:
        raise RuntimeError(f"大模型没有返回有效主体编号: {raw}")
    return np.array(subject_indices, dtype=int), ["大模型已完成主体判断。", f"大模型返回: {raw}"]


def select_subject_indices(
    image_bgr: np.ndarray,
    detections: list[dict],
    masks: np.ndarray | None = None,
    llm_config: LLMSelectionConfig | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    对外入口：
      1) 先算扩展特征（深度 + 人脸 + 完整度）
      2) 优先尝试 LLM，失败 fallback 规则
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

    if not llm_config.model.strip() or not llm_config.base_url.strip():
        idx, logs = rule_based_select(detections, image_h, image_w,
                                       features=features, llm_config=llm_config)
        logs.append("[mode] 纯规则模式（未配置 LLM）")
        return idx, logs

    try:
        idx, logs = llm_subject_select(image_bgr, detections, llm_config)
        logs.append("[mode] LLM 判断成功")
        return idx, logs
    except Exception as exc:
        idx, logs = rule_based_select(detections, image_h, image_w,
                                       features=features, llm_config=llm_config)
        logs.append(f"[mode] LLM 调用失败（{type(exc).__name__}: {exc}），已自动切换为规则判断。")
        return idx, logs