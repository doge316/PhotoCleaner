#!/usr/bin/env python3
"""测试 big-lama 修复效果。

流程：
  1. last.pt (YOLO检测) → 识别所有人物 bbox
  2. SAM (mobile_sam.pt) → 根据 bbox 生成精确人物轮廓 mask
  3. big-lama (官方 TorchScript 模型) → 修复 mask 区域
  4. 保存对比结果
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO, SAM

# ─── 配置 ────────────────────────────────────────────────────────────────
WEIGHTS = "last.pt"                           # YOLO 检测权重
SAM_MODEL = "models/mobile_sam.pt"             # SAM 模型 (精化 mask)
IMAGE = "消除路人/训练集/test2.png"             # 测试图片
OUTPUT_DIR = Path("result/big_lama_test2")     # 输出目录
CONF = 0.05                                     # YOLO 置信度阈值
IOU = 0.45                                      # NMS IoU 阈值
MODEL_PATH = "models/big-lama.pt"              # big-lama TorchScript 模型
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# last.pt 的类别: 0=pedestrian, 1=riders, 2=partially-visible, 3=ignore, 4=crowd
# 我们要消除的类别 (排除 ignore-regions)
PERSON_CLASSES = {0, 1, 2, 4}


# ─── 辅助函数 ─────────────────────────────────────────────────────────────

def ceil_modulo(x: int, mod: int) -> int:
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def pad_tensor_to_modulo(img: torch.Tensor, mod: int) -> torch.Tensor:
    """反射 padding 到 mod 的倍数 (FFC 网络要求 8 的倍数)"""
    _, _, h, w = img.shape
    out_h = ceil_modulo(h, mod)
    out_w = ceil_modulo(w, mod)
    return F.pad(img, pad=(0, out_w - w, 0, out_h - h), mode='reflect')


def load_lama_model(model_path: str, device: str) -> torch.jit.ScriptModule:
    """加载 big-lama TorchScript 模型并预热。"""
    print(f"[big-lama] 加载模型: {model_path}")
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    model.to(device)

    # 预热: 避免首次推理的 JIT 编译延迟
    print("[big-lama] 预热中...")
    dummy_img = torch.rand(1, 3, 256, 256, device=device)
    dummy_mask = torch.rand(1, 1, 256, 256, device=device)
    with torch.no_grad():
        _ = model(dummy_img, dummy_mask)
    print("[big-lama] 模型就绪")
    return model


@torch.no_grad()
def lama_inpaint(
    model: torch.jit.ScriptModule,
    image_bgr: np.ndarray,
    mask_u8: np.ndarray,
    device: str,
) -> np.ndarray:
    """用 big-lama 修复 mask 区域。

    严格遵循官方 LaMa 推理管线:
      - 图像 mask 都 pad 到 8 的倍数
      - 归一化到 [0, 1]
      - 4 通道输入: concat(masked_img, mask)
      - TorchScript 模型已内置 compositing
    """
    h_orig, w_orig = image_bgr.shape[:2]

    # BGR → RGB → CHW float32 [0,1]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy(image_rgb.astype(np.float32) / 255.0)
    img_t = img_t.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

    # mask: (H,W) uint8 → (1, 1, H, W) float32 [0,1]
    mask_t = torch.from_numpy((mask_u8 > 0).astype(np.float32))
    mask_t = mask_t.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # Pad 到 8 的倍数 (FFC downsampling 要求)
    padded_img = pad_tensor_to_modulo(img_t, 8).to(device)
    padded_mask = pad_tensor_to_modulo(mask_t, 8).to(device)

    # big-lama JIT 模型: forward(image, mask) → inpainted
    result = model(padded_img, padded_mask)

    # 裁剪回原尺寸
    result = result[:, :, :h_orig, :w_orig]

    # → HWC uint8, RGB → BGR
    result_np = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
    result_np = np.clip(result_np * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)


# ─── 主流程 ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  big-lama 修复效果测试")
    print("=" * 60)

    # ── 1. 加载 YOLO ─────────────────────────────────────────────────
    print(f"\n[1/5] 加载 YOLO 权重: {WEIGHTS}")
    yolo = YOLO(WEIGHTS)
    print(f"       任务: {yolo.task}  |  类别: {yolo.names}")
    print(f"       设备: {DEVICE}")

    # ── 2. 检测人物 ──────────────────────────────────────────────────
    print(f"\n[2/5] 检测人物: {IMAGE}")
    img_bgr = cv2.imread(IMAGE)
    if img_bgr is None:
        raise FileNotFoundError(f"无法读取图片: {IMAGE}")
    h, w = img_bgr.shape[:2]
    print(f"       图片尺寸: {w}×{h}")

    results = yolo(IMAGE, conf=CONF, iou=IOU, verbose=True)
    result = results[0]
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        print("       未检测到任何人，退出。")
        return

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)

    # 只保留人物类别 (排除 ignore-regions)
    person_idx = [i for i, c in enumerate(cls_ids) if c in PERSON_CLASSES]
    all_idx = np.arange(len(xyxy))
    ignored_idx = [i for i in all_idx if i not in person_idx]

    print(f"       检测到 {len(xyxy)} 个目标")
    print(f"       ├─ 人物 (消除): {len(person_idx)} 个")
    if ignored_idx:
        ignored_labels = {cls_ids[i]: yolo.names.get(cls_ids[i], '?') for i in ignored_idx}
        print(f"       └─ 忽略 ({ignored_labels}): {len(ignored_idx)} 个")

    if not person_idx:
        print("       没有可消除的人物，退出。")
        return

    # ── 3. SAM 精确 mask ──────────────────────────────────────────
    print(f"\n[3/6] SAM 生成精确人物轮廓...")
    person_xyxy = xyxy[person_idx]  # 只取要消除的人物 bbox

    # 加载 SAM
    print(f"       加载 SAM: {SAM_MODEL}")
    sam = SAM(SAM_MODEL)
    sam.to(DEVICE)

    # SAM 推理 (分批，避免 OOM)
    image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    sam_masks = []
    BATCH = 40
    boxes_list = person_xyxy.tolist()
    for start in range(0, len(boxes_list), BATCH):
        batch = boxes_list[start:start + BATCH]
        sam_result = sam.predict(image_rgb, bboxes=batch, verbose=False, save=False, show=False)
        if sam_result and sam_result[0].masks is not None:
            batch_masks = sam_result[0].masks.data.cpu().numpy()
            # resize 到原图尺寸
            if batch_masks.shape[1:] != (h, w):
                batch_masks = np.array([
                    cv2.resize(m.astype(np.float32), (w, h)) for m in batch_masks
                ])
            sam_masks.append(batch_masks)

    if sam_masks:
        all_sam = np.concatenate(sam_masks, axis=0)
        # 合并所有人物的 SAM mask
        person_mask = np.zeros((h, w), dtype=np.uint8)
        for m in all_sam:
            person_mask = np.maximum(person_mask, (m > 0.5).astype(np.uint8))
        person_mask = person_mask * 255
        print(f"       SAM 生成 {len(all_sam)} 个精确 mask")
    else:
        # SAM 失败，降级为 bbox mask
        print("       SAM 失败，降级为 bbox mask")
        person_mask = np.zeros((h, w), dtype=np.uint8)
        for i in person_idx:
            x1, y1, x2, y2 = map(int, xyxy[i])
            person_mask[y1:y2, x1:x2] = 255

    coverage = person_mask.mean() / 255 * 100
    print(f"       Mask 覆盖面积: {coverage:.1f}%")

    # ── 4. big-lama 双次修复 (全图消人) ──────────────────────────────
    print(f"\n[4/6] big-lama 双次修复 (全图消人, device={DEVICE})...")
    lama_model = load_lama_model(MODEL_PATH, DEVICE)

    # 膨胀 mask: 给人物边缘加 5-10 像素缓冲区，避免残留轮廓伪影
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated_mask = cv2.dilate(person_mask, kernel, iterations=2)
    dilate_coverage = dilated_mask.mean() / 255 * 100
    print(f"       mask 膨胀后覆盖: {dilate_coverage:.1f}%")

    # 第一次修复: 消除所有人物
    print("       [1/2] 第一次修复...")
    first_pass = lama_inpaint(lama_model, img_bgr, dilated_mask, DEVICE)
    # 第二次修复: 清理残留伪影
    print("       [2/2] 第二次修复...")
    clean_bgr = lama_inpaint(lama_model, first_pass, dilated_mask, DEVICE)
    print("       双次修复完成")

    # ── 5. 保存结果 ───────────────────────────────────────────────────
    print(f"\n[5/6] 保存结果 -> {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 修复结果
    out_clean = OUTPUT_DIR / f"{Path(IMAGE).stem}_clean.png"
    cv2.imwrite(str(out_clean), clean_bgr)

    # 检测框可视化 (绿色=人物要消除, 红色=忽略)
    vis = img_bgr.copy()
    for i in all_idx:
        x1, y1, x2, y2 = map(int, xyxy[i])
        conf = confs[i]
        cls_name = yolo.names.get(cls_ids[i], '?')
        if i in person_idx:
            color = (0, 255, 0)  # 绿色 = 将被消除
            label = f"{cls_name} {conf:.2f} [消除]"
        else:
            color = (0, 0, 255)  # 红色 = 忽略
            label = f"{cls_name} {conf:.2f} [忽略]"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, label, (x1, max(y1 - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    out_detect = OUTPUT_DIR / f"{Path(IMAGE).stem}_detect.png"
    cv2.imwrite(str(out_detect), vis)

    # 保存 SAM mask 可视化 (半透明叠加在原图上)
    overlay = img_bgr.copy()
    overlay[dilated_mask > 0] = overlay[dilated_mask > 0] // 2 + np.array([0, 0, 200], dtype=np.uint8) // 2
    out_mask = OUTPUT_DIR / f"{Path(IMAGE).stem}_mask.png"
    cv2.imwrite(str(out_mask), overlay)

    # 保存第一次修复结果 (中间产物)
    out_first = OUTPUT_DIR / f"{Path(IMAGE).stem}_first_pass.png"
    cv2.imwrite(str(out_first), first_pass)

    # 并排对比图 (原图 / mask / 第一次修复 / 最终修复)
    h_cap = 30
    canvas = np.ones((h + h_cap, w * 4, 3), dtype=np.uint8) * 255
    canvas[h_cap:, :w] = img_bgr
    canvas[h_cap:, w:2*w] = overlay
    canvas[h_cap:, 2*w:3*w] = first_pass
    canvas[h_cap:, 3*w:] = clean_bgr
    labels = ["Original", "Mask (dilated)", "1st Pass", "2nd Pass (final)"]
    for i, label in enumerate(labels):
        x = i * w + w // 2 - len(label) * 10
        cv2.putText(canvas, label, (x, h_cap - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    out_compare = OUTPUT_DIR / f"{Path(IMAGE).stem}_compare.png"
    cv2.imwrite(str(out_compare), canvas)

    # 打印汇总
    print(f"\n[6/6] 完成！")
    print(f"{'─' * 60}")
    print(f"  检测图:       {out_detect}")
    print(f"  SAM mask:     {out_mask}")
    print(f"  第一次修复:   {out_first}")
    print(f"  最终修复:     {out_clean}")
    print(f"  四栏对比:     {out_compare}")
    print(f"  YOLO({len(person_idx)}人) → SAM mask → mask膨胀 → big-lama×2")
    print(f"{'─' * 60}")


if __name__ == "__main__":
    main()
