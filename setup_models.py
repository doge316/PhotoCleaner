#!/usr/bin/env python3
"""PhotoCleaner 模型下载脚本

在新机器上首次使用时运行一次：
    python setup_models.py

所有模型文件下载到 models/ 目录。已有文件会自动跳过。
"""

import os
import sys
import hashlib
from pathlib import Path
from urllib import request

MODELS_DIR = Path(__file__).resolve().parent / "models"

MODELS = [
    {
        "filename": "yolov8s-seg.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s-seg.pt",
        "size_mb": 23,
    },
    {
        "filename": "mobile_sam.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/mobile_sam.pt",
        "size_mb": 39,
    },
    {
        "filename": "big-lama.pt",
        "url": "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
        "size_mb": 197,
    },
    {
        "filename": "midas_v21_small_256.pt",
        "url": "https://github.com/isl-org/MiDaS/releases/download/v3_1/midas_v21_small_256.pt",
        "size_mb": 82,
    },
    {
        "filename": "tf_efficientnet_lite3-b733e338.pth",
        "url": (
            "https://github.com/rwightman/pytorch-image-models/releases/download/"
            "v0.1-weights/tf_efficientnet_lite3-b733e338.pth"
        ),
        "size_mb": 32,
    },
]


def download_file(url: str, dest: Path, desc: str) -> bool:
    """下载文件，已有则跳过。返回 True 表示成功或已存在。"""
    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  ✓ 已存在 ({size_mb:.0f} MB)")
        return True

    print(f"  下载中 ({desc})...")
    try:
        request.urlretrieve(url, str(dest))
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  ✓ 完成 ({size_mb:.0f} MB)")
        return True
    except Exception as exc:
        print(f"  ✗ 失败: {exc}")
        # 清理不完整的文件
        if dest.exists():
            dest.unlink()
        return False


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 创建 checkpoints 目录供 geffnet 加载 efficientnet 骨架权重
    checkpoints_dir = MODELS_DIR / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("PhotoCleaner 模型下载")
    print(f"目标目录: {MODELS_DIR}")
    print("=" * 60)

    failed = []

    for model in MODELS:
        filename = model["filename"]
        url = model["url"]
        size_mb = model["size_mb"]
        dest = MODELS_DIR / filename

        print(f"\n[{filename}] ({size_mb} MB)")
        if not download_file(url, dest, f"{size_mb} MB"):
            failed.append(filename)

    # 为 geffnet 创建 efficientnet 权重的硬链接/副本
    backbone_src = MODELS_DIR / "tf_efficientnet_lite3-b733e338.pth"
    backbone_dst = checkpoints_dir / "tf_efficientnet_lite3-b733e338.pth"
    if backbone_src.exists() and not backbone_dst.exists():
        try:
            backbone_dst.symlink_to(os.path.relpath(backbone_src, checkpoints_dir))
        except OSError:
            # Windows 可能不支持 symlink，改用副本
            import shutil
            shutil.copy2(str(backbone_src), str(backbone_dst))
        print(f"\n[checkpoints] efficientnet 骨架权重已链接。")

    # BlazeFace 模型（从 Google CDN 下载，仅 ~225KB）
    blazeface_path = MODELS_DIR / "blaze_face_short_range.tflite"
    if not blazeface_path.exists():
        print("\n[blaze_face_short_range.tflite] (225 KB)")
        download_file(
            "https://storage.googleapis.com/mediapipe-models/face_detector/"
            "blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
            blazeface_path,
            "225 KB",
        )

    print("\n" + "=" * 60)
    if failed:
        print(f"以下模型下载失败，请检查网络后重试: {failed}")
        sys.exit(1)
    else:
        total = sum(m["size_mb"] for m in MODELS) + 1
        print(f"全部模型就绪（共约 {total} MB）。")
        print("现在可以运行: streamlit run app.py")


if __name__ == "__main__":
    main()
