from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from db import get_failed_records, get_recent_records, get_success_records, init_db
from photo_cleaner_core import collect_images, format_detection_lines, process_image


st.set_page_config(
    page_title="PhotoCleaner 路人消除系统",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def bgr_to_rgb(image_bgr: np.ndarray | None) -> np.ndarray | None:
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def image_to_bytes(image_bgr: np.ndarray) -> bytes:
    success, buffer = cv2.imencode(".png", image_bgr)
    if not success:
        return b""
    return buffer.tobytes()


def render_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f4efe7;
            --panel: rgba(255, 255, 255, 0.78);
            --panel-strong: #ffffff;
            --text: #1f2937;
            --muted: #6b7280;
            --accent: #0f766e;
            --accent-2: #ef7d57;
            --line: rgba(31, 41, 55, 0.10);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15, 118, 110, 0.10), transparent 32%),
                radial-gradient(circle at top right, rgba(239, 125, 87, 0.14), transparent 26%),
                linear-gradient(180deg, #f8f3ea 0%, #f4efe7 100%);
            color: var(--text);
        }

        .hero {
            padding: 2rem 2rem 1.6rem 2rem;
            border: 1px solid var(--line);
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(255,255,255,0.90), rgba(255,255,255,0.65));
            box-shadow: 0 18px 60px rgba(31, 41, 55, 0.10);
            margin-bottom: 1rem;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.2rem;
            letter-spacing: 0.02em;
        }

        .hero p {
            margin: 0.6rem 0 0;
            color: var(--muted);
            line-height: 1.7;
        }

        .feature-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 1rem 0 0.5rem;
        }

        .feature-card {
            background: var(--panel-strong);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            min-height: 78px;
        }

        .feature-card .label {
            color: var(--muted);
            font-size: 0.82rem;
            margin-bottom: 0.35rem;
        }

        .feature-card .value {
            color: var(--text);
            font-weight: 700;
            font-size: 0.98rem;
        }

        .soft-panel {
            background: rgba(255,255,255,0.70);
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 1rem 1rem 0.8rem;
            box-shadow: 0 10px 32px rgba(31, 41, 55, 0.08);
        }

        .section-title {
            margin: 0 0 0.75rem;
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--text);
        }

        .small-muted {
            color: var(--muted);
            font-size: 0.9rem;
        }

        .stButton > button {
            border-radius: 999px;
            border: none;
            padding: 0.7rem 1.15rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent), #155e75);
            color: white;
            box-shadow: 0 8px 20px rgba(15, 118, 110, 0.22);
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 26px rgba(15, 118, 110, 0.28);
        }

        div[data-testid="stFileUploaderDropzone"] {
            background: rgba(255,255,255,0.88);
            border: 1px dashed rgba(15, 118, 110, 0.35);
            border-radius: 20px;
        }

        .stTabs [role="tablist"] {
            gap: 0.3rem;
        }

        .stTabs [role="tab"] {
            border-radius: 999px;
            padding: 0.45rem 1rem;
        }

        .stDataFrame, .stTable {
            border-radius: 16px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>PhotoCleaner 路人自动识别与消除系统</h1>
            <p>
                面向景区、展馆、活动现场等真实拍照场景，支持单张与批量图片处理。
                系统自动判断主体与路人，完成分割、消除与背景修复，并保留处理日志和结果文件。
            </p>
            <div class="feature-grid">
                <div class="feature-card"><div class="label">输入格式</div><div class="value">PNG / JPG / JPEG</div></div>
                <div class="feature-card"><div class="label">AI 核心</div><div class="value">YOLOv8-seg + OpenCV 修复</div></div>
                <div class="feature-card"><div class="label">处理方式</div><div class="value">单张 / 批量 / 目录扫描</div></div>
                <div class="feature-card"><div class="label">存储方式</div><div class="value">本地输出 + SQLite 日志</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_settings() -> dict[str, object]:
    st.sidebar.markdown("### 处理参数")
    model_path = st.sidebar.text_input("模型路径", value="yolov8s-seg.pt")
    output_dir = st.sidebar.text_input("结果输出目录", value="消除路人/结果集")
    inpaint_radius = st.sidebar.slider("修复半径", 1, 15, 3)
    subject_score_ratio = st.sidebar.slider("主体保留阈值", 0.5, 0.95, 0.75, 0.01)
    min_area_ratio = st.sidebar.slider("最小人物面积占比", 0.01, 0.10, 0.03, 0.005)
    save_masks = st.sidebar.checkbox("同时保存 mask", value=True)
    st.sidebar.caption("建议先使用默认值，再根据拍照风格微调。")

    return {
        "model_path": model_path,
        "output_dir": output_dir,
        "inpaint_radius": inpaint_radius,
        "subject_score_ratio": subject_score_ratio,
        "min_area_ratio": min_area_ratio,
        "save_masks": save_masks,
    }


def show_result(result, saved_paths: dict[str, Path]) -> None:
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("#### 原图")
        st.image(bgr_to_rgb(result.original_bgr), use_container_width=True)
    with col2:
        st.markdown("#### 处理后")
        st.image(bgr_to_rgb(result.cleaned_bgr), use_container_width=True)

    metrics = st.columns(4)
    metrics[0].metric("主体数量", result.subject_count)
    metrics[1].metric("路人数量", result.stray_count)
    metrics[2].metric("耗时(秒)", f"{result.elapsed_seconds:.2f}")
    metrics[3].metric("输出文件", Path(saved_paths["output"]).name)

    st.markdown("#### 检测明细")
    lines = format_detection_lines(result.detections)
    if lines:
        for line in lines:
            st.write(line)
    else:
        st.info("未检测到人物，系统已输出原图。")

    st.download_button(
        label="下载处理结果",
        data=image_to_bytes(result.cleaned_bgr),
        file_name=Path(saved_paths["output"]).name,
        mime="image/png",
        use_container_width=True,
    )


def handle_single_mode(settings: dict[str, object]) -> None:
    uploaded_file = st.file_uploader("上传单张图片", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
    if not uploaded_file:
        st.info("请先上传一张拍照场景图片。")
        return

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.image(uploaded_file, caption="待处理图片", use_container_width=True)

    with right:
        st.markdown("#### 处理说明")
        st.write("系统会自动识别人物实例，按主体规则筛选需要保留的人物，再对路人区域进行修复。")
        start_button = st.button("开始处理单张图片", use_container_width=True)

    if not start_button:
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir) / uploaded_file.name
        temp_path.write_bytes(uploaded_file.getbuffer())

        progress = st.progress(0)
        status_box = st.empty()
        status_box.write("正在读取与分析图片...")
        progress.progress(20)

        result, saved_paths = process_image(
            image_path=temp_path,
            output_dir=settings["output_dir"],
            model_path=settings["model_path"],
            subject_score_ratio=float(settings["subject_score_ratio"]),
            min_area_ratio=float(settings["min_area_ratio"]),
            inpaint_radius=int(settings["inpaint_radius"]),
            save_masks=bool(settings["save_masks"]),
        )

        status_box.write("正在保存结果并写入日志...")
        progress.progress(90)
        st.session_state["last_result"] = result
        st.session_state["last_saved_paths"] = saved_paths
        progress.progress(100)
        status_box.success("处理完成")

    show_result(result, saved_paths)


def handle_batch_mode(settings: dict[str, object]) -> None:
    batch_dir = st.text_input("输入本地图片目录", value="消除路人/训练集")
    uploaded_files = st.file_uploader(
        "或者一次上传多张图片",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    candidates: list[Path] = []
    if batch_dir:
        candidates.extend(collect_images(batch_dir))

    if uploaded_files:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_dir = Path(tmp_dir)
            for uploaded_file in uploaded_files:
                temp_path = temp_dir / uploaded_file.name
                temp_path.write_bytes(uploaded_file.getbuffer())
                candidates.append(temp_path)

            st.write(f"待处理图片数量: {len(candidates)}")
            if st.button("开始批量处理", use_container_width=True):
                run_batch(candidates, settings)
        return

    st.write(f"待处理图片数量: {len(candidates)}")
    if st.button("开始批量处理", use_container_width=True):
        run_batch(candidates, settings)


def run_batch(candidates: list[Path], settings: dict[str, object]) -> None:
    if not candidates:
        st.warning("没有可处理的图片。")
        return

    progress = st.progress(0)
    status_box = st.empty()
    results_area = st.container()
    total = len(candidates)
    success_count = 0
    failure_count = 0

    for index, image_path in enumerate(candidates, start=1):
        status_box.write(f"正在处理 {image_path.name} ({index}/{total})")
        try:
            result, saved_paths = process_image(
                image_path=image_path,
                output_dir=settings["output_dir"],
                model_path=settings["model_path"],
                subject_score_ratio=float(settings["subject_score_ratio"]),
                min_area_ratio=float(settings["min_area_ratio"]),
                inpaint_radius=int(settings["inpaint_radius"]),
                save_masks=bool(settings["save_masks"]),
            )
            success_count += 1
            with results_area.expander(f"{image_path.name} - 成功", expanded=False):
                show_result(result, saved_paths)
        except Exception as exc:
            failure_count += 1
            with results_area.expander(f"{image_path.name} - 失败", expanded=False):
                st.error(str(exc))

        progress.progress(int(index / total * 100))

    status_box.success(f"批量处理完成，成功 {success_count} 张，失败 {failure_count} 张")


def render_history_tab() -> None:
    recent_records = get_recent_records(limit=20)
    success_records = get_success_records()
    failed_records = get_failed_records()

    st.markdown("#### 最近记录")
    if recent_records:
        st.dataframe(recent_records, use_container_width=True, hide_index=True)
    else:
        st.info("暂无处理记录。")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("#### 成功记录")
        if success_records:
            st.dataframe(success_records, use_container_width=True, hide_index=True)
        else:
            st.info("暂无成功记录。")

    with col2:
        st.markdown("#### 失败记录")
        if failed_records:
            st.dataframe(failed_records, use_container_width=True, hide_index=True)
        else:
            st.info("暂无失败记录。")


def main() -> None:
    init_db()
    render_css()
    render_hero()

    settings = sidebar_settings()

    tabs = st.tabs(["单张处理", "批量处理", "处理历史"])
    with tabs[0]:
        st.markdown('<div class="soft-panel">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">单张图片处理</p>', unsafe_allow_html=True)
        handle_single_mode(settings)
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[1]:
        st.markdown('<div class="soft-panel">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">批量图片处理</p>', unsafe_allow_html=True)
        handle_batch_mode(settings)
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[2]:
        st.markdown('<div class="soft-panel">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">处理历史</p>', unsafe_allow_html=True)
        render_history_tab()
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
