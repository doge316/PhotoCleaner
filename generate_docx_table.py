"""
生成 PhotoCleaner 项目技术难题与解决方案的 Word 表格。

用法：
    pip install python-docx
    python generate_docx_table.py

会在当前目录生成：技术难题与解决方案_PhotoCleaner.docx
"""

from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn


def set_cell_text(cell, text, bold=False, size=10.5):
    cell.text = ""
    p = cell.paragraphs[0]
    run = p.add_run(text)
    run.font.name = "Microsoft YaHei"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(size)
    run.font.bold = bold
    return p


def main():
    doc = Document()

    # 标题
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("PhotoCleaner 项目技术难题与解决方案")
    run.font.name = "Microsoft YaHei"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(18)
    run.font.bold = True

    # 表格：3 列 6 行（1 表头 + 5 数据行）
    table = doc.add_table(rows=6, cols=3)
    table.style = "Table Grid"
    table.autofit = False
    table.allow_autofit = False
    table.columns[0].width = Cm(2.0)
    table.columns[1].width = Cm(6.5)
    table.columns[2].width = Cm(8.5)

    headers = ["阶段", "遇到的技术难题", "解决方案与心得体会"]
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        p = set_cell_text(cell, h, bold=True, size=11)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        # 表头背景色（浅蓝）
        shading_elm = cell._tc.get_or_add_tcPr()
        from docx.oxml import parse_xml
        shading_elm.append(parse_xml(r'<w:shd {} w:fill="4F81BD"/>'.format('xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"')))
        for run in p.runs:
            run.font.color.rgb = RGBColor(255, 255, 255)

    rows_data = [
        (
            "第 1 周",
            "基础消除管线搭建：YOLO 检测人物后，直接用 LaMa 修复路人区域，但小目标 / 远处人物漏检严重，且修复后背景存在伪影。",
            "降低 YOLO 置信度阈值并引入多尺度检测（原图 + 2 倍放大图），用 IoU 去重合并结果，显著提升小人物召回率；修复流程采用 LaMa 双次修复减少伪影。\n\n相关代码：photo_cleaner_core.py:535-591",
        ),
        (
            "第 2 周",
            "主体 vs 路人判断不准：最初尝试让 LLM 直接判断主体，但成本高、响应慢、结果不稳定，随后被 revert。",
            "改为多维度规则打分融合：面积、中心位置、MiDaS 深度、BlazeFace 人脸完整度、人物完整度、置信度，并设计自适应权重降级。\n\n相关代码：llm_subject_selector.py:60-238、subject_features.py:127-355",
        ),
        (
            "第 3 周",
            "单人物场景修复成功，多人物合影场景仍出问题：合影中几人站得近，系统容易把主体同伴误判为路人，或把路人误判为主体。",
            "引入贪心主体集合扩展：先选分数最高者为主体，再按“分数 ≥ top1 的 80%、面积 ≥ top1 的 45%、水平相邻或中心距离 < 25% 对角线”逐步把同伴加回主体集合。\n\n相关代码：llm_subject_selector.py:164-217",
        ),
        (
            "第 4 周",
            "人物消除与边缘质量：LaMa 在复杂背景或大面积遮挡时修复效果差；YOLO 原生 mask 边缘粗糙，主体回贴后出现白边 / 黑边。",
            "增加云端 Qwen 图像编辑 API 作为优先消除后端，失败自动回退 LaMa；同时引入可选 SAM 高精度 mask、GrabCut 边缘精化、以及 LLM 轮廓精修兜底。\n\n相关代码：photo_cleaner_qwen.py:363-402、app.py:213-222、photo_cleaner_qwen.py:463-470",
        ),
        (
            "第 5 周",
            "跨平台部署与环境兼容性：不同机器 CUDA/CPU 环境差异大；MiDaS 加载 EfficientNet 骨架时会去网上找权重；MediaPipe 在 WSL/Docker 下常报 libGLESv2.so.2 找不到。",
            "统一用 device_utils.get_device() 选择设备，并补丁 torch.jit.load 的 map_location；setup_models.py 集中下载所有模型并创建骨架权重软链接；在 subject_features.py 中预加载 libGLESv2 并注入 LD_LIBRARY_PATH。\n\n相关代码：device_utils.py:13-22、photo_cleaner_core.py:16-28、setup_models.py:18-105、subject_features.py:212-237",
        ),
    ]

    for row_idx, (stage, problem, solution) in enumerate(rows_data, start=1):
        row = table.rows[row_idx]
        set_cell_text(row.cells[0], stage)
        set_cell_text(row.cells[1], problem)
        set_cell_text(row.cells[2], solution)

    output = "技术难题与解决方案_PhotoCleaner.docx"
    doc.save(output)
    print(f"已生成：{output}")


if __name__ == "__main__":
    main()
