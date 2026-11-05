"""data_sythesis/preprocess/docx.py -- Word 文档预处理，帮助模型理解 .docx 内容。

对应 skills/docx 中的能力：
  - pandoc 文本提取（含 tracked changes）
  - unpack.py 解析 XML 获取结构
  - soffice.py 转 PDF/图片

依赖:
    pandoc（命令行工具）
    python-docx：pip install python-docx
    LibreOffice：soffice（可选，用于转图片）
    Poppler：pdftoppm（可选，用于转图片）
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


def extract_text_pandoc(docx_path: str, track_changes: str = "all") -> str:
    """使用 pandoc 提取 .docx 文本，可保留修订记录。

    Args:
        docx_path: .docx 文件路径。
        track_changes: pandoc --track-changes 选项，可选 "accept" | "reject" | "all"。

    Returns:
        提取的 Markdown 格式文本字符串。
    """
    result = subprocess.run(
        ["pandoc", f"--track-changes={track_changes}", docx_path, "-t", "markdown"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pandoc 提取失败: {result.stderr}")
    return result.stdout


def extract_text_python_docx(docx_path: str) -> str:
    """使用 python-docx 提取 .docx 正文文本（段落 + 表格）。

    Args:
        docx_path: .docx 文件路径。

    Returns:
        提取的文本字符串。
    """
    from docx import Document

    doc = Document(docx_path)
    parts: List[str] = []

    for block in _iter_block_items(doc):
        if hasattr(block, "text"):
            # 段落
            text = block.text.strip()
            if text:
                parts.append(text)
        else:
            # 表格
            for row in block.rows:
                row_cells = [cell.text.strip() for cell in row.cells]
                parts.append(" | ".join(row_cells))

    return "\n".join(parts)


def _iter_block_items(doc: Any):
    """按文档顺序遍历段落和表格（python-docx 内部辅助）。"""
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    parent = doc.element.body
    for child in parent.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, parent)
        elif child.tag == qn("w:tbl"):
            yield Table(child, parent)


def extract_metadata(docx_path: str) -> Dict[str, Any]:
    """提取 .docx 核心属性（标题、作者、创建/修改时间等）。

    Returns:
        包含文档属性的字典。
    """
    from docx import Document

    doc = Document(docx_path)
    props = doc.core_properties
    return {
        "title": props.title or "",
        "author": props.author or "",
        "subject": props.subject or "",
        "description": props.comments or "",
        "created": str(props.created) if props.created else "",
        "modified": str(props.modified) if props.modified else "",
        "last_modified_by": props.last_modified_by or "",
        "revision": props.revision,
    }


def extract_tables(docx_path: str) -> List[List[List[str]]]:
    """提取 .docx 中所有表格数据。

    Returns:
        列表，每个元素为一张表格（二维列表，行 × 列）。
    """
    from docx import Document

    doc = Document(docx_path)
    tables: List[List[List[str]]] = []
    for table in doc.tables:
        rows = [[cell.text for cell in row.cells] for row in table.rows]
        tables.append(rows)
    return tables


def extract_headings(docx_path: str) -> List[Dict[str, Any]]:
    """提取 .docx 中的各级标题，便于模型理解文档结构。

    Returns:
        [{"level": 1, "text": "..."}, ...]
    """
    from docx import Document
    from docx.oxml.ns import qn

    doc = Document(docx_path)
    headings: List[Dict[str, Any]] = []
    for para in doc.paragraphs:
        style_name = para.style.name if para.style else ""
        if style_name.startswith("Heading"):
            try:
                level = int(style_name.split()[-1])
            except ValueError:
                level = 0
            text = para.text.strip()
            if text:
                headings.append({"level": level, "text": text})
    return headings


def convert_to_images(
    docx_path: str,
    output_dir: str,
    soffice_script: Optional[str] = None,
    use_soffice: bool = True,
) -> List[str]:
    """将 .docx 转为逐页图片。

    use_soffice=True（默认）：docx → PDF（soffice） → 图片（pdftoppm）
    use_soffice=False：docx → PDF（python-docx2pdf / docx2pdf）→ 图片（pymupdf）
                       若 docx2pdf 不可用，则仅用 pymupdf 直接渲染 PDF（需先手动转好）。
                       实际降级路径：先尝试 docx2pdf，失败则跳过图片。

    Args:
        docx_path: .docx 文件路径。
        output_dir: 图片输出目录。
        soffice_script: skills/docx/scripts/office/soffice.py 路径（use_soffice=True 时有效），
                        若为 None 则直接调用系统 soffice。
        use_soffice: 是否使用 LibreOffice（soffice）进行转换，默认 True。

    Returns:
        生成的图片路径列表。
    """
    os.makedirs(output_dir, exist_ok=True)
    docx_path = str(Path(docx_path).resolve())

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path: Optional[str] = None

        if use_soffice:
            # --- 路径 A: soffice 转 PDF ---
            if soffice_script:
                cmd = ["python", soffice_script, "--headless", "--convert-to", "pdf",
                       "--outdir", tmp, docx_path]
            else:
                user_install = f"file://{tmp}/.soffice_profile"
                cmd = ["soffice", f"-env:UserInstallation={user_install}",
                       "--headless", "--convert-to", "pdf",
                       "--outdir", tmp, docx_path]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"soffice 转换失败: {result.stderr}")

            pdf_files = list(Path(tmp).glob("*.pdf"))
            if not pdf_files:
                raise RuntimeError("soffice 未生成 PDF 文件")
            pdf_path = str(pdf_files[0])

            # pdf → images（pdftoppm）
            prefix = os.path.join(output_dir, "page")
            result2 = subprocess.run(
                ["pdftoppm", "-jpeg", "-r", "150", pdf_path, prefix],
                capture_output=True,
                text=True,
            )
            if result2.returncode != 0:
                raise RuntimeError(f"pdftoppm 失败: {result2.stderr}")

            images = sorted(Path(output_dir).glob("page-*.jpg"))
            return [str(p) for p in images]

        else:
            # --- 路径 B: docx2pdf 转 PDF，再用 pymupdf 渲染 ---
            try:
                from docx2pdf import convert as _docx2pdf_convert
                pdf_path = os.path.join(tmp, Path(docx_path).stem + ".pdf")
                _docx2pdf_convert(docx_path, pdf_path)
            except Exception as e:
                raise RuntimeError(f"docx2pdf 转换失败（请确认已安装 docx2pdf 且系统有 Word/LibreOffice）: {e}")

            try:
                import fitz  # pymupdf
            except ImportError:
                raise RuntimeError("pymupdf 未安装，请执行 pip install pymupdf")

            doc = fitz.open(pdf_path)
            saved: List[str] = []
            for i, page in enumerate(doc):
                mat = fitz.Matrix(150 / 72, 150 / 72)
                pix = page.get_pixmap(matrix=mat)
                out_path = os.path.join(output_dir, f"page_{i + 1}.png")
                pix.save(out_path)
                saved.append(out_path)
            doc.close()
            return saved


def preprocess(docx_path: str, output_dir: Optional[str] = None, use_soffice: bool = True) -> Dict[str, Any]:
    """一键预处理 .docx，返回供模型使用的结构化摘要。

    Args:
        docx_path: .docx 文件路径。
        output_dir: 可选，图片输出目录。
        use_soffice: 是否使用 LibreOffice 转图片，默认 True。
                     False 时改用 docx2pdf + pymupdf。

    Returns:
        {
            "metadata": {...},
            "text": "...",        # pandoc 提取，失败则用 python-docx
            "headings": [...],
            "tables": [...],
            "images": [...],      # 仅在 output_dir 非空时存在
        }
    """
    result: Dict[str, Any] = {}

    # 元数据
    try:
        result["metadata"] = extract_metadata(docx_path)
    except Exception:
        result["metadata"] = {}

    # 正文文本（优先 pandoc）
    try:
        result["text"] = extract_text_pandoc(docx_path, track_changes="accept")
    except Exception:
        try:
            result["text"] = extract_text_python_docx(docx_path)
        except Exception:
            result["text"] = ""

    # 标题结构
    try:
        result["headings"] = extract_headings(docx_path)
    except Exception:
        result["headings"] = []

    # 表格
    try:
        result["tables"] = extract_tables(docx_path)
    except Exception:
        result["tables"] = []

    # 图片（可选）
    if output_dir:
        try:
            result["images"] = convert_to_images(docx_path, output_dir, use_soffice=use_soffice)
        except Exception:
            result["images"] = []

    return result
