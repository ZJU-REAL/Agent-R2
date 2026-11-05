"""data_sythesis/preprocess/pdf.py -- PDF 文件预处理，帮助模型理解 PDF 内容。

依赖:
    pip install pypdf pdfplumber pdf2image pytesseract
    apt-get install poppler-utils tesseract-ocr
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


def extract_text(pdf_path: str) -> str:
    """提取 PDF 全文（含布局）。

    优先使用 pdfplumber（保留空白布局），失败时回退到 pypdf。

    Args:
        pdf_path: PDF 文件路径。

    Returns:
        提取的文本字符串。
    """
    try:
        import pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            pages_text: List[str] = []
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages_text.append(f"[Page {i + 1}]\n{text}")
            return "\n\n".join(pages_text)
    except Exception:
        pass

    # 回退：pypdf
    try:
        from pypdf import PdfReader

        reader = PdfReader(pdf_path)
        pages_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages_text.append(f"[Page {i + 1}]\n{text}")
        return "\n\n".join(pages_text)
    except Exception as exc:
        raise RuntimeError(f"无法提取 PDF 文本: {exc}") from exc


def extract_tables(pdf_path: str) -> List[List[List[Optional[str]]]]:
    """提取 PDF 中所有表格数据。

    Returns:
        每个元素是一个页面的表格列表；表格本身为二维列表（行 × 列）。
        结构：[page_tables, ...] 其中 page_tables = [table, ...]，
        table = [[cell, ...], ...]
    """
    try:
        import pdfplumber

        all_tables: List[List[List[Optional[str]]]] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                all_tables.extend(tables or [])
        return all_tables
    except Exception as exc:
        raise RuntimeError(f"无法提取 PDF 表格: {exc}") from exc


def extract_metadata(pdf_path: str) -> Dict[str, Any]:
    """提取 PDF 元数据（标题、作者、页数等）。

    Returns:
        包含 title、author、subject、creator、num_pages 的字典。
    """
    try:
        from pypdf import PdfReader

        reader = PdfReader(pdf_path)
        meta = reader.metadata or {}
        return {
            "title": meta.title or "",
            "author": meta.author or "",
            "subject": meta.subject or "",
            "creator": meta.creator or "",
            "num_pages": len(reader.pages),
        }
    except Exception as exc:
        raise RuntimeError(f"无法提取 PDF 元数据: {exc}") from exc


def convert_to_images(
    pdf_path: str,
    output_dir: str,
    max_dim: int = 1000,
    dpi: int = 200,
) -> List[str]:
    """将 PDF 每页转换为 PNG 图片，便于视觉理解。

    Args:
        pdf_path: PDF 文件路径。
        output_dir: 图片输出目录。
        max_dim: 图片最大边长（像素），超出则等比缩放。
        dpi: 渲染分辨率。

    Returns:
        生成的图片路径列表（按页码顺序）。
    """
    from pdf2image import convert_from_path

    os.makedirs(output_dir, exist_ok=True)
    images = convert_from_path(pdf_path, dpi=dpi)
    saved: List[str] = []

    for i, image in enumerate(images):
        w, h = image.size
        if w > max_dim or h > max_dim:
            scale = min(max_dim / w, max_dim / h)
            image = image.resize((int(w * scale), int(h * scale)))
        out_path = os.path.join(output_dir, f"page_{i + 1}.png")
        image.save(out_path)
        saved.append(out_path)

    return saved


def ocr_text(pdf_path: str, lang: str = "eng") -> str:
    """对扫描版 PDF 进行 OCR，提取文字。

    需要安装 tesseract-ocr。

    Args:
        pdf_path: PDF 文件路径。
        lang: tesseract 语言代码，中文为 "chi_sim"，英文为 "eng"。

    Returns:
        OCR 识别出的文本。
    """
    try:
        import pytesseract
        from pdf2image import convert_from_path

        images = convert_from_path(pdf_path, dpi=200)
        pages_text: List[str] = []
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image, lang=lang)
            if text.strip():
                pages_text.append(f"[Page {i + 1}]\n{text}")
        return "\n\n".join(pages_text)
    except Exception as exc:
        raise RuntimeError(f"OCR 失败: {exc}") from exc


def extract_text_with_pdftotext(pdf_path: str, layout: bool = True) -> str:
    """使用系统工具 pdftotext（poppler-utils）提取文本，布局保真度更高。

    Args:
        pdf_path: PDF 文件路径。
        layout: True 则保留原始排版（-layout 选项）。

    Returns:
        提取的文本字符串。
    """
    import subprocess

    args = ["pdftotext"]
    if layout:
        args.append("-layout")
    args += [pdf_path, "-"]

    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"pdftotext 失败: {result.stderr}")
    return result.stdout


def preprocess(pdf_path: str, output_dir: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """一键预处理 PDF，返回供模型使用的结构化摘要。

    提取元数据、正文文本和表格，若正文为空则尝试 OCR。
    若提供 output_dir，还会将每页转成图片。

    Args:
        pdf_path: PDF 文件路径。
        output_dir: 可选，图片输出目录。

    Returns:
        {
            "metadata": {...},
            "text": "...",
            "tables": [...],
            "images": [...],   # 仅在 output_dir 非空时存在
        }
    """
    result: Dict[str, Any] = {}

    # 元数据
    try:
        result["metadata"] = extract_metadata(pdf_path)
    except Exception:
        result["metadata"] = {}

    # 正文文本
    try:
        text = extract_text(pdf_path)
    except Exception:
        text = ""

    if not text.strip():
        # 可能是扫描件，尝试 OCR
        try:
            text = ocr_text(pdf_path)
        except Exception:
            text = ""

    result["text"] = text

    # 表格
    try:
        result["tables"] = extract_tables(pdf_path)
    except Exception:
        result["tables"] = []

    # 图片（可选）
    if output_dir:
        try:
            result["images"] = convert_to_images(pdf_path, output_dir)
        except Exception:
            result["images"] = []

    return result
