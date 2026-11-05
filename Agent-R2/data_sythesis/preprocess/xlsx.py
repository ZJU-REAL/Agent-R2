"""data_sythesis/preprocess/xlsx.py -- Excel 文件预处理，帮助模型理解 .xlsx 内容。

skills 中没有独立的 xlsx skill，此处基于常用 Python 库实现：
  - openpyxl：读取工作表、单元格、公式、命名区域
  - pandas：表格数据转 Markdown / CSV 文本

依赖:
    pip install openpyxl pandas
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def list_sheets(xlsx_path: str) -> List[str]:
    """列出 .xlsx 中所有工作表名称。

    Returns:
        工作表名称列表，顺序与文件一致。
    """
    import openpyxl

    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    names = wb.sheetnames
    wb.close()
    return names


def extract_sheet_data(
    xlsx_path: str,
    sheet_name: Optional[str] = None,
    max_rows: int = 500,
    max_cols: int = 50,
) -> List[List[Any]]:
    """读取指定工作表的单元格数据（二维列表）。

    Args:
        xlsx_path: .xlsx 文件路径。
        sheet_name: 工作表名；None 则读取第一个活动表。
        max_rows: 最多读取的行数，防止超大表格撑爆上下文。
        max_cols: 最多读取的列数。

    Returns:
        二维列表，[row][col] 为单元格值（字符串化）。
    """
    import openpyxl

    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb[sheet_name] if sheet_name else wb.active

    rows: List[List[Any]] = []
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i >= max_rows:
            break
        cells = [str(c) if c is not None else "" for c in row[:max_cols]]
        rows.append(cells)

    wb.close()
    return rows


def extract_all_sheets(
    xlsx_path: str,
    max_rows: int = 500,
    max_cols: int = 50,
) -> Dict[str, List[List[Any]]]:
    """读取 .xlsx 中所有工作表数据。

    Returns:
        {sheet_name: [[cell, ...], ...], ...}
    """
    sheets = list_sheets(xlsx_path)
    return {
        name: extract_sheet_data(xlsx_path, name, max_rows=max_rows, max_cols=max_cols)
        for name in sheets
    }


def sheet_to_markdown(
    rows: List[List[Any]],
    has_header: bool = True,
) -> str:
    """将二维列表转为 Markdown 表格字符串，便于放入 prompt。

    Args:
        rows: extract_sheet_data 返回的二维列表。
        has_header: 第一行是否作为表头。

    Returns:
        Markdown 表格文本。
    """
    if not rows:
        return ""

    def _row_str(cells: List[Any]) -> str:
        return "| " + " | ".join(str(c).replace("|", "\\|") for c in cells) + " |"

    lines: List[str] = []
    if has_header and len(rows) >= 1:
        lines.append(_row_str(rows[0]))
        lines.append("| " + " | ".join(["---"] * len(rows[0])) + " |")
        data_rows = rows[1:]
    else:
        data_rows = rows

    for row in data_rows:
        lines.append(_row_str(row))

    return "\n".join(lines)


def extract_metadata(xlsx_path: str) -> Dict[str, Any]:
    """提取 .xlsx 文件属性（标题、作者、工作表数量等）。

    Returns:
        包含文档属性的字典。
    """
    import openpyxl

    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    props = wb.properties
    metadata = {
        "title": props.title or "",
        "subject": props.subject or "",
        "creator": props.creator or "",
        "description": props.description or "",
        "created": str(props.created) if props.created else "",
        "modified": str(props.modified) if props.modified else "",
        "sheet_count": len(wb.sheetnames),
        "sheet_names": wb.sheetnames,
    }
    wb.close()
    return metadata


def extract_named_ranges(xlsx_path: str) -> Dict[str, str]:
    """提取 .xlsx 中定义的命名区域（Named Ranges）。

    Returns:
        {name: range_string, ...}
    """
    import openpyxl

    wb = openpyxl.load_workbook(xlsx_path, read_only=False, data_only=True)
    ranges: Dict[str, str] = {}
    for name, defn in wb.defined_names.items():
        ranges[name] = str(defn.value)
    wb.close()
    return ranges


def extract_sheet_summary(
    xlsx_path: str,
    sheet_name: Optional[str] = None,
    max_rows: int = 500,
) -> Dict[str, Any]:
    """获取单个工作表的摘要信息（维度、非空行数、列名）。

    Returns:
        {"sheet_name": ..., "dimensions": ..., "non_empty_rows": ..., "columns": [...]}
    """
    import openpyxl

    wb = openpyxl.load_workbook(xlsx_path, read_only=False, data_only=True)
    ws = wb[sheet_name] if sheet_name else wb.active
    actual_name = ws.title

    dimensions = ws.dimensions  # e.g. "A1:Z100"
    non_empty = sum(
        1 for row in ws.iter_rows(values_only=True)
        if any(c is not None for c in row)
    )

    # 取第一行作为列名
    columns: List[str] = []
    for row in ws.iter_rows(min_row=1, max_row=1, values_only=True):
        columns = [str(c) if c is not None else "" for c in row]
        break

    wb.close()
    return {
        "sheet_name": actual_name,
        "dimensions": dimensions,
        "non_empty_rows": non_empty,
        "columns": columns,
    }


def preprocess(
    xlsx_path: str,
    max_rows_per_sheet: int = 200,
    max_cols: int = 30,
) -> Dict[str, Any]:
    """一键预处理 .xlsx，返回供模型使用的结构化摘要。

    对每个工作表提取摘要和 Markdown 表格文本（行数限制），
    避免超大文件把 prompt 撑爆。

    Args:
        xlsx_path: .xlsx 文件路径。
        max_rows_per_sheet: 每个工作表最多放入 prompt 的行数。
        max_cols: 每个工作表最多放入 prompt 的列数。

    Returns:
        {
            "metadata": {...},
            "named_ranges": {...},
            "sheets": {
                sheet_name: {
                    "summary": {...},
                    "markdown": "...",   # Markdown 表格
                    "data": [[...]],     # 原始二维列表
                },
                ...
            }
        }
    """
    result: Dict[str, Any] = {}

    # 元数据
    try:
        result["metadata"] = extract_metadata(xlsx_path)
    except Exception:
        result["metadata"] = {}

    # 命名区域
    try:
        result["named_ranges"] = extract_named_ranges(xlsx_path)
    except Exception:
        result["named_ranges"] = {}

    # 各工作表
    sheets: Dict[str, Any] = {}
    try:
        sheet_names = list_sheets(xlsx_path)
    except Exception:
        sheet_names = []

    for name in sheet_names:
        sheet_info: Dict[str, Any] = {}

        try:
            sheet_info["summary"] = extract_sheet_summary(xlsx_path, name)
        except Exception:
            sheet_info["summary"] = {}

        try:
            data = extract_sheet_data(
                xlsx_path, name,
                max_rows=max_rows_per_sheet,
                max_cols=max_cols,
            )
            sheet_info["data"] = data
            sheet_info["markdown"] = sheet_to_markdown(data)
        except Exception:
            sheet_info["data"] = []
            sheet_info["markdown"] = ""

        sheets[name] = sheet_info

    result["sheets"] = sheets
    return result
