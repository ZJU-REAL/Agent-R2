"""data_sythesis/preprocess -- 文件预处理模块。

各子模块分别处理对应格式，对外统一暴露 preprocess() 入口：

    from data_sythesis.preprocess import pdf, docx, xlsx, pptx

    result = pdf.preprocess("report.pdf")
    result = docx.preprocess("doc.docx")
    result = xlsx.preprocess("sheet.xlsx")
    result = pptx.preprocess("deck.pptx")
"""
from . import docx, pdf, pptx, xlsx

__all__ = ["pdf", "docx", "xlsx", "pptx"]
