from pathlib import Path
from collections import defaultdict
def mean(lst): return sum(lst) / len(lst)
def median(lst):
    s = sorted(lst); n = len(s)
    return (s[n // 2] + s[~(n // 2)]) / 2

try:
    import fitz  # PyMuPDF for pdf
except ImportError:
    fitz = None

try:
    from pptx import Presentation
except ImportError:
    Presentation = None

try:
    from docx import Document
except ImportError:
    Document = None


def count_pages(path: Path) -> int | None:
    ext = path.suffix.lower()
    try:
        if ext == ".pdf" and fitz:
            with fitz.open(str(path)) as doc:
                return doc.page_count
        if ext == ".pptx" and Presentation:
            return len(Presentation(str(path)).slides)
        if ext == ".docx" and Document:
            doc = Document(str(path))
            page_breaks = sum(
                1 for p in doc.paragraphs
                for run in p.runs
                if "<w:lastRenderedPageBreak" in run._r.xml or ('<w:br ' in run._r.xml and 'type="page"' in run._r.xml)
            )
            return page_breaks + 1
    except Exception:
        pass
    return None


output_dir = Path(__file__).parent / "output"
INCLUDE_EXTS = {".pdf", ".pptx", ".docx", ".xlsx"}
PAGE_EXTS = {".pdf", ".pptx", ".docx"}

sizes: dict[str, list[float]] = defaultdict(list)
pages: dict[str, list[int]] = defaultdict(list)

for input_dir in output_dir.glob("*/question_tmp/input"):
    for f in input_dir.iterdir():
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        if ext not in INCLUDE_EXTS:
            continue
        mb = f.stat().st_size / 1024 / 1024
        sizes[ext].append(mb)
        if ext in PAGE_EXTS:
            n = count_pages(f)
            if n is not None:
                pages[ext].append(n)

all_exts = sorted(sizes.keys())
print(f"{'ext':<8} {'count':>6} {'avg_mb':>10} {'med_mb':>10} {'max_mb':>10} {'avg_pages':>10} {'med_pages':>10} {'max_pages':>10}")
print("-" * 80)
for ext in all_exts:
    s = sizes[ext]
    avg_mb = mean(s)
    med_mb = median(s)
    max_mb = max(s)
    if ext in PAGE_EXTS and pages.get(ext):
        p = pages[ext]
        avg_p = f"{mean(p):.1f}"
        med_p = f"{median(p):.1f}"
        max_p = str(max(p))
    else:
        avg_p = med_p = max_p = "-"
    print(f"{ext:<8} {len(s):>6} {avg_mb:>10.2f} {med_mb:>10.2f} {max_mb:>10.2f} {avg_p:>10} {med_p:>10} {max_p:>10}")
