"""Step 1: load documents, keeping the file name and page number of every piece of text."""
from dataclasses import dataclass
from pathlib import Path

TEXT_EXTS = {".txt", ".md"}


@dataclass
class Page:
    text: str
    source: str  # file name
    page: int | None  # 1-based page for PDFs, None for plain text files


def load(folder: str | Path) -> list[Page]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} is not a folder")
    pages: list[Page] = []
    for path in sorted(folder.rglob("*")):
        ext = path.suffix.lower()
        if ext == ".pdf":
            from pypdf import PdfReader

            for i, pdf_page in enumerate(PdfReader(path).pages, start=1):
                text = (pdf_page.extract_text() or "").strip()
                if text:
                    pages.append(Page(text, path.name, i))
        elif ext in TEXT_EXTS:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                pages.append(Page(text, path.name, None))
    return pages
