from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.split("\n"))
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def chunk_text(text: str, *, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_docx(path: str) -> str:
    from docx import Document

    d = Document(path)
    parts: List[str] = []
    for p in d.paragraphs:
        t = (p.text or "").strip()
        if t:
            parts.append(t)
    return "\n".join(parts)


def read_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PDF support requires pypdf. Install it in your venv: pip install pypdf"
        ) from e

    reader = PdfReader(path)
    pages: List[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        t = t.strip()
        if t:
            pages.append(t)
    return "\n\n".join(pages)


@dataclass
class Chunk:
    doc_id: str
    source_path: str
    chunk_id: int
    text: str


def iter_input_files(data_dir: str, patterns: Optional[List[str]] = None) -> Iterable[str]:
    if patterns is None:
        patterns = ["**/*.txt", "**/*.pdf", "**/*.docx"]
    for pat in patterns:
        for p in glob.glob(os.path.join(data_dir, pat), recursive=True):
            if os.path.isfile(p):
                yield p


def read_any(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        return read_txt(path)
    if ext == ".docx":
        return read_docx(path)
    if ext == ".pdf":
        return read_pdf(path)
    raise ValueError(f"Unsupported file type: {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", default="artifacts/chunks.jsonl")
    ap.add_argument("--chunk_size", type=int, default=900)
    ap.add_argument("--overlap", type=int, default=150)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    chunks: List[Chunk] = []
    for idx, path in enumerate(sorted(iter_input_files(args.data_dir))):
        raw = read_any(path)
        texts = chunk_text(raw, chunk_size=args.chunk_size, overlap=args.overlap)
        doc_id = f"doc_{idx}"
        for j, t in enumerate(texts):
            chunks.append(Chunk(doc_id=doc_id, source_path=path, chunk_id=j, text=t))

    with open(args.out, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    print(f"Wrote {len(chunks)} chunks to {args.out}")


if __name__ == "__main__":
    main()
