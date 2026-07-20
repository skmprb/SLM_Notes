"""
Extractor: PDF / DOCX / images → unified DocPayload
Handles: text, tables (→ markdown), images & graphs (→ base64)
"""
import base64
import io
from pathlib import Path

from unstructured.partition.auto import partition
from unstructured.documents.elements import (
    Table, Image, NarrativeText, Title, ListItem, FigureCaption
)
from PIL import Image as PILImage

from app.agents.state import DocPayload


def _to_markdown_table(html_table: str) -> str:
    """Convert unstructured HTML table to markdown."""
    try:
        import pandas as pd
        tables = pd.read_html(html_table)
        if tables:
            return tables[0].to_markdown(index=False)
    except Exception:
        pass
    return html_table


def _image_to_base64(image_bytes: bytes) -> str:
    img = PILImage.open(io.BytesIO(image_bytes))
    # Resize large images to reduce token cost (max 1024px wide)
    if img.width > 1024:
        ratio = 1024 / img.width
        img = img.resize((1024, int(img.height * ratio)))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def extract(file_path: str | Path) -> DocPayload:
    """
    Extract all content from a document file.
    Returns a DocPayload with text, tables, images, and metadata.
    """
    path = Path(file_path)
    elements = partition(
        filename=str(path),
        strategy="hi_res",              # best quality for tables + images
        infer_table_structure=True,
        extract_images_in_pdf=True,
    )

    text_parts: list[str] = []
    tables: list[str] = []
    images: list[str] = []
    page_numbers: set[int] = set()

    for el in elements:
        # Track pages
        if hasattr(el, "metadata") and el.metadata.page_number:
            page_numbers.add(el.metadata.page_number)

        if isinstance(el, Table):
            md = _to_markdown_table(el.metadata.text_as_html or str(el))
            tables.append(md)
            text_parts.append(f"[TABLE]\n{md}\n[/TABLE]")

        elif isinstance(el, Image):
            if el.metadata.image_base64:
                images.append(el.metadata.image_base64)
            elif el.metadata.image_path:
                raw = Path(el.metadata.image_path).read_bytes()
                images.append(_image_to_base64(raw))
            caption = str(el).strip()
            if caption:
                text_parts.append(f"[IMAGE: {caption}]")

        elif isinstance(el, (NarrativeText, Title, ListItem, FigureCaption)):
            text_parts.append(str(el).strip())

    return DocPayload(
        text="\n\n".join(filter(None, text_parts)),
        tables=tables,
        images=images,
        page_count=max(page_numbers, default=1),
        doc_name=path.name,
    )
