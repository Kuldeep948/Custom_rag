"""
PDF processing service.

Extraction strategy (tried in order):
  1. Text layer   — PyMuPDF direct text extraction (fast, free)
  2. pymupdf4llm  — Markdown extraction for PDFs with mixed content
  3. Gemini Vision OCR — page-by-page image OCR for fully scanned PDFs
                         (requires GEMINI_API_KEY)

Each page is chunked independently to preserve page-number metadata.
"""
import asyncio
import io
from pathlib import Path
from typing import List, Tuple

from app.core.config import settings
from app.core.logging import get_logger
from app.services.ingestion.chunker import TextChunk, chunk_text

logger = get_logger(__name__)


class PDFProcessor:
    """
    Extracts text from PDFs and returns a flat list of TextChunks.

    Automatically selects the best available extraction method:
      - If the PDF has a text layer → PyMuPDF direct extraction
      - If pymupdf4llm produces real content → use that
      - If the PDF is fully scanned → Gemini Vision OCR per page
    """

    # Minimum chars across the whole doc to consider text layer usable
    MIN_TEXT_CHARS = 200

    def process(self, file_path: str) -> Tuple[List[TextChunk], dict]:
        """
        Process a PDF and return (chunks, metadata).

        Args:
            file_path: Absolute or relative path to the PDF file.

        Returns:
            Tuple of (list[TextChunk], metadata_dict)
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        logger.info("pdf_processing_start", file=path.name, size_kb=path.stat().st_size // 1024)

        # ── Strategy 1: direct text layer ────────────────────────
        pages, meta = self._extract_text_layer(path)
        total_chars = sum(len(p.strip()) for p in pages)

        if total_chars >= self.MIN_TEXT_CHARS:
            logger.info("pdf_text_layer_found", chars=total_chars)
            meta["extraction_method"] = "pymupdf_text"
            return self._chunk_pages(pages, meta)

        # ── Strategy 2: pymupdf4llm markdown ─────────────────────
        try:
            pages_md, meta_md = self._extract_with_pymupdf4llm(path)
            total_md = sum(len(p.strip()) for p in pages_md)
            if total_md >= self.MIN_TEXT_CHARS:
                logger.info("pdf_pymupdf4llm_used", chars=total_md)
                meta_md["extraction_method"] = "pymupdf4llm"
                return self._chunk_pages(pages_md, meta_md)
        except Exception as e:
            logger.warning("pymupdf4llm_failed", error=str(e))

        # ── Strategy 3: Gemini Vision OCR ────────────────────────
        if settings.gemini_api_key:
            logger.info("pdf_scanned_using_gemini_ocr", pages=meta.get("page_count", 0))
            try:
                pages_ocr, meta_ocr = self._extract_with_gemini_ocr(path)
                total_ocr = sum(len(p.strip()) for p in pages_ocr)
                if total_ocr >= self.MIN_TEXT_CHARS:
                    meta_ocr["extraction_method"] = "gemini_vision_ocr"
                    return self._chunk_pages(pages_ocr, meta_ocr)
            except Exception as e:
                logger.error("gemini_ocr_failed", error=str(e))
        else:
            logger.warning(
                "pdf_scanned_no_ocr_key",
                message="Set GEMINI_API_KEY to enable OCR for scanned PDFs",
            )

        # ── Fallback: structural metadata chunks ──────────────────
        logger.warning("pdf_fallback_metadata_chunks", file=path.name)
        return self._metadata_chunks(path, meta)

    # ── Extraction methods ────────────────────────────────────────────────────

    def _extract_text_layer(self, path: Path) -> Tuple[List[str], dict]:
        """Extract text directly from the PDF text layer using PyMuPDF."""
        import fitz

        doc = fitz.open(str(path))
        pages = [page.get_text("text") for page in doc]
        meta = {
            "page_count": doc.page_count,
            "title":  doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
        }
        doc.close()
        return pages, meta

    def _extract_with_pymupdf4llm(self, path: Path) -> Tuple[List[str], dict]:
        """
        Use pymupdf4llm to extract markdown per page.
        Works well for PDFs with mixed text + images.
        """
        import fitz
        import pymupdf4llm

        doc = fitz.open(str(path))
        page_count = doc.page_count
        doc.close()

        pages = []
        for i in range(page_count):
            md = pymupdf4llm.to_markdown(str(path), pages=[i])
            # Strip picture-placeholder lines — they add no semantic value
            lines = [
                ln for ln in md.splitlines()
                if "intentionally omitted" not in ln
            ]
            pages.append("\n".join(lines).strip())

        meta = {"page_count": page_count}
        return pages, meta

    def _extract_with_gemini_ocr(self, path: Path) -> Tuple[List[str], dict]:
        """
        Render each page as a JPEG image and OCR it with Gemini Vision.
        Used for fully scanned PDFs that have no text layer.
        Retries with exponential backoff on rate-limit errors (429).
        Falls back to gemini-2.0-flash-lite if primary model quota is exhausted.
        """
        import time
        import fitz
        import google.generativeai as genai
        from PIL import Image

        genai.configure(api_key=settings.gemini_api_key)

        # Try primary model first, fall back to lite on quota errors
        models_to_try = ["models/gemini-2.0-flash", "models/gemini-2.0-flash-lite"]
        active_model_name = models_to_try[0]
        model = genai.GenerativeModel(active_model_name)

        OCR_PROMPT = (
            "You are an OCR engine. Extract ALL text from this image exactly "
            "as it appears. Preserve headings, bullet points, tables, and code "
            "blocks using markdown formatting. "
            "Return only the extracted text — no commentary, no preamble."
        )

        doc = fitz.open(str(path))
        pages: List[str] = []

        for page_num, page in enumerate(doc, start=1):
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img = Image.open(io.BytesIO(pix.tobytes("jpeg")))

            logger.debug("gemini_ocr_page", page=page_num, total=doc.page_count,
                         model=active_model_name)

            # Retry loop with backoff
            page_text = ""
            for attempt in range(3):
                try:
                    response  = model.generate_content([OCR_PROMPT, img])
                    page_text = response.text.strip()
                    break
                except Exception as e:
                    err_str = str(e)
                    # Rate limit — wait the suggested retry delay
                    if "429" in err_str or "quota" in err_str.lower():
                        # Extract retry_delay seconds from error if present
                        import re
                        m = re.search(r"retry in (\d+)", err_str, re.IGNORECASE)
                        wait = int(m.group(1)) + 2 if m else 30
                        logger.warning("gemini_ocr_rate_limit",
                                       page=page_num, wait_s=wait, attempt=attempt+1)
                        # Switch to lite model on second attempt
                        if attempt == 1 and len(models_to_try) > 1:
                            active_model_name = models_to_try[1]
                            model = genai.GenerativeModel(active_model_name)
                            logger.info("gemini_ocr_model_fallback",
                                        model=active_model_name)
                        time.sleep(wait)
                    else:
                        logger.error("gemini_ocr_page_failed",
                                     page=page_num, error=err_str)
                        break

            pages.append(page_text)

        meta = {"page_count": doc.page_count, "ocr_model": active_model_name}
        doc.close()
        return pages, meta

    def _metadata_chunks(
        self, path: Path, meta: dict
    ) -> Tuple[List[TextChunk], dict]:
        """
        Last-resort fallback: create one structural chunk per page
        describing the page dimensions and image count.
        Used when no OCR key is available and the PDF has no text layer.
        """
        import fitz

        doc = fitz.open(str(path))
        chunks: List[TextChunk] = []

        for page_num, page in enumerate(doc, start=1):
            imgs  = page.get_images(full=True)
            rect  = page.rect
            content = (
                f"Document: {path.name}\n"
                f"Page: {page_num} of {doc.page_count}\n"
                f"Dimensions: {int(rect.width)}×{int(rect.height)} pt\n"
                f"Visual elements: {len(imgs)} image(s)\n"
                f"[Scanned page — no text layer available]"
            )
            chunks.append(
                TextChunk(
                    content=content,
                    chunk_index=page_num - 1,
                    token_count=max(1, len(content) // 4),
                    page_number=page_num,
                    metadata={"chunk_type": "scanned_page"},
                )
            )

        doc.close()
        meta["extraction_method"] = "metadata_fallback"
        meta["chunk_count"] = len(chunks)
        return chunks, meta

    # ── Chunking ──────────────────────────────────────────────────────────────

    def _chunk_pages(
        self, pages: List[str], meta: dict
    ) -> Tuple[List[TextChunk], dict]:
        """Chunk each page's text independently, preserving page numbers."""
        all_chunks: List[TextChunk] = []
        global_index = 0

        for page_num, page_text in enumerate(pages, start=1):
            if not page_text or not page_text.strip():
                continue
            for chunk in chunk_text(page_text, page_number=page_num):
                chunk.chunk_index = global_index
                all_chunks.append(chunk)
                global_index += 1

        meta["chunk_count"] = len(all_chunks)
        meta["page_count"]  = len(pages)

        logger.info(
            "pdf_processing_complete",
            pages=len(pages),
            chunks=len(all_chunks),
            method=meta.get("extraction_method"),
        )
        return all_chunks, meta
