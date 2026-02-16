# PDF Parser Research: Models and Architecture Comparison

## Overview

The rag_sdk supports two PDF parsing backends with fundamentally different approaches:

| | **PyMuPDF** (default) | **Docling** (optional) |
|---|---|---|
| **Approach** | Deterministic geometry | ML-powered layout analysis |
| **Install** | Base install | `pip install rag-sdk[docling]` |
| **Speed** | Sub-second per page | ~2-20s per page (model inference) |
| **Best for** | Simple digital PDFs, grid-bordered tables | Complex layouts, borderless tables, scanned PDFs |
| **Dependencies** | PyMuPDF (fitz) only | torch, transformers, docling-ibm-models, OCR engines |

---

## PyMuPDF Backend — Deterministic Geometry Pipeline

**No ML models.** Pure algorithmic extraction using PyMuPDF (fitz) as the C library.

### Pipeline per page

1. **Text extraction** — `page.get_text("dict")` returns spans with bbox, font name, font size, bold/italic flags
2. **Line clustering** — Y-proximity clustering groups TextSpans into TextLines (configurable `line_y_tolerance`)
3. **Segment extraction** — `page.get_cdrawings()` returns vector line segments from drawing commands; rectangles decomposed into 4 segments
4. **Checkbox detection** — Small square rects (6-24pt) with optional marks inside classified as checked/unchecked
5. **Table grid detection** — Intersection-point algorithm: find where horizontal and vertical segments cross, snap nearby intersections, build rectangular grid cells, validate minimum row/col counts
6. **Text-to-cell mapping** — For each grid cell bbox, find TextSpans whose center falls inside, concatenate per cell
7. **Reading order** — Top-to-bottom, left-to-right sort with Y-clustering for row detection

### Strengths

- Zero ML overhead, sub-second parsing
- Accurate font info (name, size, bold, italic) per span
- Checkbox detection (unique to this backend)
- No external model downloads

### Known Limitations

- **Multi-column layouts**: Y-clustering intermingles columns (text interleaves by Y-position)
- **Borderless/ruled tables**: No vertical grid lines = no intersections = no table detection
- **Scanned PDFs**: No embedded text, requires OCR
- **Complex academic layouts**: Headers, footers, figures, captions can confuse reading order

---

## Docling Backend — ML-Powered Document Understanding

**IBM Docling** (MIT-licensed, open-source). Uses multiple ML models in a pipeline.

### Pipeline per document

1. **PDF parsing** — `docling-parse` (C++ library) extracts raw text, paths, and bitmap images with coordinates
2. **Layout analysis** — Object detection model identifies page elements (paragraphs, tables, figures, section headers, captions, etc.)
3. **Table structure** — TableFormer model recognizes table structure (rows, columns, spanning cells) from detected table regions
4. **OCR** (optional) — For scanned pages or image-based content, runs OCR on regions without embedded text
5. **Reading order** — ML-inferred document structure provides natural reading order (handles multi-column correctly)
6. **Assembly** — `DoclingDocument` combines all elements with provenance (bounding box, page number)

### ML Models in Detail

#### 1. Layout Analysis — RT-DETR (Real-Time Detection Transformer)

The core model that identifies and localizes page elements.

| Variant | HuggingFace Repo | Size | Use Case |
|---------|-------------------|------|----------|
| **Heron** (default) | `docling-project/docling-layout-heron` | Base | Balanced accuracy/speed for general documents |
| **Heron-101** | `docling-project/docling-layout-heron-101` | Base | Enhanced version with improved accuracy |
| **Egret Medium** | `docling-project/docling-layout-egret-medium` | Medium | Higher accuracy for complex layouts |
| **Egret Large** | `docling-project/docling-layout-egret-large` | Large | Complex documents needing precision |
| **Egret XLarge** | `docling-project/docling-layout-egret-xlarge` | XLarge | Maximum accuracy, highest resource usage |

**Default: Heron.** Detects: paragraphs, section headers, tables, figures, captions, lists, page headers/footers, footnotes, formulas.

**Hardware**: CPU, CUDA, MPS (Apple Silicon), XPU (Intel).

#### 2. Table Structure — TableFormer

State-of-the-art table structure recognition model developed by IBM/Docling team.

| Mode | Speed | Accuracy | Use Case |
|------|-------|----------|----------|
| **ACCURATE** (default) | Slower | Higher | Complex tables, spanning cells, borderless tables |
| **FAST** | Faster | Lower | Simple tables, high-volume processing |

- **Architecture**: Transformer-based encoder-decoder, trained on PubTabNet dataset
- **Representation**: OTSL (Optimized Table Structure Language) — outperforms HTML representation for complex tables
- **Features**: Cell matching (aligns detected cells with PDF text), handles merged cells, borderless tables, ruled tables
- **HuggingFace**: `docling-project/docling-models` (v2.3.0, path `model_artifacts/tableformer`)
- **Input**: Images at 144 DPI (2x upscaling from 72 DPI base)

**Key advantage over PyMuPDF**: TableFormer detects table structure from visual appearance, not vector line intersections. This means it works on borderless tables, ruled tables (horizontal lines only), and even tables in scanned images.

#### 3. OCR Engines

Docling supports multiple OCR engines with automatic platform-based selection:

| Engine | Type | Platform | Default Languages | Notes |
|--------|------|----------|-------------------|-------|
| **OcrMac** | Apple Vision framework | macOS only | en-US, fr-FR, de-DE, es-ES | Native, fast, auto-selected on macOS |
| **EasyOCR** | Deep learning (CRAFT + CRNN) | Cross-platform | en, fr, de, es | 80+ languages, GPU-accelerated |
| **RapidOCR** | PP-OCRv4 (PaddleOCR) | Cross-platform | CJK + multilingual | Lightweight, multiple backends (ONNX, OpenVINO, Paddle, Torch) |
| **Tesseract** | Traditional OCR | Cross-platform | eng, fra, deu, spa | Requires system install, CLI or Python bindings |

**Auto mode** (default): Selects the best available engine for the platform. On macOS, this is OcrMac.

OCR is applied selectively — only to page regions where no embedded text is found, making it efficient for mixed digital/scanned documents.

#### 4. Picture Classification

- **Model**: `docling-project/DocumentFigureClassifier-v2.0`
- **Purpose**: Classifies detected pictures into categories (photos, diagrams, charts, etc.)
- Not used in our parser (we extract text and tables only)

#### 5. Code & Formula Enrichment (Optional)

- **Model**: `docling-project/CodeFormulaV2`
- **Purpose**: Specialized extraction of code blocks and mathematical formulas
- Enabled via `do_code_enrichment=True` / `do_formula_enrichment=True`
- Not enabled in our default config

#### 6. Vision-Language Models (Optional, for VLM-based conversion)

Docling also supports full VLM-based document conversion as an alternative to the standard pipeline:

| Model | Parameters | Source | Format |
|-------|-----------|--------|--------|
| **Granite-Docling** | 258M | IBM | DocTags |
| **SmolDocling** | 256M | Docling team | DocTags |
| Granite-Vision | 2B | IBM | Markdown |
| Pixtral | 12B | Mistral | Markdown |
| Qwen2.5-VL | 3B | Alibaba | Markdown |
| DeepSeek-OCR | 3B | DeepSeek | Markdown |
| GOT-OCR 2.0 | - | StepFun | Markdown |
| Phi-4 | - | Microsoft | Markdown |

These are **not used** in our standard pipeline — they're an alternative approach where a VLM converts the entire page image to structured text directly. The standard pipeline (layout model + TableFormer + OCR) is more reliable for structured extraction.

---

## Comparison on RAG_original.pdf Page 6

This page has a two-column layout with two borderless tables (Table 1 and Table 2) and body text.

| Aspect | PyMuPDF | Docling |
|--------|---------|---------|
| **Tables detected** | 0 (no grid borders = no intersections) | 2 (TableFormer detects from visual layout) |
| **Column handling** | Interleaved (Y-sort mixes left/right columns) | Correct reading order per column |
| **Section headers** | Detected via bold font heuristic | Detected via layout model (`SectionHeaderItem`) |
| **Table captions** | Mixed into body text | Separate elements with correct association |
| **Checkboxes** | Supported | Not supported (no native detection) |
| **Processing time** | <0.5s | ~20s (cached models), ~166s (first run, model download) |
| **Font info** | Full (name, size, bold, italic per span) | Label-level only (bold for headers, no font details) |

### Recommendation

- **Use PyMuPDF** for: simple digital PDFs, forms with checkboxes, high-throughput processing, when speed matters
- **Use Docling** for: academic papers, multi-column layouts, borderless/ruled tables, scanned PDFs, complex document structures

---

## References

- [Docling GitHub](https://github.com/docling-project/docling) — IBM, MIT License
- [Docling Technical Report](https://arxiv.org/abs/2408.09869) — Describes the architecture and models
- [TableFormer](https://arxiv.org/abs/2203.01017) — Table structure recognition paper
- [RT-DETR](https://arxiv.org/abs/2304.08069) — Real-time object detection transformer
- [PyMuPDF](https://pymupdf.readthedocs.io/) — C library for PDF rendering and text extraction
