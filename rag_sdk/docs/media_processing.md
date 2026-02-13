# Media Processing & OCR {#media-processing}

## Document Ingestion Pipeline

The SDK supports a comprehensive document ingestion pipeline that handles various file formats, including scanned PDFs and images via OCR.

### Supported Loaders

*   **Standard documents:** PDF (pdfplumber, pypdf), DOCX, TXT
*   **Images:** JPG, PNG, TIFF

### OCR Pipeline

The OCR engine is powered by a multi-stage pipeline:

1.  **Preprocessing:** Deskewing, denoising, binarization, contrast enhancement.
2.  **Layout Analysis:** Using models like LayoutLMv3 to detect headers, footers, columns, and sections.
3.  **Text Recognition:**
    *   **Primary:** PaddleOCR (supports 80+ languages)
    *   **Fallback:** Tesseract
    *   **Cloud:** Azure OCR, AWS Textract, Google Vision (optional)
4.  **Table Extraction:** Microsoft Table Transformer for preserving table structure.

### Configuration

```yaml
document_ingestion:
  loaders:
    ocr:
      enabled: true
      providers:
        primary: "paddleocr"
        fallback: "tesseract"
      
      paddleocr:
        lang: "en"
        use_angle_cls: true
        use_gpu: true
        
    image_preprocessing:
      enabled: true
      steps:
        - deskew: true
        - denoise: true
        - binarization: "adaptive"
```

## Multilingual Support

The SDK supports multilingual OCR with automatic language detection per page and per region.

```python
from rag_sdk.ocr import MultilingualOCR

ocr = MultilingualOCR(config)
pages = ocr.process_document("mixed_language.pdf")
```

## Handwriting Recognition

For documents with handwritten notes, the SDK uses a hybrid pipeline combining printed text OCR with specialized handwriting models (like Microsoft TrOCR).

```yaml
handwriting_recognition:
  enabled: true
  models:
    primary: "microsoft_trocr"
```

## Complex Layouts

We handle complex PDF layouts including:
*   Multi-column text detection
*   Table structure preservation
*   Reading order reconstruction using vision models
*   Formula detection (LaTeX OCR) 
