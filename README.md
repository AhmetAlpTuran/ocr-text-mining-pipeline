# Turkish PDF Text Mining Pipeline

This repo contains a local text mining pipeline for Turkish PDF books. It extracts sentences, finds keyword roots, applies contextual rules, and produces CSV + DOCX reports. It also includes an OCR helper for scanned PDFs.

## Files

- `text_mining_pipeline.py`  
  Main pipeline: reads PDFs, finds keyword roots, classifies context with Ollama, writes a full CSV and a summary DOCX.
- `ocr_if_no_text.py`  
  OCR helper: scans PDFs and OCRs only those with no extractable text, overwriting the original PDF.
- `export_summary_to_docx.py`  
  Converts a summary CSV to a DOCX table (standalone utility).
- `review_summary.py`  
  Older summary generator (CSV). Kept for reference.
- `kelimeler.txt`  
  Keyword roots list (one per line).

## Folders

- `kitaplar/`  
  Place PDFs here for batch processing.
- `Çıktılar/`  
  Full CSV outputs (one per PDF).
- `özet/`  
  Summary DOCX outputs (one per PDF).

## Requirements

Python packages:
```
pip install pdfplumber nltk pandas tqdm ollama python-docx ocrmypdf
```

System tools (for OCR):
- Tesseract OCR (with Turkish language data `tur`)
- Ghostscript

## Quick Start

1) Optional: OCR scanned PDFs
```
python ocr_if_no_text.py
```

2) Run the pipeline on all PDFs in `kitaplar/`
```
python text_mining_pipeline.py
```

## Single PDF

OCR a single PDF:
```
python ocr_if_no_text.py --pdf "kitaplar\\book.pdf"
```

Process a single PDF:
```
python text_mining_pipeline.py --pdf "kitaplar\\book.pdf"
```

## Outputs

For each PDF named `book.pdf`:
- Full CSV: `Çıktılar\\book_çıktı.csv`
- Summary DOCX: `özet\\book_özet.docx`

The summary table has:
- `keyword_root`
- `count`
- `category` (Çağ ayrımı, Gregoryen Miladı, Roma Rakamları, Olgusal Kavramlar, Bağlamsal Kavramlar)

## Notes

- Sentence tokenization uses NLTK (`punkt` + `punkt_tab`).
- Contextual rules apply only to specified terms; for them, only matches that pass the contextual filter appear in the summary.
- If Ollama is not reachable, the pipeline continues and marks LLM output as unavailable.
