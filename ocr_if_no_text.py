import argparse
import re
import subprocess
import sys
from pathlib import Path

import pdfplumber


def has_extractable_text(
    pdf_path: Path,
    min_chars: int,
    min_alpha_ratio: float,
) -> bool:
    total_chars = 0
    total_alpha = 0
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                normalized = re.sub(r"\s+", "", text)
                if not normalized:
                    continue
                total_chars += len(normalized)
                total_alpha += sum(1 for ch in normalized if ch.isalpha())
                if total_chars >= min_chars:
                    ratio = total_alpha / total_chars if total_chars else 0.0
                    if ratio >= min_alpha_ratio:
                        return True
    except Exception:
        return False
    if total_chars == 0:
        return False
    ratio = total_alpha / total_chars if total_chars else 0.0
    return ratio >= min_alpha_ratio and total_chars >= min_chars


def run_ocr(
    pdf_path: Path,
    output_path: Path,
    language: str,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "ocrmypdf",
        "--skip-text",
        "--rotate-pages",
        "--deskew",
        "--optimize",
        "1",
        "-l",
        language,
        str(pdf_path),
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def collect_pdf_paths(pdf_arg: str | None, input_dir: Path) -> list[Path]:
    if pdf_arg:
        pdf_path = Path(pdf_arg)
        if pdf_path.is_dir():
            return sorted(pdf_path.glob("*.pdf"))
        if pdf_path.is_file():
            return [pdf_path]
        raise FileNotFoundError(f"PDF path not found: {pdf_path}")

    input_dir.mkdir(parents=True, exist_ok=True)
    return sorted(input_dir.glob("*.pdf"))


def process_pdf(
    pdf_path: Path,
    language: str,
    min_chars: int,
    min_alpha_ratio: float,
) -> None:
    if has_extractable_text(pdf_path, min_chars=min_chars, min_alpha_ratio=min_alpha_ratio):
        print(f"SKIP (text found): {pdf_path.name}")
        return

    temp_output = pdf_path.with_name(f"{pdf_path.stem}_ocr_tmp.pdf")
    try:
        run_ocr(pdf_path, temp_output, language=language)
        temp_output.replace(pdf_path)
        print(f"OCR OK (replaced): {pdf_path.name}")
    except subprocess.CalledProcessError as exc:
        print(f"OCR FAILED: {pdf_path.name} ({exc})")
        if temp_output.exists():
            temp_output.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OCR PDFs that have no extractable text and overwrite originals."
    )
    parser.add_argument(
        "--pdf",
        default=None,
        help="Path to a PDF file (optional). If omitted, all PDFs in kitaplar are processed.",
    )
    parser.add_argument(
        "--input-dir",
        default="kitaplar",
        help="Folder containing PDFs when --pdf is not provided.",
    )
    parser.add_argument(
        "--language",
        default="tur+eng",
        help="OCR language(s) for ocrmypdf.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=30,
        help="Minimum extracted characters to consider a PDF as having text.",
    )
    parser.add_argument(
        "--min-alpha-ratio",
        type=float,
        default=0.02,
        help="Minimum alphabetic ratio to consider a PDF as having text.",
    )
    args = parser.parse_args()

    pdf_paths = collect_pdf_paths(args.pdf, Path(args.input_dir))
    if not pdf_paths:
        print(f"No PDFs found in {Path(args.input_dir)}.")
        return

    for pdf_path in pdf_paths:
        process_pdf(
            pdf_path=pdf_path,
            language=args.language,
            min_chars=args.min_chars,
            min_alpha_ratio=args.min_alpha_ratio,
        )


if __name__ == "__main__":
    main()
