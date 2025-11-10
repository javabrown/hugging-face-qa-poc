# hello_ocr.py
import os
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter
import pytesseract

INPUT_IMAGE = os.environ.get("INPUT_IMAGE", "/inputs/sample.jpg")
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", "/outputs/ocr.txt")

def preprocess(im: Image.Image) -> Image.Image:
    # 1) convert to grayscale
    im = ImageOps.grayscale(im)
    # 2) upscale to help tiny fonts (2x)
    w, h = im.size
    im = im.resize((w*2, h*2), Image.LANCZOS)
    # 3) increase contrast a bit
    im = ImageOps.autocontrast(im)
    # 4) slight sharpen
    im = im.filter(ImageFilter.UnsharpMask(radius=1.2, percent=150, threshold=3))
    return im

def main():
    in_path = Path(INPUT_IMAGE)
    out_path = Path(OUTPUT_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise SystemExit(f"[error] input image not found: {in_path}")

    img = Image.open(in_path).convert("RGB")
    img = preprocess(img)

    # Tesseract config:
    #  -l eng : English
    #  --oem 3: default LSTM engine
    #  --psm 6: assume a block of text with multiple lines (good default for pages)
    config = r'-l eng --oem 3 --psm 6'
    text = pytesseract.image_to_string(img, config=config)

    out_path.write_text(text, encoding="utf-8")
    preview = text.strip().splitlines()[0] if text.strip() else ""
    print(f"✅ OCR text saved to: {out_path}")
    if preview:
        print(f"🔎 First line: {preview[:120]}")

if __name__ == "__main__":
    main()
