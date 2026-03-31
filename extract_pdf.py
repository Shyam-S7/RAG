import fitz
import json
import os

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

if __name__ == "__main__":
    pdf_path = r"d:\PROJECTS\DL\pro_rag\file\rag.pdf"
    if os.path.exists(pdf_path):
        content = extract_text(pdf_path)
        # Write first 5000 chars to a temp file to see what it is
        with open("extract_peek.txt", "w", encoding="utf-8") as f:
            f.write(content[:5000])
        print(f"Extracted {len(content)} characters. Peek saved to extract_peek.txt")
