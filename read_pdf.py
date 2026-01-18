
import PyPDF2
import sys

try:
    with open('diffusion model.pdf', 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        if len(reader.pages) > 0:
            page = reader.pages[0]
            text = page.extract_text()
            print("---START OF TEXT---")
            print(text[:2000]) # Print first 2000 chars
            print("---END OF TEXT---")
        else:
            print("PDF has no pages")
except Exception as e:
    print(f"Error reading PDF: {e}")
