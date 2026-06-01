import os
import PyPDF2

def main():
    path = "documents/Resume.pdf"
    out_path = "resume.txt"
    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
            print(f"Extracted {len(text)} characters.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
