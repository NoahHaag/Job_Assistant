import os
import sys
from tools_2 import read_document

def analyze_doc(filename):
    print(f"--- Analyzing {filename} ---")
    try:
        content = read_document(filename)
        if content.startswith("Error"):
            print(content)
            return
        
        char_count = len(content)
        word_count = len(content.split())
        # Approx tokens (1 token ~= 4 chars)
        token_count = char_count / 4
        
        print(f"Character Count: {char_count}")
        print(f"Word Count: {word_count}")
        print(f"Approx Tokens: {int(token_count)}")
        print("First 500 chars preview:")
        print(content[:500])
        print("\n")
    except Exception as e:
        print(f"Failed to analyze: {e}")

def main():
    with open("comparison_results.txt", "w", encoding="utf-8") as f:
        sys.stdout = f
        analyze_doc("Professional Curriculum Vitae.docx")
        analyze_doc("Resume.docx")
        sys.stdout = sys.__stdout__
    print("Analysis complete. Results saved to comparison_results.txt")

if __name__ == "__main__":
    main()
