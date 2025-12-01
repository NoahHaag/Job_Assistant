import argparse
import os
import sys
from tools_2 import (
    elevator_pitch_tool,
    company_brief_tool,
    qr_code_tool,
    portfolio_export_tool
)
from google.adk.models import Gemini
from google.genai import types

# Setup Gemini for standalone use (similar to agent.py)
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    api_key = api_key.strip().strip('"').strip("'")

gemini_kwargs = {
    "api_key": api_key,
    "retry_options": types.HttpRetryOptions(attempts=5)
}
model = Gemini(model="gemini-2.5-flash", **gemini_kwargs)

def generate_content(prompt):
    """Simple wrapper to call Gemini."""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating content: {e}"

def main():
    parser = argparse.ArgumentParser(description="Job Fair Tool Runner")
    parser.add_argument("--tool", required=True, choices=["pitch", "brief", "qr", "portfolio"], help="Tool to run")
    parser.add_argument("--input", help="Input text (Company Name or QR Data)")
    parser.add_argument("--jd", help="Job Description (for pitch)", default="")
    
    args = parser.parse_args()
    
    output = ""
    
    if args.tool == "pitch":
        if not args.input:
            print("Error: --input (Company Name) is required for pitch.")
            sys.exit(1)
        
        # Tool returns the prompt, we need to execute it
        prompt = elevator_pitch_tool(args.input, args.jd)
        print(f"Generating pitch for {args.input}...")
        output = generate_content(prompt)
        
    elif args.tool == "brief":
        if not args.input:
            print("Error: --input (Company Name) is required for brief.")
            sys.exit(1)
            
        # Tool returns the prompt, we need to execute it
        prompt = company_brief_tool(args.input)
        print(f"Generating brief for {args.input}...")
        output = generate_content(prompt)
        
    elif args.tool == "qr":
        if not args.input:
            print("Error: --input (Data) is required for QR code.")
            sys.exit(1)
            
        result = qr_code_tool(args.input)
        output = result
        
    elif args.tool == "portfolio":
        output = portfolio_export_tool()
        
        # Save HTML version too
        with open("portfolio.html", "w", encoding="utf-8") as f:
            # Simple HTML wrapper for the markdown/mermaid
            html_content = f"""
            <html>
            <head><title>Job Search Portfolio</title></head>
            <body>
            <pre>{output}</pre>
            <script type="module">
              import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.js';
              mermaid.initialize({{ startOnLoad: true }});
            </script>
            </body>
            </html>
            """
            f.write(html_content)
        print("Generated portfolio.html")

    # Output to console
    print("\n" + "="*30)
    print("       TOOL OUTPUT")
    print("="*30 + "\n")
    print(output)
    
    # Output to GitHub Summary if running in GHA
    github_step_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if github_step_summary:
        with open(github_step_summary, "a", encoding="utf-8") as f:
            f.write(f"# Job Fair Tool: {args.tool.upper()}\n\n")
            f.write(output)

if __name__ == "__main__":
    main()
