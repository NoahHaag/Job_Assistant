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
        
        # Parse the output to separate Markdown text and Mermaid graph
        # The tool returns markdown with a ```mermaid block
        parts = output.split("```mermaid")
        md_content = parts[0]
        mermaid_content = parts[1].split("```")[0] if len(parts) > 1 else ""
        
        # Convert simple markdown headers/bullets to HTML for better display
        html_body = md_content.replace("# ", "<h1>").replace("## ", "<h2>").replace("- ", "<li>").replace("\n", "<br>")
        # Fix the <h1> and <h2> tags not closing (simple hack for this specific output structure)
        html_body = html_body.replace("<h1>", "<h1>", 1).replace("<br><h2>", "</h1><h2>").replace("<br><li>", "</h2><ul><li>")
        # This is a bit brittle, let's just use a cleaner template approach since we know the structure
        
        # Re-fetching data directly might be cleaner, but let's stick to parsing the tool output 
        # or just formatting the known structure since we control tools_2.py.
        # Actually, let's just make a nice HTML template and inject the raw data if we can, 
        # but since we are calling the tool, let's just format the known output string.
        
        # Better approach: Just wrap the mermaid part in the div class="mermaid"
        # and wrap the rest in a div for text.
        
        # Prepare HTML content
        # Convert simple markdown headers/bullets to HTML for better display
        # We do this outside the f-string to avoid SyntaxError with backslashes
        
        formatted_content = md_content.replace("# 🚀 Job Search Portfolio", "<h1>🚀 Job Search Portfolio</h1>")
        formatted_content = formatted_content.replace("## 📊 At a Glance", "<h2>📊 At a Glance</h2><ul>")
        formatted_content = formatted_content.replace("- **", "<li><strong>")
        formatted_content = formatted_content.replace("**:", ":</strong>")
        formatted_content = formatted_content.replace("\n\n## 🕸️ Network Graph", "</ul><h2>🕸️ Network Graph</h2>")
        formatted_content = formatted_content.replace("(Rendered in Mermaid.js below)", "")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Job Search Portfolio</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background-color: #f5f5f5; }}
                .container {{ background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                ul {{ list-style-type: none; padding: 0; }}
                li {{ background: #e8f4f8; margin: 5px 0; padding: 10px; border-radius: 5px; border-left: 4px solid #3498db; }}
                .mermaid {{ margin-top: 20px; text-align: center; }}
                .footer {{ margin-top: 40px; font-size: 0.8em; color: #7f8c8d; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="container">
                <!-- Injected Content -->
                {formatted_content}
                
                <div class="mermaid">
                {mermaid_content}
                </div>

                <div class="footer">
                    Built with my AI Agent 🤖
                </div>
            </div>

            <script type="module">
              import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.js';
              mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
            </script>
        </body>
        </html>
        """
        
        with open("index.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        print("Generated index.html (portfolio) with rendered graph.")

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
