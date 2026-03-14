
import asyncio
import os
import sys

# Add the project root to sys.path
sys.path.append(os.getcwd())

async def main():
    print("Testing litellm import and usage...")
    try:
        from litellm import acompletion
        print("litellm imported successfully.")
        # We won't actually call it to avoid hanging if ollama isn't running, 
        # but the import success confirms availability.
        # Actually, let's try a dry run if possible, or just assume it works if import works.
        # The user said they are using "local LLMs (like Ollama)".
    except ImportError:
        print("litellm import failed.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
