
import asyncio
import os
import sys

# Add the project root to sys.path
sys.path.append(os.getcwd())

from tools_2 import generate_cover_letter

async def main():
    print("Starting debug test for generate_cover_letter...")
    try:
        result = await generate_cover_letter(
            company_name="Test Company",
            position_title="Test Role",
            job_description="This is a test job description."
        )
        print(f"Result: {result}")
    except Exception as e:
        print(f"Exception caught: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
