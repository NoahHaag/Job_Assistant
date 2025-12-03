from tools_2 import search_jobs_serpapi
import json

def test_location(loc):
    print(f"Testing location: '{loc}'")
    result = search_jobs_serpapi("Marine Scientist", location=loc)
    if isinstance(result, dict) and "error" in result:
        print(f"❌ Failed: {result['error']}")
    else:
        print(f"✅ Success! Found {len(result)} jobs")

print("--- Testing Locations ---")
test_location("Florida")
test_location("United States")
test_location("Remote")
