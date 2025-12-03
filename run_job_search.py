"""
Standalone script for automated job searching via GitHub Actions.

This script is designed to run on a schedule (via GitHub Actions cron) to:
1. Search for jobs using SerpAPI
2. Save new opportunities to job_opportunities.json
3. Track API usage
4. Exit gracefully if usage limit is reached
"""

import sys
import os
from datetime import datetime

# Import job search functions from tools_2
from tools_2 import search_jobs, get_serpapi_usage_report

# Configure your job searches here
# NOTE: SerpAPI does not support "Remote" as a location parameter or "State, Country" format
# For remote jobs, use a broad location like "United States" and add "Remote" to the query
# Valid location examples: "Florida", "United States", "New York", "California"
JOB_SEARCHES = [
    {
        "query": "Marine Scientist",
        "location": "Florida",  # Changed from "Florida, USA" - SerpAPI doesn't support "State, Country" format
        "date_posted": "week"
    },
    {
        "query": "Research Marine Biologist Remote",  # Added "Remote" to query instead of location
        "location": "United States",  # Changed from "Remote" - not a valid location parameter
        "date_posted": "week"
    },
    {
        "query": "Machine Learning Engineer (Computational Biology/Ecology) Remote",  # Added "Remote" to query
        "location": "United States",  # Changed from "Remote" - not a valid location parameter
        "date_posted": "week"
    }
]

def main():
    """Run predefined job searches and report results."""
    print(f"🔍 Job Search Script - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Check usage before starting
    print("\n📊 Checking SerpAPI usage...")
    usage_report = get_serpapi_usage_report()
    print(usage_report)
    print()
    
    total_new_jobs = 0
    all_warnings = []
    
    # Run each search
    for i, search in enumerate(JOB_SEARCHES, 1):
        print(f"\n🔎 Search {i}/{len(JOB_SEARCHES)}: {search['query']} - {search['location']}")
        print("-" * 60)
        
        try:
            result = search_jobs(
                query=search["query"],
                location=search["location"],
                date_posted=search.get("date_posted", "week"),
                max_results=10,
                usage_limit=95,
                save_results=True
            )
            
            # Debug: Print raw result keys
            # print(f"DEBUG: Result keys: {result.keys()}")
            
            # Check for warnings/errors
            if result.get("warning"):
                all_warnings.append(result["warning"])
                print(f"⚠️  {result['warning']}")
                
                # If usage limit reached, stop searching
                if "Usage limit reached" in result["warning"]:
                    print("\n🛑 Stopping job search - usage limit reached.")
                    break
        except Exception as e:
            print(f"❌ Error in search {i}: {e}")
            all_warnings.append(f"Search {i} failed: {e}")
            continue
        
        # Report results
        new_count = result.get("new_jobs_count", 0)
        total_jobs = len(result.get("jobs", []))
        total_new_jobs += new_count
        
        print(f"✅ Found {total_jobs} jobs, {new_count} new")
        
        # Show usage stats
        usage_stats = result.get("usage_stats", {})
        print(f"📈 Usage: {usage_stats.get('used', 0)}/{usage_stats.get('limit', 100)} searches this month")
    
    # Final summary
    print("\n" + "=" * 60)
    print(f"🎯 Job Search Complete!")
    print(f"   Total new opportunities discovered: {total_new_jobs}")
    
    if all_warnings:
        print(f"   Warnings: {len(all_warnings)}")
        for w in all_warnings:
            print(f"   - {w}")
    
    # Exit code
    # 0 = success
    # 1 = usage limit reached (warn but don't fail the workflow)
    if any("Usage limit reached" in w for w in all_warnings):
        print("\n⚠️  Exiting with code 1 (usage limit)")
        sys.exit(1)
    else:
        print("\n✅ Exiting with code 0 (success)")
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)  # Exit code 2 = error
