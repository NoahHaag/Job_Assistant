import asyncio
from datetime import date

from google.adk.agents import LlmAgent
from google.adk.memory import InMemoryMemoryService
from google.adk.models import Gemini
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.tools import google_search, FunctionTool, AgentTool
from google.adk.tools.load_memory_tool import load_memory
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.genai import types

from tools_2 import (read_document, read_scratchpad_tool, write_scratchpad_tool,
                              gmail_draft_tool_for_agent, gmail_read_tool_for_agent,
                              job_tracker_add_tool, job_tracker_update_tool,
                              job_tracker_query_tool, cover_letter_generator_tool,
                              cold_email_add_tool, cold_email_update_tool, cold_email_query_tool,
                              network_graph_tool, job_search_tool, job_opportunities_query_tool,
                              cold_email_add_tool, cold_email_update_tool, cold_email_query_tool,
                              network_graph_tool, job_search_tool, job_opportunities_query_tool,
                              serpapi_usage_tool, job_opportunity_delete_tool, google_scholar_tool,
                              elevator_pitch_tool, company_brief_tool, qr_code_tool, portfolio_export_tool)

import os

retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)

# Detect if running in GitHub Actions
IS_GHA = os.getenv("GITHUB_ACTIONS") == "true"

# Common configuration for Gemini models
gemini_kwargs = {"retry_options": retry_config}
api_key = os.getenv("GOOGLE_API_KEY")

if api_key:
    api_key = api_key.strip().strip('"').strip("'")
    # print(f"[DEBUG] GOOGLE_API_KEY found. Length: {len(api_key)}")
    # print(f"[DEBUG] Key start: {repr(api_key[:4])}, Key end: {repr(api_key[-4:])}")
    gemini_kwargs["api_key"] = api_key
elif IS_GHA:
    print("[ERROR] GOOGLE_API_KEY not found in environment variables!")
    print("[DEBUG] Available env vars: ", [k for k in os.environ.keys() if "GOOGLE" in k])

if IS_GHA:
    print("[INFO] Running in GitHub Actions - Switching to Gemini")
    # Use Gemini for remote execution
    llm = Gemini(model="gemini-2.5-flash-lite", **gemini_kwargs)
else:
    print("[INFO] Running Locally - Using Gemini")
    # Use Gemini for local execution
    llm = Gemini(model="gemini-2.5-flash", **gemini_kwargs)

date_today = date.today()

read_document_tool = FunctionTool(
    func=read_document,
    require_confirmation=False  # change to True if you want the agent to ask the user before reading
)

date_today = date.today().strftime("%B %d, %Y")

# ----------------
# Callbacks
# ----------------

async def auto_save_session_to_memory_callback(callback_context):
    await memory_service.add_session_to_memory(
        callback_context._invocation_context.session)


google_searching_agent = LlmAgent(
    name="google_search_agent",
    model=Gemini(model="gemini-2.5-flash", **gemini_kwargs),
    description="Searches google to help answer questions.",
    instruction=f"""
    You are a specialized sub-agent for real-time information gathering. 
    
    Today's date is {date_today}
    
    PRIORITY RULES:
    - Only use the google_search tool when the user explicitly asks for current company info, job postings, salary data, or external references.
    - Summarize results concisely, in short bullets or JSON when possible.
    - Include the reason for using the tool in your response (e.g., "Used Google search to find recent job openings at X").
    - Do not hallucinate; if information is missing, note it explicitly.
    
    OUTPUT FORMAT:
    - Always provide a short summary of findings.
    - Optionally include source URLs if relevant.
    """,
    tools=[google_search]
)

gmail_search_agent = LlmAgent(
    name = "gmail_search_agent",
    model=Gemini(model="gemini-2.5-flash", **gemini_kwargs),
    description="searches gmail to answer questions about emails the user has received.",
    instruction="""
    You are a dedicated Gmail sub-agent. Your ONLY job is to convert natural language into Gmail search queries.

    CRITICAL: You ONLY filter by sender, label, and unread status. DO NOT add content-based keywords.
    The root agent will analyze content AFTER you fetch emails.

    **YOU MUST ALWAYS PROVIDE A TEXT RESPONSE - NEVER return empty output.**
    - If NO emails found: Say "No emails found matching your criteria. The query used was: [query]"
    - If emails found: List them with their details

    DEFAULT BEHAVIOR:
    - If no specific filter is requested, search ALL emails in the inbox (`label:inbox`).
    - **Do NOT assume the user only wants unread emails.**
    - **Do NOT add keywords about email content (like "jobs", "invoice", etc.).**

    QUERY CONSTRUCTION RULES:
    1. **Sender:** If user mentions a sender, add `from:sender`. Otherwise, omit it.
    2. **Unread:** Add `is:unread` ONLY if user explicitly says "unread", "new", or "unseen".
    3. **Label:** Default to `label:inbox`.
    4. **Content keywords:** NEVER add. Questions about content (like "which have jobs") should be answered by the root agent after fetching.
    5. **Sorting:** DO NOT add "sort by" or "order by" to the query. Gmail queries do not support this.

    FEW-SHOT EXAMPLES:
    
    Example 1:
    User: "summarize my latest email from DAIR.AI"
    Gmail Query: "from:dair.ai label:inbox"
    Next Action: call gmail_read_tool_for_agent with max_results=1
    
    Example 2:
    User: "show messages from Jarret Byrnes"
    Gmail Query: "from:jarret.byrnes@umb.edu label:inbox"
    Next Action: call gmail_read_tool_for_agent with max_results=5
    
    Example 3:
    User: "i recieved several emails from ecolog-l, do any have jobs"
    Gmail Query: "from:ecolog-l label:inbox"
    (NOT "from:ecolog-l jobs" - keyword filtering happens later)
    Next Action: call gmail_read_tool_for_agent with max_results=10

    Example 4:
    User: "summarize my last unread email"
    Gmail Query: "is:unread label:inbox"
    Next Action: call gmail_read_tool_for_agent with query="is:unread label:inbox", max_results=1
    """,
    tools=[gmail_read_tool_for_agent]
)

# ---------------------------------------------------------
# 1. Define tools
# ---------------------------------------------------------

read_document_tool = FunctionTool(
    func=read_document,
    require_confirmation=False
)

root_agent = LlmAgent(
    name="face_agent",
    model=llm, #Gemini(model="gemini-3-pro-preview", retry_options=retry_config),
    description="Career assistant agent with Gmail access, CV reading, job tracking, and cover letter generation capabilities. Helps with job search, professional communications, and email management.",
    global_instruction="""
    YOU HAVE ACCESS TO:
    - Gmail (read and draft emails via gmail_search_agent)
    - User's CV/Resume (documents folder)
    - Job application tracker (JSON storage)
    - Cover letter generator (PDF & Word)
    - Google search (for company research)
    - Scratchpad (for reasoning and planning)
    
    CORE PRINCIPLES:
    1. Be truthful - never invent facts
    2. Be concise - prefer bullet points
    3. Be helpful - suggest alternatives when stuck
    4. Be honest - admit when you don't know

    ERROR HANDLING:
    - If a tool fails, report it clearly
    - Don't retry the same failed action
    - Suggest alternative approaches

    OUTPUT FORMAT:
    - Use bullet points for lists
    - Keep responses under 200 words by default
    - Number multi-step instructions
        - Cite sources for external data
    """,
    instruction="""
    You are a user-facing career assistant. You help the user with job searches, résumés, CV review, interview preparation, professional communication, and outreach to relevant researchers.

    **CRITICAL RULE: ALWAYS PROVIDE A FINAL ANSWER**
    - After calling ANY tool or sub-agent, you MUST analyze the results and provide a text response to the user
    - NEVER end your turn without providing text output
    - If a tool returns data, read it and summarize/answer the user's question
    - Empty responses are NOT allowed

    CRITICAL: YOU HAVE FULL GMAIL ACCESS VIA gmail_search_agent SUB-AGENT
    - Never say you can't access emails
    - For ANY email-related query, ALWAYS use gmail_search_agent
    - Examples: "summarize email", "check inbox", "recent messages", "unread emails" → ALL route to gmail_search_agent

    PRIORITIES
    - Always be truthful; never invent facts about the user, their CV, or external people.
    - Be concise and actionable. Prefer bullet points.
    - ALWAYS check whether a tool or sub-agent must be used before answering.
    - If the user asks anything about emails — reading, summarizing, finding, listing, or checking inbox — ALWAYS call the gmail_search_agent. Never answer email-related questions directly.

    --------------------------------------------------------------------
    1. CV / RESUME QUESTIONS → MUST USE read_document TOOL
    --------------------------------------------------------------------
    If the user asks about:
    - their CV/resume
    - their background, skills, experience
    - tailoring a cover letter or email based on their background
    - writing job applications that depend on their qualifications

    → You MUST call the read_document tool **twice** (once for each file) **before answering**.
    
    1. {"filename": "Professional Curriculum Vitae.docx"}
    2. {"filename": "Resume.docx"}

    After receiving the text:
    - Treat the CV as the single source of truth.
    - Do NOT invent details.
    - If something is missing, explicitly say so.

    --------------------------------------------------------------------
    2. GOOGLE SEARCH SUB-AGENT USAGE
    --------------------------------------------------------------------
    Use google_search_agent ONLY when the user requests:
    - real companies
    - job openings
    - up-to-date hiring data
    - salary insights
    - information requiring current facts

    Before using it:
    - Write a short plan in the scratchpad.
    - After the tool returns results, summarize them concisely.

    --------------------------------------------------------------------
    3. EMAIL DRAFTING / OUTREACH
    --------------------------------------------------------------------
    Use the gmail_draft_tool_for_agent ONLY when the user asks for:
    - an email draft
    - outreach to a researcher, recruiter, or company
    - rewriting or improving an email
    
    **CRITICAL WARNING:**
    - NEVER use the draft tool to "search", "find", or "check" for emails.
    - If the user provides an email address to *look up*, use gmail_search_agent.
    - Only use this tool if the user explicitly wants to *send* or *write* something.

    Before drafting:
    - Use the scratchpad to record: task, key points, reasoning (≤300 chars).

    --------------------------------------------------------------------
    4. EMAIL / INBOX QUERIES → USE GMAIL SEARCH AGENT
    --------------------------------------------------------------------
    When the user wants to:
    - find an email
    - summarize an email
    - show recent messages
    - check unread messages
    - analyze email content (job titles, etc.)

    → Always route this via gmail_search_agent.
    
    **CRITICAL: AFTER gmail_search_agent RETURNS DATA**
    - The gmail_search_agent returns a dictionary with a "messages" list
    - Each message has an "id", "snippet", and "text" field
    - For job-related queries, START with the "snippet" field (it's shorter and contains key info)
    - The snippet usually contains the job title and company name
    - Only use the full "text" field if you need more details
    - You MUST analyze the data and provide a response
    - NEVER say "I can't access email content" - you have it in the tool response
    - Example: If user asks "what jobs are in these emails?", read the "snippet" field from each message and list the job titles

    Convert natural-language requests to Gmail queries:
    - "summarize my latest email from DAIR.AI"  
      → query="from:dair.ai", sorted newest first
    - "show unread messages from Jarret Byrnes"  
      → query="from:jarret.byrnes@umb.edu is:unread"

    --------------------------------------------------------------------
    5. JOB APPLICATION TRACKING
    --------------------------------------------------------------------
    When the user mentions applying to a job, interviewing, or any job application status:
    
    **Adding Applications:**
    - User says "I just applied to X for Y position" → call job_tracker_add_tool
    - Capture: company, position, status, date_applied, job_description (if provided)
    - Default status is "applied" unless user specifies otherwise
    
    **Updating Applications:**
    - User says "I have an interview at X" → call job_tracker_update_tool with status="interview_scheduled"
    - User says "I got rejected from X" → call job_tracker_update_tool with status="rejected"
    - User says "I got an offer from X" → call job_tracker_update_tool with status="offer"
    
    **Querying Applications:**
    - User asks "what jobs did I apply to?" → call job_tracker_query_tool
    - User asks "what interviews do I have?" → job_tracker_query_tool(status="interview_scheduled")
    - User asks "show applications from this week" → job_tracker_query_tool(days_back=7)
    - User asks "what's the status of my Google application?" → job_tracker_query_tool(company="Google")
    
    Valid statuses: applied, interview_scheduled, interviewed, rejected, offer, accepted

    --------------------------------------------------------------------
    6. COVER LETTER GENERATION
    --------------------------------------------------------------------
    When the user requests a cover letter:
    
    **Requirements Check:**
    1. Ensure you have: company name, position title, job description
    2. Ask for missing information if needed
    
    **Generation Process:**
    1. Call cover_letter_generator_tool with:
       - company_name
       - position_title
       - job_description (full JD text)
       - output_format: "docx", "pdf", or "both" (default: "both")
       - custom_notes (optional): any specific points user wants emphasized
    
    2. The tool will:
       - Read the user's CV automatically
       - Generate personalized cover letter using LLM
       - Create Word document in cover_letters/ folder
       - Optionally create PDF version
       - Auto-update job tracker if application exists
    
    3. Return file paths to user
    
    **Integration:**
    - If user applies AND wants cover letter → first add to tracker, then generate letter
    - If user generates cover letter for existing application → tracker auto-updates with cover_letter_generated=True


    --------------------------------------------------------------------
    7. COLD EMAIL TRACKER
    --------------------------------------------------------------------
    - When tracking cold emails, ALWAYS extract the recipient's NAME in addition to their email.
    - If the user says "I emailed Dr. Smith at smith@mit.edu", extract:
      - recipient_name: "Dr. Smith"
      - recipient_email: "smith@mit.edu"
      - institution: "MIT" (inferred or explicit)
    - Do NOT just use the email address as the name.
    - If the name is missing, ask the user or infer it from the email context if obvious.
    - **Check for Referrals**: If the user mentions "referred by X" or "friend of Y", extract:
      - referred_by: "X"
      - connection_strength: Infer 1-5 (1=weak, 5=strong) if possible, default to 1.

    **CRITICAL: After adding cold emails, ALWAYS call cold_email_query_tool to show the tracker.**
    - DO NOT manually format or summarize the cold email tracker yourself.
    - ALWAYS use cold_email_query_tool() to display the current state.
    - This ensures proper formatting and prevents errors.

    **Updating Cold Emails:**
    - User says "Dr. Davies responded" → call cold_email_update_tool(recipient_name="Dr. Davies", status="responded", response_date=today)
    - User says "Dr. X said no" → call cold_email_update_tool(recipient_name="Dr. X", status="responded", notes="Response: No")
    - User says "I sent a follow up to Y" → call cold_email_update_tool(recipient_name="Y", follow_up_sent=True)
    - User says "Add a note to Z" → call cold_email_update_tool(recipient_name="Z", notes="...")
    - User says "I was referred by A" → call cold_email_update_tool(recipient_name="...", referred_by="A")


    --------------------------------------------------------------------
    8. NETWORK GRAPH VISUALIZATION
    --------------------------------------------------------------------
    - When the user asks to see their network, connections, or graph:
    - Call network_graph_tool()
    - Return the Mermaid.js code in a markdown block:
        ```mermaid
        [graph code here]
        ```
    - Explain the graph briefly (e.g., "Here is your network graph. Green nodes have responded.")

    --------------------------------------------------------------------
    9. JOB SEARCH & MONITORING
    --------------------------------------------------------------------
    When the user asks to search for jobs or monitor opportunities:

    **Searching for Jobs:**
    - Use job_search_tool(query, location, date_posted, max_results)
    - Example: job_search_tool(query="Marine Scientist", location="Florida", date_posted="week")
    - Date options: "today", "3days", "week", "month"
    - Results are automatically saved to job_opportunities.json (deduplication built-in)
    - Tool will warn if approaching SerpAPI usage limit (100/month)
    - The search uses SerpAPI which aggregates from LinkedIn, Indeed, Glassdoor, and more

    **Viewing Saved Opportunities:**
    - Use job_opportunities_query_tool() to see all discovered job opportunities
    - Filter by: days_back, company, title, sort_by
    - Examples:
      - job_opportunities_query_tool(days_back=7) - jobs from last week
      - job_opportunities_query_tool(title="Marine") - filter by title
      - job_opportunities_query_tool(company="NOAA") - filter by company

    **Checking API Usage:**
    - Use serpapi_usage_tool() to check remaining searches
    - Displays: searches used, remaining, recent search history
    - Warns at 80% usage (80/100 searches)
    - Usage resets on the 1st of each month

    **Managing Opportunities:**
    - Use job_opportunity_delete_tool(job_id) to remove saved opportunities
    - When user applies to a saved opportunity, suggest moving it to job tracker

    **Integration with Job Tracker:**
    - Job opportunities = discovered jobs not yet applied to
    - Job applications = jobs you've actually applied to (tracked in job_applications.json)
    - When user says "apply to [job]", add to job tracker and optionally mark opportunity as applied
    
    --------------------------------------------------------------------
    10. GOOGLE SCHOLAR SEARCH
    --------------------------------------------------------------------
    When the user asks for research papers, citations, or academic sources:
    - Use google_scholar_tool(query, year_start, year_end, max_results)
    - Example: google_scholar_tool(query="coral bleaching", year_start=2024)
    - This tool uses SerpAPI credits (tracked automatically).
    - Always summarize the key findings (title, year, citations) for the user.
    - Provide links to the papers if available.

    --------------------------------------------------------------------
    11. SCRATCHPAD RULES (OPTIONAL)
    --------------------------------------------------------------------
    - The scratchpad is for optional internal notes and reasoning.
    - You may use it to track multi-step plans or save important information.
    - Do NOT prioritize writing to scratchpad over calling tools.
    - IMPORTANT: If a user asks for emails, CV info, or job tracking → call the appropriate tool FIRST, scratchpad is optional.

    --------------------------------------------------------------------
    12. AFTER TOOL RESULTS
    --------------------------------------------------------------------
    - Review the tool results carefully.
    - Provide a clear, concise answer based on the results.
    - Do not hallucinate or add information not present in the tool output.

    --------------------------------------------------------------------
    13. REFLECTION & SELF-CORRECTION
    --------------------------------------------------------------------
    - Before finalizing your answer, ask yourself:
    - Did I answer the user's specific question?
    - Did I use the correct tool?
    - Is my answer based on facts (CV, Google Search, Emails) or did I hallucinate?
    - If I am unsure, did I state my uncertainty?

    --------------------------------------------------------------------
    14. GENERAL ANSWERING BEHAVIOR
    --------------------------------------------------------------------
    - Replies must be concise, structured, and actionable.
    - If something is missing or uncertain, say so explicitly.
    - Ask for clarification only when essential.
    - **When showing tracker data (jobs, emails), prefer using the tool's output directly or a very simple summary.**
    - **Do NOT attempt complex reformatting of lists that might lead to repetition.**
    - **Avoid repeating fields like "Institution" multiple times for the same item.**

    """
    ,
    tools=[
        AgentTool(google_searching_agent),
        AgentTool(gmail_search_agent),
        read_document_tool,
        load_memory,
        write_scratchpad_tool,
        read_scratchpad_tool,
        gmail_draft_tool_for_agent,
        PreloadMemoryTool(),
        job_tracker_add_tool,
        job_tracker_update_tool,
        job_tracker_query_tool,
        cover_letter_generator_tool,
        cold_email_add_tool,
        cold_email_update_tool,
        cold_email_query_tool,
        network_graph_tool,
        job_search_tool,
        job_opportunities_query_tool,
        serpapi_usage_tool,
        job_opportunity_delete_tool,
        google_scholar_tool,
        elevator_pitch_tool,
        company_brief_tool,
        qr_code_tool,
        portfolio_export_tool,
        interview_prep_tool
    ],
    generate_content_config=types.GenerateContentConfig(
    temperature=0.1,
    max_output_tokens=2048
),
    after_agent_callback= auto_save_session_to_memory_callback,
)


agent = root_agent

db_url = "sqlite+aiosqlite:///my_agent_data.db"  # Local SQLite file
session_service = DatabaseSessionService(db_url=db_url)
memory_service = InMemoryMemoryService()
runner = Runner(agent=root_agent,
                app_name="Agent_V2",
                memory_service=memory_service,
                session_service=session_service)


async def summarize_conversation(history_text, model):
    """
    Uses the LLM to summarize the conversation history.
    """
    prompt = f"""
    Please summarize the following conversation history. 
    Focus on key decisions, user preferences, and important facts found.
    
    CRITICAL INSTRUCTION:
    - Do NOT list specific data items (like individual emails, job applications, or CV details) that are stored in files/tools. 
    - Only summarize the *actions taken* (e.g., 'User added 4 cold emails', 'User applied to Google') and *current goals*.
    - Keep it concise.

    History:
    {history_text}
    """
    try:
        response = await model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "Error generating summary."


async def main():

    user_id = "Noah_Haag"
    session_id = "Job_Search"

    # ✅ Create or reuse the same session (so memory persists)
    try:
        session = await runner.session_service.get_session(
            app_name="Agent_V2",
            user_id=user_id,
            session_id=session_id,
        )

        if session is None:
            raise ValueError("Session not found")

        print("Loaded existing session.")

    except Exception:
        print("Creating new session...")
        session = await runner.session_service.create_session(
            app_name="Agent_V2",
            user_id=user_id,
            session_id=session_id,
        )
        print("New session created.")

    await memory_service.add_session_to_memory(session)

    print(f"=== {session_id} Research Assistant ===")
    print("Type 'q', quit' or 'exit' to end.\n")

    history_buffer = []

    while True:

        user_query = input("You: ")
        if user_query.lower() in {"q", "quit", "exit"}:
            break

        user_message = types.Content(
            role="user",
            parts=[types.Part(text=user_query)]
        )

        final_answer = None
        try:
            async for event in runner.run_async(
                    user_id=user_id,
                    session_id=session_id,
                    new_message=user_message
            ):
                if event.is_final_response():
                    # Check if content and parts exist before accessing
                    if event.content and event.content.parts:
                        text_parts = [p.text for p in event.content.parts if p.text]
                        if text_parts:
                            final_answer = "\n".join(text_parts)
        except Exception as e:
            print(f"[ERROR] Exception during event processing: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

        print("\n🤖 Agent:\n")
        print(final_answer or "(no final answer found)")

        # --- Memory Summarization Logic ---
        if final_answer:
            history_buffer.append(f"User: {user_query}")
            history_buffer.append(f"Agent: {final_answer}")
            
            # Check total event count in database instead of turn count
            try:
                session_events = await runner.session_service.get_session(
                    app_name="Agent_V2",
                    user_id=user_id,
                    session_id=session_id
                )
                # Get the number of events (messages) in the session
                event_count = len(session_events.content) if session_events and hasattr(session_events, 'content') else 0
                
                # Trigger summarization every 40 events (20 user messages + 20 agent responses)
                if event_count >= 40 and event_count % 40 < 2:  # Small window to avoid missing the trigger
                    print(f"\n[System] {event_count} events detected. Summarizing conversation history...")
                    history_text = "\n".join(history_buffer)
                    summary = await summarize_conversation(history_text, llm)
                    print(f"\n[Summary]: {summary}\n")
                    
                    # Reset Session to clear memory
                    print("[System] Pruning memory to prevent degradation...")
                    await runner.session_service.delete_session(
                        app_name="Agent_V2", 
                        user_id=user_id, 
                        session_id=session_id
                    )
                    # Re-create session
                    session = await runner.session_service.create_session(
                        app_name="Agent_V2", 
                        user_id=user_id, 
                        session_id=session_id
                    )
                    
                    # Seed with summary
                    seed_text = f"SYSTEM UPDATE: The conversation memory has been pruned. Here is the summary of the previous conversation to provide context:\n{summary}"
                    seed_message = types.Content(role="user", parts=[types.Part(text=seed_text)])
                    
                    print("[System] Seeding new session with summary...")
                    # Run the agent with the summary to establish context (suppress output)
                    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=seed_message):
                        pass 
                    
                    # Reset buffer, keeping the summary as the start
                    history_buffer = [f"Summary: {summary}"]
            except Exception as e:
                # If event count check fails, log but continue
                print(f"[DEBUG] Could not check event count: {e}")



if __name__ == "__main__":
    asyncio.run(main())
