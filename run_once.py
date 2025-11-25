"""
Single-shot runner for the Antigravity agent.
Accepts one prompt via command line and returns the agent's response.
"""
import asyncio
import sys
from datetime import date
from google.genai import types
from google.adk.runners import Runner
from google.adk.memory import InMemoryMemoryService

# Import your existing agent setup
# We'll import the already-configured runner and memory_service
from agent import runner, memory_service


async def run_single_query(prompt: str) -> str:
    """
    Run a single query through the agent and return the response.
    
    Args:
        prompt: The user's query
        
    Returns:
        The agent's response text
    """
    user_id = "GitHub_Actions"
    session_id = f"oneshot_{date.today().strftime('%Y%m%d')}"
    
    # Create a fresh session for this run
    try:
        # Try to clean up any existing session first
        await runner.session_service.delete_session(
            app_name="Agent_V2",
            user_id=user_id,
            session_id=session_id
        )
    except:
        pass  # Session didn't exist, that's fine
    
    # Create new session
    session = await runner.session_service.create_session(
        app_name="Agent_V2",
        user_id=user_id,
        session_id=session_id
    )
    
    await memory_service.add_session_to_memory(session)
    
    # Create the user message
    user_message = types.Content(
        role="user",
        parts=[types.Part(text=prompt)]
    )
    
    # Run the agent and collect the final response
    final_answer = None
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=user_message
    ):
        if event.is_final_response():
            if event.content.parts and event.content.parts[0].text:
                final_answer = event.content.parts[0].text
    
    # Clean up the session
    try:
        await runner.session_service.delete_session(
            app_name="Agent_V2",
            user_id=user_id,
            session_id=session_id
        )
    except:
        pass
    
    return final_answer or "(No response from agent)"


async def main():
    """Main entry point for single-shot execution."""
    if len(sys.argv) < 2:
        print("Usage: python run_once.py '<your prompt here>'")
        sys.exit(1)
    
    prompt = sys.argv[1]
    
    print("=" * 60)
    print("Antigravity Agent - Single Query Mode")
    print("=" * 60)
    print(f"\n📝 Prompt: {prompt}\n")
    print("⏳ Processing...\n")
    
    response = await run_single_query(prompt)
    
    print("=" * 60)
    print("🤖 Agent Response:")
    print("=" * 60)
    print(response)
    print("\n" + "=" * 60)
    
    # Also save to file for GitHub Actions artifact
    with open("agent_output.txt", "w", encoding="utf-8") as f:
        f.write(f"PROMPT:\n{prompt}\n\n")
        f.write(f"RESPONSE:\n{response}\n")
    
    print("\n✅ Output saved to agent_output.txt")


if __name__ == "__main__":
    asyncio.run(main())
