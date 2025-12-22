import os
import asyncio


from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_agent


load_dotenv()
# --- Configuration ---
llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL"),
    base_url=os.getenv("OLLAMA_BASE_URL"),
    temperature=0
)
# --- Define Tools ---
@tool
def search_information(query: str) -> str:
    """
    Provides factual information on a given topic. Use this tool to find answers to questions
    like 'What is the capital of France?' or 'What is the weather in London?'.
    """
    print(f"\n--- 🛠️ Tool Called: search_information with query: '{query}' ---")
    # Simulate a search tool with a dictionary of predefined results.
    simulated_results = {
        "weather in london": "The weather in London is currently cloudy with a temperature of 15°C.",
        "capital of france": "The capital of France is Paris.",
        "population of earth": "The estimated population of Earth is around 8 billion people.",
        "tallest mountain": "Mount Everest is the tallest mountain above sea level.",
        "default": f"Simulated search result for '{query}': No specific information found, but the topic seems interesting."
    }
    result = simulated_results.get(query.lower(), simulated_results["default"])
    print(f"--- TOOL RESULT: {result} ---")
    return result

tools = [search_information]

# --- Create the Tool-calling Agent --- 
 # Create the agent, binding the LLM, tools, and prompt together.
agent = create_agent(llm, tools, system_prompt = """
You are a helpful assistant.
When a tool returns information, use it to directly answer the user's question.
Do not explain the tool usage unless asked.
"""
)
# AgentExecutor is the runtime that invokes the agent and executes the chosen tools.
# The 'tools' argument is not needed here as they are already bound to the agent.


async def run_agent_with_tool(query: str):
    """Invokes the agent executor with a query and prints the final response."""
    print(f"\n--- 🏃 Running Agent with Query: '{query}' ---")
    try:
        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": query}]}
        )
        print("\n--- ✅ Final Agent Response ---")
        final_answer = response["messages"][-1].content
        print(final_answer)
    except Exception as e:
        print(f"\n🛑 An error occurred during agent execution: {e}")

async def main():
    """Runs all agent queries concurrently."""
    tasks = [
        run_agent_with_tool("What is the capital of France?"),
        run_agent_with_tool("What's the weather like in London?"),
        run_agent_with_tool("Tell me something about dogs.") # Should trigger the default tool response
    ]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    # Run all async tasks in a single event loop.
    asyncio.run(main())