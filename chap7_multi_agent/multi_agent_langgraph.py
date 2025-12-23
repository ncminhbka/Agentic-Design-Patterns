from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL"),
    base_url=os.getenv("OLLAMA_BASE_URL"),
    temperature=0
)
from typing import TypedDict

class AgentState(TypedDict):
    query: str
    research_notes: str
    analysis: str
    final_answer: str

from langchain_core.prompts import ChatPromptTemplate

research_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research agent. Collect key factual points."),
    ("human", "{query}")
])

def research_agent(state: AgentState) -> AgentState:
    response = llm.invoke(
        research_prompt.format_messages(query=state["query"])
    )

    return {
        **state,
        "research_notes": response.content
    }

analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an analysis agent. Extract insights and reasoning."),
    ("human", "Research notes:\n{research_notes}")
])

def analysis_agent(state: AgentState) -> AgentState:
    response = llm.invoke(
        analysis_prompt.format_messages(
            research_notes=state["research_notes"]
        )
    )

    return {
        **state,
        "analysis": response.content
    }

synthesis_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a synthesis agent. Produce a concise final answer."),
    ("human", 
     "Question:\n{query}\n\nAnalysis:\n{analysis}")
])

def synthesis_agent(state: AgentState) -> AgentState:
    response = llm.invoke(
        synthesis_prompt.format_messages(
            query=state["query"],
            analysis=state["analysis"]
        )
    )

    return {
        **state,
        "final_answer": response.content
    }


from langgraph.graph import StateGraph, END

graph = StateGraph(AgentState)

graph.add_node("research", research_agent)
graph.add_node("analysis", analysis_agent)
graph.add_node("synthesis", synthesis_agent)

graph.set_entry_point("research")
graph.add_edge("research", "analysis")
graph.add_edge("analysis", "synthesis")
graph.add_edge("synthesis", END)

app = graph.compile()


def main():
    initial_state: AgentState = {
        "query": "What are the advantages of multi-agent systems over single-agent systems?",
        "research_notes": "",
        "analysis": "",
        "final_answer": ""
    }

    result = app.invoke(initial_state)

    print("FINAL ANSWER:\n")
    print(result["final_answer"])

if __name__ == "__main__":
    main()
