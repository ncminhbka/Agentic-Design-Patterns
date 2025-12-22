from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate


task_prompt = "do math"
message_history = [HumanMessage(content=task_prompt)]
print(message_history)
synthesis_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are an expert AI assistant that synthesizes information."),
    HumanMessage(content=task_prompt)
])
print(synthesis_prompt)