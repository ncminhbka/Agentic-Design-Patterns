import operator
from typing import Annotated, List, TypedDict, Union
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END

# --- 1. Cấu hình LLM ---

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL"),
    base_url=os.getenv("OLLAMA_BASE_URL"),
    temperature=0
)

# --- 2. Định nghĩa State (Trạng thái bộ nhớ) ---
# State này sẽ lưu trữ lịch sử chat và bản tóm tắt hiện tại
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add] # Danh sách tin nhắn (ngắn hạn)
    summary: str # Bản tóm tắt nội dung (dài hạn)

# --- 3. Định nghĩa các Nodes (Chức năng) ---

def call_model(state: AgentState):
    """Node này chịu trách nhiệm sinh câu trả lời dựa trên tóm tắt và chat history."""
    summary = state.get("summary", "")
    messages = state["messages"]
    
    # Nếu có tóm tắt, ta đưa nó vào System Prompt để model "nhớ" lại quá khứ
    if summary:
        system_message = f"Bạn là trợ lý AI hữu ích. Đây là tóm tắt cuộc trò chuyện trước đó: {summary}"
        # Trong thực tế, bạn có thể xóa bớt messages cũ ở đây để tiết kiệm token
        # Ở đây ta giữ lại để demo luồng chạy
    else:
        system_message = "Bạn là trợ lý AI hữu ích."

    # Tạo prompt kết hợp context
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    # Gọi Ollama
    chain = prompt | llm
    response = chain.invoke({"messages": messages})
    
    # Trả về message mới để append vào state
    return {"messages": [response]}

def summarize_conversation(state: AgentState):
    """Node này chạy sau mỗi lượt chat để cập nhật bản tóm tắt."""
    summary = state.get("summary", "")
    messages = state["messages"]
    
    # Nếu cuộc hội thoại quá ngắn, chưa cần tóm tắt lại (để tiết kiệm)
    # Nhưng ở đây ta set > 2 tin nhắn là tóm tắt luôn để demo cho bạn thấy
    if len(messages) > 2:
        summary_prompt = (
            "Hãy tóm tắt ngắn gọn cuộc hội thoại trên, bao gồm cả tóm tắt cũ (nếu có) "
            "và nội dung mới trao đổi. Chỉ trả về nội dung tóm tắt, không thêm lời dẫn."
        )
        
        # Gọi LLM để tóm tắt
        response = llm.invoke(
            [
                SystemMessage(content=summary_prompt),
                HumanMessage(content=f"Tóm tắt cũ: {summary}\n\nNội dung hội thoại mới: {messages}")
            ]
        )
        
        # Cập nhật lại summary vào state
        print(f"\n--- [SYSTEM] Đang cập nhật bộ nhớ dài hạn (Summary)... ---")
        return {"summary": response.content}
    
    return {}

# --- 4. Xây dựng Graph (Luồng xử lý) ---

workflow = StateGraph(AgentState)

# Thêm các nodes
workflow.add_node("chatbot", call_model)
workflow.add_node("summarizer", summarize_conversation)

# Định nghĩa luồng đi: Start -> Chatbot -> Summarizer -> End
workflow.set_entry_point("chatbot")
workflow.add_edge("chatbot", "summarizer")
workflow.add_edge("summarizer", END)

# Compile graph
app = workflow.compile()

# --- 5. Chạy thử nghiệm (Vòng lặp Chat) ---

def main():
    print("🤖 Bot đã sẵn sàng! (Gõ 'exit' để thoát)")
    print("Mẹo: Hãy kể tên bạn, sở thích, sau đó hỏi lại xem bot có nhớ không.")
    
    # Khởi tạo bộ nhớ rỗng
    current_state = {"messages": [], "summary": ""}
    
    while True:
        user_input = input("\nBạn: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        # Thêm tin nhắn user vào input
        input_message = HumanMessage(content=user_input)
        current_state["messages"].append(input_message)
        
        # Chạy Graph
        # stream_mode="values" để chúng ta lấy được state cập nhật
        events = app.stream(current_state, stream_mode="values")
        
        for event in events:
            # Lấy trạng thái cuối cùng sau khi chạy qua các node
            current_state = event
            
        # In câu trả lời của Bot (tin nhắn cuối cùng trong list)
        last_msg = current_state["messages"][-1]
        if isinstance(last_msg, AIMessage):
            print(f"Bot: {last_msg.content}")
            
        # [DEBUG] In ra xem Bot đang "nhớ" cái gì trong đầu (Summary)
        if current_state["summary"]:
            print(f"\n🔍 [Memory Dump]: {current_state['summary']}")

if __name__ == "__main__":
    main()