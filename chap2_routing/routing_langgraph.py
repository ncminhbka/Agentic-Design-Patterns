#ROUTING WITH LLM

import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch

load_dotenv()
# --- 1. Khởi tạo Local LLM (Ollama) ---
try:
    # 2. Khởi tạo Local LLM (Ollama)
    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL"),
        base_url=os.getenv("OLLAMA_BASE_URL"),
        temperature=0
    )
    print(f"Ollama model '{llm.model}' initialized.")
except Exception as e:
    print(f"Lỗi khởi tạo Ollama: {e}")
    llm = None

# --- 2. Định nghĩa các Handler ---
def booking_handler(request_dict: dict) -> str:
    req = request_dict['request']
    print(f"\n--- ĐANG ĐIỀU HƯỚNG ĐẾN: BOOKING ---")
    return f"Hệ thống Đặt chỗ đã nhận yêu cầu: '{req}'"

def info_handler(request_dict: dict) -> str:
    req = request_dict['request']
    print(f"\n--- ĐANG ĐIỀU HƯỚNG ĐẾN: THÔNG TIN ---")
    return f"Hệ thống Thông tin trả lời cho câu hỏi: '{req}'"

def unclear_handler(request_dict: dict) -> str:
    req = request_dict['request']
    print(f"\n--- YÊU CẦU KHÔNG RÕ RÀNG ---")
    return f"Tôi không chắc chắn cách xử lý yêu cầu: '{req}'. Bạn vui lòng làm rõ hơn."

# --- 3. Prompt điều hướng (Router) ---
router_prompt = ChatPromptTemplate.from_messages([
    ("system", """Phân tích yêu cầu và trả về DUY NHẤT một từ:
    - 'booker' nếu liên quan đến đặt phòng, vé máy bay, khách sạn.
    - 'info' nếu là câu hỏi kiến thức chung hoặc thông tin.
    - 'unclear' nếu không thuộc hai loại trên.
    Chỉ trả về 1 từ duy nhất, không thêm giải thích."""),
    ("user", "{request}")
])

if llm:
    # Chuỗi logic phân loại
    router_chain = router_prompt | llm | StrOutputParser()

    # --- 4. Thiết lập Nhánh điều hướng (RunnableBranch) ---
    # Lưu ý: Ollama đôi khi trả về chuỗi có khoảng trắng, nên dùng .strip().lower()
    delegation_branch = RunnableBranch(
        (lambda x: x['decision'].strip().lower() == 'booker', booking_handler),
        (lambda x: x['decision'].strip().lower() == 'info', info_handler),
        unclear_handler # Nhánh mặc định
    )

    # Kết hợp thành Coordinator Agent
    coordinator_agent = {
        "decision": router_chain, # quyết định phân luồng mà llm đưa ra
        "request": RunnablePassthrough() # gửi request ban đầu cho handler mà llm chọn
    } | delegation_branch

# --- 5. Chạy thử nghiệm ---
if __name__ == "__main__" and llm:
    # Thử nghiệm đặt vé
    print(coordinator_agent.invoke({"request": "Tôi muốn đặt một phòng ở Đà Nẵng"}))
    
    # Thử nghiệm hỏi thông tin
    print(coordinator_agent.invoke({"request": "Thủ đô của Pháp là gì?"}))

    # Thử nghiệm yêu cầu không rõ ràng
    print(coordinator_agent.invoke({"request": "dfvscdscsdc"}))