import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load cấu hình từ file .env
load_dotenv()

# 2. Khởi tạo Local LLM (Ollama)
llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL"),
    base_url=os.getenv("OLLAMA_BASE_URL"),
    temperature=0
)

# --- Prompt 1: Trích xuất thông tin ---
prompt_extract = ChatPromptTemplate.from_template(
    "Extract the technical specifications from the following text. "
    "Only list the specs, no conversational filler:\n\n{text_input}"
)

# --- Prompt 2: Chuyển sang định dạng JSON ---
prompt_transform = ChatPromptTemplate.from_template(
    "Transform the following specifications into a valid JSON object "
    "with 'cpu', 'memory', and 'storage' as keys. "
    "Return ONLY the JSON block:\n\n{specifications}"
)

# --- Xây dựng Chain ---
# Bước 1: Trích xuất
extraction_chain = prompt_extract | llm | StrOutputParser()

# Bước 2: Tổng hợp thành chuỗi hoàn chỉnh
full_chain = (
    {"specifications": extraction_chain} # lấy phần specifications cho prompt_transform
    | prompt_transform # cho vào specs vào prompt_transform
    | llm # cho vào llm
    | StrOutputParser() # chỉ lấy chuỗi kết quả
)

# --- Chạy thực tế ---
if __name__ == "__main__":
    input_text = "The new laptop model features a 3.5 GHz octa-core processor, 16GB of RAM, and a 1TB NVMe SSD."
    
    print(f"--- Đang xử lý với model: {os.getenv('OLLAMA_MODEL')} ---")
    
    try:
        final_result = full_chain.invoke({"text_input": input_text})
        print("\n--- Kết quả JSON cuối cùng ---")
        print(final_result)
    except Exception as e:
        print(f"Lỗi: Hãy đảm bảo Ollama đã được mở và model đã được tải. \nChi tiết: {e}")