# api.py
# Cài đặt: pip install fastapi uvicorn pydantic openai
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import pandas as pd
import io
import os
from openai import OpenAI

# Import các engine lõi của hệ thống
from evidence_engine import extract_pdf, get_embeddings
from statistical_engine import crosstab_test, binary_logistic_regression

app = FastAPI(title="Research Evidence API", version="1.0")

# --- Models Dữ liệu ---
class QueryRequest(BaseModel):
    query: str
    top_k: int = 8

class StatsRequest(BaseModel):
    dependent_vars: List[str]
    independent_vars: List[str]

class MeshRequest(BaseModel):
    vietnamese_topic: str

# --- Endpoints ---

@app.post("/api/v1/pubmed/translate_mesh")
async def translate_to_mesh(request: MeshRequest):
    """API chuyên dụng dịch đề tài tiếng Việt sang cấu trúc truy vấn PubMed (MeSH)"""
    # Khai báo key Groq (có thể dùng biến môi trường để bảo mật hơn trong thực tế)
    groq_key = os.getenv("GROQ_API_KEYS", "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") 
    
    system_prompt = (
        "Bạn là một Chuyên gia Thư viện Y khoa (Medical Librarian). "
        "Nhiệm vụ: Chuyển đổi tên đề tài nghiên cứu tiếng Việt thành chuỗi truy vấn PubMed tối ưu.\n"
        "1. Phân tách các khái niệm cốt lõi: Bệnh lý, Thuốc, Đối tượng.\n"
        "2. Chuyển sang chuẩn MeSH. VD: 'Tăng huyết áp' -> 'Hypertension'[MeSH], 'Kháng sinh' -> 'Anti-Bacterial Agents'[MeSH].\n"
        "3. Kết hợp bằng toán tử Boolean (AND, OR) một cách logic.\n"
        "4. CHỈ TRẢ VỀ DUY NHẤT chuỗi truy vấn, tuyệt đối không giải thích."
    )
    
    try:
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1", 
            api_key=groq_key, 
            timeout=15.0
        )
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant", # Tốc độ phản hồi cực nhanh cho API
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Đề tài: {request.vietnamese_topic}"}
            ],
            temperature=0.1, # Đảm bảo tính chính xác, không sáng tạo từ vựng
            max_tokens=200
        )
        
        # Bóc tách và làm sạch chuỗi kết quả, xóa bỏ ngoặc kép thừa
        mesh_query = response.choices[0].message.content.strip().strip('"')
        return {"status": "success", "mesh_query": mesh_query}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/evidence/upload_pdf")
async def upload_evidence_pdf(file: UploadFile = File(...)):
    """API bất đồng bộ xử lý PDF nặng mà không làm đơ giao diện"""
    try:
        content = await file.read()
        # Mock file object cho engine cũ
        class MockFile:
            def __init__(self, data, name):
                self.data = data
                self.name = name
            def getvalue(self): return self.data
            
        source, chunks = extract_pdf(MockFile(content, file.filename))
        return {"status": "success", "source_id": source.source_id, "total_chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/statistics/crosstab")
async def run_crosstab_analysis(file: UploadFile = File(...), indep: str = "", dep: str = ""):
    """API tính toán bảng chéo tách biệt khỏi UI"""
    content = await file.read()
    df = pd.read_excel(io.BytesIO(content))
    
    try:
        result = crosstab_test(df, indep, dep)
        # Convert DataFrame thành dạng JSON an toàn để trả về Streamlit
        result["table"] = result["table"].to_dict(orient="records") 
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
