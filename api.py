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
    
    # ⚠️ Lưu ý bảo mật: Bạn đang lộ API Key cứng trong code. Hãy đảm bảo dùng biến môi trường khi deploy.
    groq_key = os.getenv("GROQ_API_KEYS", "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") 
    
    # ÉP KHUÔN (FEW-SHOT PROMPTING) ĐỂ CHỐNG ẢO GIÁC LỆCH TỪ KHÓA
    system_prompt = (
        "Bạn là một Chuyên gia Thư viện Y khoa (Medical Librarian) chuyên tra cứu PubMed.\n"
        "Nhiệm vụ: Dịch đề tài nghiên cứu tiếng Việt sang cú pháp tìm kiếm MeSH của PubMed.\n\n"
        "Quy tắc tuyệt đối:\n"
        "- CHỈ trả về chuỗi boolean query cuối cùng.\n"
        "- TUYỆT ĐỐI KHÔNG giải thích, KHÔNG thêm câu chào, KHÔNG bọc ngoặc kép ở 2 đầu chuỗi kết quả.\n\n"
        "--- VÍ DỤ CHUẨN (FEW-SHOT) ---\n"
        "Input: Phân tích tình hình sử dụng thuốc đái tháo đường\n"
        "Output: \"Diabetes Mellitus\"[Mesh] AND (\"Drug Utilization\"[Mesh] OR \"Drug Therapy\"[Mesh])\n\n"
        "Input: Đánh giá hiệu quả điều trị viêm dạ dày bằng kháng sinh\n"
        "Output: \"Gastritis\"[Mesh] AND \"Anti-Bacterial Agents\"[Mesh] AND \"Treatment Outcome\"[Mesh]\n\n"
        "Input: Tuân thủ điều trị tăng huyết áp ở người cao tuổi\n"
        "Output: \"Hypertension\"[Mesh] AND \"Treatment Adherence and Compliance\"[Mesh] AND \"Aged\"[Mesh]\n"
        "------------------------------"
    )
    
    try:
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1", 
            api_key=groq_key, 
            timeout=15.0
        )
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                # Thêm tiền tố "Input: " và "Output: " để kích hoạt nhận diện mẫu của LLM
                {"role": "user", "content": f"Input: {request.vietnamese_topic}\nOutput:"}
            ],
            temperature=0.0, # Đưa về 0.0 để loại bỏ hoàn toàn tính "sáng tạo" rủi ro
            max_tokens=200
        )
        
        # Làm sạch chuỗi kết quả triệt để
        mesh_query = response.choices[0].message.content.strip().strip('"').strip("'")
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
