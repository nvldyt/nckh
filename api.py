# api.py
# Cài đặt: pip install fastapi uvicorn pydantic
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

# Import các engine lõi của hệ thống
from evidence_engine import extract_pdf, get_embeddings
from statistical_engine import crosstab_test, binary_logistic_regression
import pandas as pd
import io

app = FastAPI(title="Research Evidence API", version="1.0")

# --- Models Dữ liệu ---
class QueryRequest(BaseModel):
    query: str
    top_k: int = 8

class StatsRequest(BaseModel):
    dependent_vars: List[str]
    independent_vars: List[str]

# --- Endpoints ---

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
