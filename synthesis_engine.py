# synthesis_engine.py
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from writing_engine import call_gemini, MODEL_LITE

def build_literature_matrix(documents: Dict[str, Any], chunks: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Quét toàn bộ tài liệu trong Evidence Database, sử dụng AI bóc tách thông tin 
    thành Ma trận tổng hợp y văn chuẩn hóa cho luận văn y khoa.
    """
    if not documents:
        return pd.DataFrame()

    # Tổng hợp nội dung tóm tắt từ các tài liệu để AI phân tích
    corpus_summary = ""
    for source_id, meta in documents.items():
        doc_chunks = [c["text"] for c in chunks if c.get("source_id") == source_id]
        sample_text = " ".join(doc_chunks[:3])[:1500]  # Lấy mẫu tối đa 3 đoạn đầu tiên
        
        corpus_summary += f"\n--- TÀI LIỆU [ID: {source_id}] ---\n"
        corpus_summary += f"Tiêu đề: {meta.get('title', meta.get('file_name', ''))}\n"
        corpus_summary += f"Tác giả: {meta.get('authors', 'N/A')} | Năm: {meta.get('year', 'N/A')}\n"
        corpus_summary += f"Nội dung tóm tắt: {sample_text}\n"

    prompt = f"""
Bạn là chuyên gia nghiên cứu y khoa. Dựa vào danh sách các tài liệu dưới đây trong cơ sở dữ liệu bằng chứng, hãy trích xuất và tổng hợp thành một **Ma trận tổng hợp y văn** chuẩn mực cho luận văn Chuyên khoa Dược lâm sàng.

TRẢ VỀ DUY NHẤT MỘT ĐỊNH DẠNG JSON MẢNG (ARRAY OF OBJECTS), không kèm giải thích hay markdown rườm rà ngoài JSON.
Cấu trúc mỗi object JSON bắt buộc gồm đúng 5 trường sau:
- "authors_year": "Tên tác giả (Năm)"
- "title": "Tên bài báo / nghiên cứu"
- "design": "Thiết kế nghiên cứu (VD: Mô tả cắt ngang, Cohort, RCT, Tổng quan...)"
- "sample_size": "Cỡ mẫu (N)"
- "main_results": "Kết quả chính / Phát hiện quan trọng liên quan đến lâm sàng"

DANH SÁCH TÀI LIỆU:
{corpus_summary}
"""

    response = call_gemini(prompt, model=MODEL_LITE, temperature=0.1)
    if not response:
        return pd.DataFrame()

    try:
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        
        data = json.loads(cleaned.strip())
        if isinstance(data, list) and data:
            df = pd.DataFrame(data)
            df = df.rename(columns={
                "authors_year": "Tác giả (Năm)",
                "title": "Tên nghiên cứu",
                "design": "Thiết kế nghiên cứu",
                "sample_size": "Cỡ mẫu (N)",
                "main_results": "Kết quả chính"
            })
            return df
    except Exception:
        pass

    return pd.DataFrame()
