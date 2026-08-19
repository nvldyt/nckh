# gemini_service.py
import time
import uuid
import streamlit as st
from typing import Optional, Tuple, List, Dict, Any

from google import genai
from google.genai import types
from retrieval_engine import retrieve_evidence
from citation_engine import CitationEngine

BASE_SYSTEM_RULES = """
Bạn là trợ lý nghiên cứu khoa học, hỗ trợ viết luận văn Chuyên khoa cấp I ngành Dược lâm sàng.
NGUYÊN TẮC BẮT BUỘC:
1. Tài liệu được cung cấp là nguồn bằng chứng ưu tiên duy nhất.
2. Không tự tạo số liệu, p-value, OR, HR, tỷ lệ, độ thanh thải hoặc cỡ mẫu nếu không có trong bằng chứng.
3. Mọi khẳng định phải chèn MÃ ĐỊNH DANH của tài liệu ngay sau câu. VD: "Tỷ lệ này là 12% [REF-001]."
4. TUYỆT ĐỐI KHÔNG tự đánh số [1], [2], [3] và không dùng citation dạng tác giả-năm.
5. Dùng chính xác thuật ngữ chuyên ngành Dược lâm sàng, văn phong học thuật, khô khan và trực diện.
6. Nếu context không đủ bằng chứng, phải nói rõ: "Tài liệu được cung cấp chưa đủ bằng chứng để kết luận phần này."
"""

def get_gemini_client():
    try: 
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception: 
        import os
        api_key = os.getenv("GEMINI_API_KEY")
        
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

def call_gemini(prompt: str, model: str = "gemini-3.6-flash", max_retries: int = 4) -> Optional[str]:
    """Hàm gọi API Gemini với cơ chế Exponential Backoff an toàn."""
    client = get_gemini_client()
    if not client:
        st.error("⚠️ Lỗi cấu hình API Key.")
        return None

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1) # Nhiệt độ thấp = Đề cao tính chính xác, giảm sáng tạo
            )
            return getattr(response, "text", "").strip()
        except Exception as exc:
            if attempt == max_retries - 1:
                st.error(f"Lỗi kết nối Gemini: {exc}")
                return None
            time.sleep(2 ** attempt) # Thử lại sau 1s, 2s, 4s...
    return None

def generate_evidence_based(task: str, query: str, k: int = 8) -> Tuple[Optional[str], List[Dict[str, Any]], List[str], List[Dict]]:
    """
    Quy trình: Rút trích bằng chứng -> Đăng ký Citation -> Xây dựng Prompt -> Gọi LLM -> Chuyển đổi mã Citation.
    """
    # 1. Truy xuất bằng chứng
    evidence = retrieve_evidence(query, k=k)
    if not evidence:
        return "Tài liệu được cung cấp chưa đủ bằng chứng để kết luận.", [], [], []

    # 2. Xây dựng Context và đăng ký Citation an toàn
    engine = CitationEngine()
    evidence_context = ""
    
    for ev in evidence:
        meta = st.session_state.get("documents", {}).get(ev["source_id"], {})
        # Đăng ký với CitationEngine, trả về mã REF-... (để ép LLM không bịa số)
        ref_tag = engine.register_evidence(ev["source_id"], meta)
        
        table_note = f"\n(Ghi chú bảng: {ev['table_hint']})" if ev.get("table_hint") else ""
        evidence_context += f"\n--- TÀI LIỆU {ref_tag} ---\nNguồn: {ev['file_name']} | Trang: {ev['page']}\nNội dung: {ev['text']}{table_note}\n"

    # 3. Gửi Prompt cho Gemini
    prompt = f"""{BASE_SYSTEM_RULES}
    
NHIỆM VỤ CỦA BẠN:
{task}

BẰNG CHỨNG ĐƯỢC PHÉP SỬ DỤNG:
{evidence_context}

LƯU Ý CUỐI: PHẢI dùng nguyên vẹn mã [REF-...] từ tài liệu trên để trích dẫn.
"""
    
    selected_model = st.session_state.get("selected_model", "gemini-3.6-flash")
    raw_output = call_gemini(prompt, model=selected_model)
    
    if raw_output is None: 
        return None, evidence, [], []

    # 4. Xử lý hậu kỳ (Chuyển [REF-...] thành [1][2], bắt lỗi Hallucination)
    final_text, references, invalid_tags = engine.process_vancouver_citations(raw_output)
    
    if invalid_tags:
        final_text += f"\n\n> ⚠️ CẢNH BÁO KIỂM SOÁT TỰ ĐỘNG: Phát hiện AI tự tạo mã trích dẫn không có trong bằng chứng: {', '.join(invalid_tags)}."

    # 5. Lưu lại lịch sử bản nháp để phục vụ tab Audit sau này
    draft_record = {
        "id": f"draft_{uuid.uuid4().hex[:12]}",
        "task": task.splitlines()[0][:50],
        "text": final_text,
        "references": references,
        "evidence": evidence,
        "created_at": time.time()
    }
    
    if "draft_history" not in st.session_state:
        st.session_state["draft_history"] = []
    
    st.session_state["draft_history"].append(draft_record)

    return final_text, evidence, invalid_tags, references
