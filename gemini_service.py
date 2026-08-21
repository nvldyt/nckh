import os
import time
import gc
import io
import uuid
import itertools
import threading
import streamlit as st
from typing import List, Dict, Any, Tuple, Optional

# CHỈ DÙNG THƯ VIỆN MỚI GOOGLE-GENAI
from google import genai
from google.genai import types

# ============================================================
# CẤU HÌNH MODEL MẶC ĐỊNH
# ============================================================
DEFAULT_MODEL = "gemini-3.7-flash"
MODEL_LITE = "gemini-3.5-flash-lite" 

# ============================================================
# 1. QUẢN LÝ API & CƠ CHẾ XOAY VÒNG KEY (ROUND-ROBIN)
# ============================================================

# Khai báo các Key trực tiếp (Thêm các key khác vào danh sách này)
_GEMINI_KEYS = [
    "AQ.Ab8RN6LSSmOHMOa3EtvwwCVqAA13e9z_3LN2309uXev0pLsHyg,
    AQ.Ab8RN6JazLovPr7vvTFVBiUS8NKwAVzTxM3theZkK4Bj41MjYA,
    AQ.Ab8RN6IojyD8oxt2G_QdadzK0cs7MMKOvCfQMEx9K6i-m7hUkg,
    AQ.Ab8RN6J13twVBkGQlETIl68pTiUC-zs4Yv_zLvbOqjY4FOAU9g,
    AQ.Ab8RN6If-EN_ZpABL7_YZu8H8Ziwfz5sK94kSaNSJxgRFSeBLg,
    AQ.Ab8RN6LPAIgE8dbypq2pj9cea2dJDKE2B0hd0ivzCnInLfU3-A,
    AQ.Ab8RN6KSY6NOw7_M6jBUJEpxTTCueWT4TaBPhQg0VT1w2sW9hA"
]

# Thiết lập cơ chế xoay vòng chống sập bằng itertools & threading lock
_key_cycle = itertools.cycle(_GEMINI_KEYS)
_lock = threading.Lock()

def get_next_key() -> str:
    """Lấy Key Gemini tiếp theo theo hình thức xoay vòng tròn an toàn."""
    with _lock:
        return next(_key_cycle)

@st.cache_resource
def get_gemini_client(api_key: str):
    """Khởi tạo Client mới nhất của Google theo chuẩn AQ."""
    return genai.Client(api_key=api_key)

def call_gemini(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_retries: int = 5,
) -> Optional[str]:
    """Hàm gọi API Gemini với cơ chế Tự động Xoay Vòng Key khi bị quá tải."""
    model_name = model if model else DEFAULT_MODEL
    
    for attempt in range(max_retries):
        # Lấy Key trực tiếp từ hàm xoay vòng toàn cục
        current_key = get_next_key().strip()
        
        if not current_key:
            st.error("❌ Không tìm thấy API Key nào trong hệ thống!")
            return None

        try:
            client = get_gemini_client(current_key)
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    safety_settings=[
                        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                    ]
                )
            )
            
            return getattr(response, "text", "").strip()
            
        except Exception as exc:
            err_msg = str(exc).lower()
            
            # Bắt lỗi 429 (Too many requests) hoặc Quota Limit của Google
            if any(code in err_msg for code in ["429", "resource_exhausted", "503", "unavailable", "quota"]):
                st.toast("🔄 Key hiện tại bị quá tải. Đang tự động đổi sang Key khác...")
                time.sleep(1.5) # Nghỉ 1 nhịp ngắn để chuyển key
                continue # Nhảy ngay sang lần thử tiếp theo với Key mới
            
            # Nếu là lỗi khác (như đứt mạng hoặc lỗi logic)
            if attempt == max_retries - 1:
                st.error(f"❌ Đã thử xoay vòng key nhưng vẫn lỗi kết nối Gemini: {exc}")
                return None
            
            time.sleep(2 ** attempt)
            
    return None

# ============================================================
# 2. XỬ LÝ TÀI LIỆU
# ============================================================

def extract_text_from_file(uploaded_file):
    import fitz, docx, pandas as pd
    file_name = uploaded_file.name.lower()
    try:
        if file_name.endswith('.pdf'):
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                text = "\n".join([page.get_text() for page in doc])
            return text
        elif file_name.endswith('.docx'):
            doc = docx.Document(io.BytesIO(uploaded_file.read()))
            text = "\n".join([p.text for p in doc.paragraphs])
            return text
        elif file_name.endswith(('.xlsx', '.xls', '.csv')):
            df = pd.read_csv(uploaded_file) if file_name.endswith('.csv') else pd.read_excel(uploaded_file)
            return df.to_markdown()
        return uploaded_file.read().decode('utf-8', errors='ignore')
    except Exception as e:
        return f"Lỗi trích xuất: {e}"
    finally:
        gc.collect()

# ============================================================
# 3. HỆ THỐNG PROMPT VÀ WRITING PIPELINE 
# ============================================================

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

def generate_evidence_based(
    task_prompt: str, 
    evidence: List[Dict[str, Any]], 
    citation_engine: Any,
    study_context: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[str], List[Dict[str, Any]], List[str]]:
    """
    Quy trình: Xây dựng Context -> Lập Dàn Ý -> Gọi LLM -> Chuyển đổi mã Citation.
    """
    if not evidence:
        return "Tài liệu được cung cấp chưa đủ bằng chứng để kết luận phần này.", evidence, []

    evidence_context = ""
    for ev in evidence:
        tag = citation_engine.register_evidence(ev["source_id"], ev.get("metadata", {}))
        table_note = f"\n(Ghi chú bảng: {ev['table_hint']})" if ev.get("table_hint") else ""
        evidence_context += f"\n--- TÀI LIỆU {tag} ---\nNguồn: {ev.get('file_name', 'N/A')} | Trang: {ev.get('page', 'N/A')}\nNội dung: {ev.get('text', '')}{table_note}\n"

    context_str = ""
    if study_context and any(study_context.values()):
        context_str = "\nBỐI CẢNH ĐỀ TÀI:\n" + "\n".join([f"- {k}: {v}" for k, v in study_context.items()])

    # Bước 1: Dàn ý (Sử dụng Model LITE cho tốc độ)
    outline_prompt = f"{BASE_SYSTEM_RULES}\n{context_str}\nNHIỆM VỤ: Lập dàn ý 3-4 luận điểm chính cho: {task_prompt}\nBẰNG CHỨNG: {evidence_context}"
    structured_outline = call_gemini(outline_prompt, model=MODEL_LITE, temperature=0.1)
    if not structured_outline:
        structured_outline = "1. Đặt vấn đề\n2. Phân tích\n3. Kết luận"

    # Bước 2: Viết nháp (Sử dụng Model chính để đảm bảo chất lượng, nhiệt độ 0.0 để loại bỏ ảo giác)
    draft_prompt = f"""
{BASE_SYSTEM_RULES}
{context_str}

DÀN Ý ĐÃ ĐƯỢC PHÊ DUYỆT:
{structured_outline}

NHIỆM VỤ GỐC CỦA BẠN:
{task_prompt}

BẰNG CHỨNG ĐƯỢC PHÉP SỬ DỤNG:
{evidence_context}

LƯU Ý CUỐI: Bám sát dàn ý trên. PHẢI dùng nguyên vẹn mã [REF-...] từ tài liệu trên để trích dẫn.
"""
    raw_output = call_gemini(draft_prompt, temperature=0.0) 
    
    if not raw_output: 
        return None, evidence, []

    # Bước 3: Xử lý hậu kỳ Citation (Chuyển [REF-...] thành [1][2], bắt lỗi Hallucination)
    final_text, references, invalid_tags = citation_engine.process_vancouver_citations(raw_output)

    # Lưu lại lịch sử bản nháp
    draft_record = {
        "id": f"draft_{uuid.uuid4().hex[:12]}",
        "task": task_prompt.splitlines()[0][:50],
        "text": final_text,
        "references": references,
        "evidence": evidence,
        "created_at": time.time()
    }
    
    if "draft_history" not in st.session_state:
        st.session_state["draft_history"] = []
    
    st.session_state["draft_history"].append(draft_record)

    return final_text, references, invalid_tags

def render_writing_chat():
    pass
