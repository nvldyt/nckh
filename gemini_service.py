import time
import uuid
import streamlit as st
from typing import Optional, Tuple, List, Dict, Any

from google import genai
from google.genai import types
from retrieval_engine import retrieve_evidence
from citation_engine import CitationEngine
import itertools
import threading
from typing import List

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

# 1. Khai báo 8 Key cứng trực tiếp
_GEMINI_KEYS = [
    "AQ.Ab8RN6JazLovPr7vvTFVBiUS8NKwAVzTxM3theZkK4Bj41MjYA",
    "AQ.Ab8RN6IojyD8oxt2G_QdadzK0cs7MMKOvCfQMEx9K6i-m7hUkg",
    "AQ.Ab8RN6J13twVBkGQlETIl68pTiUC-zs4Yv_zLvbOqjY4FOAU9g",
    "AQ.Ab8RN6If-EN_ZpABL7_YZu8H8Ziwfz5sK94kSaNSJxgRFSeBLg",
    "AQ.Ab8RN6LPAIgE8dbypq2pj9cea2dJDKE2B0hd0ivzCnInLfU3-A",
    "AQ.Ab8RN6KSY6NOw7_M6jBUJEpxTTCueWT4TaBPhQg0VT1w2sW9hA",
    "AQ.Ab8RN6I5cxAGLSXJgy76-zwSulkj1-rjOpKKny_ylvrkOmRWkA",
    "AQ.Ab8RN6JhBJ5w9bnl4pcVuf_NBh8gb2pwRq756ybmvXnar9Q18A"
]

# 2. Thiết lập cơ chế xoay vòng (Round-Robin) chống sập
_key_cycle = itertools.cycle(_GEMINI_KEYS)
_lock = threading.Lock()

def get_all_api_keys() -> List[str]:
    """Trả về danh sách toàn bộ API Keys (dùng khi cần lấy tổng số lượng Key)."""
    return _GEMINI_KEYS

def get_next_key() -> str:
    """
    Lấy Key Gemini tiếp theo theo hình thức xoay vòng tròn. 
    Tránh lỗi 429 (Too Many Requests) khi gọi AI liên tục.
    """
    with _lock:
        return next(_key_cycle)

def call_gemini(prompt: str, model: str = "gemini-3.5-flash", max_retries: int = 5) -> Optional[str]:
    """Hàm gọi API Gemini với cơ chế Tự động Xoay Vòng Key (Round-Robin) khi bị quá tải."""
    
    for attempt in range(max_retries):
        # Lấy Key trực tiếp từ hàm xoay vòng toàn cục (Gọn gàng, không cần session_state)
        current_key = get_next_key()
        
        try:
            client = genai.Client(api_key=current_key)
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1)
            )
            return getattr(response, "text", "").strip()
            
        except Exception as exc:
            err_msg = str(exc).lower()
            
            # Bắt lỗi 429 (Too many requests) hoặc Quota Limit của Google
            if "429" in err_msg or "quota" in err_msg or "exhausted" in err_msg:
                st.toast(f"🔄 Key hiện tại bị quá tải. Đang tự động đổi sang Key khác...")
                time.sleep(1.5) # Nghỉ 1 nhịp ngắn để chuyển key
                continue # Nhảy ngay sang lần thử tiếp theo với Key mới
            
            # Nếu là lỗi khác (như đứt mạng)
            if attempt == max_retries - 1:
                st.error(f"❌ Đã thử xoay vòng key nhưng vẫn lỗi kết nối Gemini: {exc}")
                return None
                
            time.sleep(2 ** attempt)
            
    return None

    # Biến nhớ vị trí Key đang dùng trong session
    if "current_key_idx" not in st.session_state:
        st.session_state.current_key_idx = 0

    for attempt in range(max_retries):
        # Lấy Key theo vòng lặp (hết danh sách tự động quay lại key đầu tiên)
        current_idx = st.session_state.current_key_idx % len(keys)
        current_key = keys[current_idx]
        
        try:
            client = genai.Client(api_key=current_key)
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1)
            )
            return getattr(response, "text", "").strip()
            
        except Exception as exc:
            err_msg = str(exc).lower()
            
            # Bắt lỗi 429 (Too many requests) hoặc Quota Limit của Google
            if "429" in err_msg or "quota" in err_msg or "exhausted" in err_msg:
                st.toast(f"🔄 Key thứ {current_idx + 1} bị quá tải. Đang tự động đổi sang Key khác...")
                st.session_state.current_key_idx += 1
                time.sleep(1.5) # Nghỉ 1 nhịp ngắn để chuyển key
                continue # Nhảy ngay sang lần thử tiếp theo với Key mới
            
            # Nếu là lỗi khác (như đứt mạng)
            if attempt == max_retries - 1:
                st.error(f"❌ Đã thử xoay vòng key nhưng vẫn lỗi kết nối Gemini: {exc}")
                return None
                
            time.sleep(2 ** attempt)
            
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
