# writing_engine.py
import os
import time
import logging
from typing import List, Dict, Any, Tuple, Optional
import streamlit as st
from google import genai
from google.genai import types

# Cấu hình Model mặc định
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
MODEL_LITE = "gemini-3.5-flash-lite" # Dùng cho các tác vụ phụ trợ, đọc metadata

# ============================================================
# 1. QUẢN LÝ VÀ XOAY VÒNG API KEY SIÊU TỐC
# ============================================================

@st.cache_resource
def get_gemini_client(api_key: str):
    """Khởi tạo và lưu cache Client để không phải tạo lại liên tục."""
    return genai.Client(api_key=api_key)

def get_api_keys() -> List[str]:
    """
    Bộ quét API Key thông minh: Tự động gom mọi Key có chữ 'GEMINI' trong secrets.
    Anh có thể khai báo kiểu:
    GEMINI_API_KEYS = "key1, key2, key3" 
    Hoặc:
    GEMINI_KEY_1 = "key1"
    GEMINI_KEY_2 = "key2"
    """
    keys_list = []
    
    # 1. Quét trong Streamlit Secrets
    try:
        for key_name, value in st.secrets.items():
            if "GEMINI" in key_name.upper() and isinstance(value, str):
                cleaned_val = value.replace("\n", ",").replace("\r", ",")
                for k in cleaned_val.split(","):
                    k = k.strip()
                    if k and k not in keys_list:
                        keys_list.append(k)
    except Exception:
        pass
        
    # 2. Quét dự phòng trong Biến môi trường (Environment Variables)
    for env_key, env_val in os.environ.items():
        if "GEMINI" in env_key.upper() and isinstance(env_val, str):
            cleaned_env = env_val.replace("\n", ",").replace("\r", ",")
            for k in cleaned_env.split(","):
                k = k.strip()
                if k and k not in keys_list:
                    keys_list.append(k)

    return keys_list

def call_gemini(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_retries: int = 5,
) -> Optional[str]:
    """
    Hàm gọi AI với cơ chế Fallback (Tự động nhảy Key khi bị lỗi 429/Quota).
    """
    api_keys = get_api_keys()
    if not api_keys:
        st.error("❌ Chưa có API Key nào được cấu hình trong Secrets!")
        return None
        
    model_name = model or DEFAULT_MODEL
    
    # Quản lý chỉ số Key hiện tại trong session_state để xoay vòng đều đặn
    if "current_key_idx" not in st.session_state:
        st.session_state["current_key_idx"] = 0

    total_keys = len(api_keys)
    
    for attempt in range(max_retries):
        current_idx = st.session_state["current_key_idx"] % total_keys
        current_key = api_keys[current_idx]
        
        client = get_gemini_client(current_key)

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=temperature),
            )
            text = getattr(response, "text", None)
            if text:
                return text.strip()
            return None

        except Exception as exc:
            error_msg = str(exc).lower()
            
            # Bắt lỗi quá tải mạng hoặc hết hạn mức của 1 Key
            if any(code in error_msg for code in ["429", "resource_exhausted", "503", "unavailable", "quota"]):
                if total_keys > 1:
                    # REROUTE: Nhảy sang Key tiếp theo ngay lập tức
                    st.session_state["current_key_idx"] += 1
                    next_idx = st.session_state["current_key_idx"] % total_keys
                    status = st.warning(f"🔄 Key {current_idx + 1} đang bận. Tự động chuyển sang Key {next_idx + 1}...")
                    time.sleep(1) # Nghỉ nhịp rất ngắn để tránh bị spam block
                    status.empty()
                    continue 
                else:
                    # Nếu chỉ có 1 Key duy nhất, buộc phải chờ
                    if attempt < max_retries - 1:
                        wait_time = 15  
                        status = st.warning(f"⏳ Hệ thống Google đang quá tải. Đợi {wait_time} giây rồi thử lại...")
                        time.sleep(wait_time)
                        status.empty()  
                    else:
                        st.error("❌ Máy chủ Google Gemini hiện đang quá bận. Vui lòng đợi 1-2 phút rồi thử lại!")
                        return None
            else:
                # Lỗi nghiêm trọng khác (ví dụ: Key bị khóa, model không tồn tại)
                if attempt == max_retries - 1:
                    st.error(f"Lỗi hệ thống Gemini: {error_msg}")
                    return None
                time.sleep(3)

    return None

# ============================================================
# 2. HỆ THỐNG PROMPT & QUẢN LÝ SINH VĂN BẢN (WRITING PIPELINE)
# ============================================================

BASE_SYSTEM_RULES = """
Bạn là trợ lý nghiên cứu khoa học, hỗ trợ viết luận văn Chuyên khoa cấp I ngành Dược lâm sàng.
NGUYÊN TẮC BẮT BUỘC:
1. Tài liệu được cung cấp là nguồn bằng chứng ưu tiên duy nhất.
2. Không tự tạo số liệu, p-value, OR, RR, HR, CI95%, tỷ lệ %, liều dùng hoặc cỡ mẫu nếu không có trong bằng chứng.
3. Không tự tạo tên tác giả, năm, tên bài báo, DOI, PMID.
4. Nếu context không đủ bằng chứng, phải diễn đạt tự nhiên (VD: "Tuy nhiên, y văn hiện tại chưa đề cập..."). 
5. Mọi khẳng định dựa trên tài liệu phải chèn MÃ ĐỊNH DANH của tài liệu đó ngay sau câu. Ví dụ: "Tỷ lệ này là 12% [REF-001]."
6. TUYỆT ĐỐI KHÔNG tự tạo [1], [2], [3] và không dùng citation dạng tác giả-năm.
7. KHÔNG sử dụng các nhãn phân chia máy móc như "Dữ kiện (FACT):", "Diễn giải (INTERPRETATION):".
8. Dùng chính xác thuật ngữ chuyên ngành Dược lâm sàng, văn phong y khoa liền mạch.
"""

def generate_evidence_based(
    task_prompt: str, 
    evidence: List[Dict[str, Any]], 
    citation_engine: Any,
    study_context: Optional[Dict[str, Any]] = None  # <--- BỔ SUNG THAM SỐ NÀY
) -> Tuple[Optional[str], List[Dict[str, Any]], List[str]]:
    """
    Trái tim của Writing Pipeline. Nhận yêu cầu, Bằng chứng, Citation Engine, 
    và Study Context (nếu có) để sinh bản nháp.
    """
    if not evidence:
        return "Tài liệu được cung cấp trong Evidence Database chưa đủ bằng chứng để viết mục này.", evidence, []

    # 1. Định dạng bằng chứng đưa vào ngữ cảnh
    evidence_context = ""
    for ev in evidence:
        tag = citation_engine.register_evidence(ev["source_id"], ev.get("metadata", {}))
        table_note = f"\nGhi chú bảng: {ev['table_hint']}" if ev.get("table_hint") else ""
        
        evidence_context += (
            f"\nTài liệu {tag}:\n"
            f"Nguồn: {ev.get('file_name', 'N/A')} | Trang: {ev.get('page', 'N/A')}\n"
            f"Nội dung: {ev.get('text', '')}{table_note}\n"
        )

    # 2. Xử lý Study Context (Nếu người dùng đã khai báo)
    context_str = ""
    if study_context and any(study_context.values()):
        context_str = "\nBỐI CẢNH ĐỀ TÀI NGHIÊN CỨU (STUDY CONTEXT):\n"
        context_str += "Bạn đang viết luận văn cho đề tài có các đặc điểm sau. Hãy bám sát bối cảnh này, tuyệt đối không đi lạc đề:\n"
        if study_context.get("title"): context_str += f"- Tên đề tài: {study_context['title']}\n"
        if study_context.get("design"): context_str += f"- Thiết kế: {study_context['design']}\n"
        if study_context.get("population"): context_str += f"- Đối tượng: {study_context['population']}\n"
        if study_context.get("sample_size"): context_str += f"- Cỡ mẫu: {study_context['sample_size']}\n"
        if study_context.get("objectives"): context_str += f"- Mục tiêu: {study_context['objectives']}\n"

    # 3. Xây dựng Prompt
    prompt = f"""
{BASE_SYSTEM_RULES}
{context_str}

NHIỆM VỤ CỦA BẠN:
{task_prompt}

BẰNG CHỨNG LÂM SÀNG ĐƯỢC PHÉP SỬ DỤNG:
{evidence_context}

YÊU CẦU TRÍCH DẪN:
- LƯU Ý: KHÔNG ĐƯỢC tự đánh số [1], [2]. PHẢI dùng nguyên vẹn mã [REF-...] được cung cấp trong phần bằng chứng ở trên.
- Đặt mã [REF-...] ở ngay cuối câu chứa thông tin lấy từ nguồn đó.
"""

    # 4. Gọi Gemini
    raw_output = call_gemini(prompt)
    if not raw_output:
        return None, evidence, []

    # 5. Hậu xử lý trích dẫn Vancouver
    final_text, references, invalid_tags = citation_engine.process_vancouver_citations(raw_output)

    if invalid_tags:
        final_text += (
            f"\n\n> ⚠️ CẢNH BÁO KIỂM ĐỊNH: Phát hiện AI tự tạo mã trích dẫn không có "
            f"trong dữ liệu truy xuất: {', '.join(invalid_tags)}. Vui lòng rà soát lại đoạn này."
        )

    return final_text, references, invalid_tags
