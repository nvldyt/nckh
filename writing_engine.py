import os
import time
import logging
from typing import List, Dict, Any, Tuple, Optional
import streamlit as st
from google import genai
from google.genai import types

# ============================================================
# CẤU HÌNH MODEL MẶC ĐỊNH
# ============================================================
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
MODEL_LITE = "gemini-3.5-flash-lite" # Dùng cho các tác vụ phụ trợ, lập dàn ý, review

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
    Hỗ trợ danh sách chuỗi phẩy hoặc các biến riêng lẻ.
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
        
    # Tự động điều chỉnh model nếu có lỗi phiên bản ngầm từ Google
    model_name = model or DEFAULT_MODEL
    if "3.6" in model_name or "3.7" in model_name:
        # Dự phòng trường hợp Google chưa update model name trên API
        model_name = "gemini-1.5-flash" if "flash" in model_name else model_name
    
    if "current_key_idx" not in st.session_state:
        st.session_state["current_key_idx"] = 0

    total_keys = len(api_keys)
    # Ép vòng lặp quét đủ số lượng key hiện có
    total_attempts = max(max_retries, total_keys + 1)
    
    for attempt in range(total_attempts):
        current_idx = st.session_state["current_key_idx"] % total_keys
        current_key = api_keys[current_idx]
        
        try:
            client = get_gemini_client(current_key)
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
            
            # Bắt lỗi quá tải, cạn Quota hoặc server bận
            if any(code in error_msg for code in ["429", "resource_exhausted", "503", "unavailable", "quota"]):
                if total_keys > 1:
                    st.session_state["current_key_idx"] += 1
                    next_idx = st.session_state["current_key_idx"] % total_keys
                    st.toast(f"🔄 Key {current_idx + 1} đang bận. Đổi sang Key {next_idx + 1}...")
                    time.sleep(1) 
                    continue 
                else:
                    if attempt < total_attempts - 1:
                        wait_time = 15  
                        status = st.warning(f"⏳ Hệ thống Google đang quá tải. Đợi {wait_time} giây rồi thử lại...")
                        time.sleep(wait_time)
                        status.empty()  
                    else:
                        st.error("❌ Máy chủ Google Gemini hiện đang quá bận. Vui lòng đợi 1-2 phút rồi bấm thử lại!")
                        return None
            else:
                if attempt == total_attempts - 1:
                    st.error(f"❌ Lỗi hệ thống Gemini: {error_msg}")
                    return None
                time.sleep(2)

    return None

# ============================================================
# 2. HỆ THỐNG PROMPT VÀ WRITING PIPELINE 5 BƯỚC (AGENTIC WORKFLOW)
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
    study_context: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[str], List[Dict[str, Any]], List[str]]:
    """
    Writing Pipeline 5 Bước (Agentic Workflow):
    - Bước 1: Lập dàn ý (Outline Generation)
    - Bước 2: Lập bản đồ bằng chứng (Evidence Mapping)
    - Bước 3: Viết nháp có kiểm soát từng luận điểm (Controlled Drafting)
    - Bước 4: Kiểm định trích dẫn (Citation Validation)
    - Bước 5: Review học thuật (Scientific Reviewer) - Đã ẩn để tối ưu tốc độ
    """
    if not evidence:
        return "Tài liệu được cung cấp trong Evidence Database chưa đủ bằng chứng để viết mục này.", evidence, []

    # Định dạng bằng chứng đưa vào ngữ cảnh chung
    evidence_context = ""
    for ev in evidence:
        tag = citation_engine.register_evidence(ev["source_id"], ev.get("metadata", {}))
        table_note = f"\nGhi chú bảng: {ev['table_hint']}" if ev.get("table_hint") else ""
        evidence_context += (
            f"\nTài liệu {tag}:\n"
            f"Nguồn: {ev.get('file_name', 'N/A')} | Trang: {ev.get('page', 'N/A')}\n"
            f"Nội dung: {ev.get('text', '')}{table_note}\n"
        )

    # Xử lý Study Context (Bối cảnh đề tài)
    context_str = ""
    if study_context and any(study_context.values()):
        context_str = "\nBỐI CẢNH ĐỀ TÀI NGHIÊN CỨU (STUDY CONTEXT):\n"
        if study_context.get("title"): context_str += f"- Tên đề tài: {study_context['title']}\n"
        if study_context.get("design"): context_str += f"- Thiết kế: {study_context['design']}\n"
        if study_context.get("population"): context_str += f"- Đối tượng: {study_context['population']}\n"
        if study_context.get("sample_size"): context_str += f"- Cỡ mẫu: {study_context['sample_size']}\n"
        if study_context.get("objectives"): context_str += f"- Mục tiêu: {study_context['objectives']}\n"

    # ==========================================
    # BƯỚC 1 & 2: LẬP DÀN Ý & MAP BẰNG CHỨNG (Dùng Model LITE)
    # ==========================================
    outline_prompt = f"""
{BASE_SYSTEM_RULES}
{context_str}
NHIỆM VỤ: Dựa vào yêu cầu và bằng chứng dưới đây, hãy lập dàn ý gồm 3-4 luận điểm chính bằng tiếng Việt trước khi viết chi tiết.
YÊU CẦU: Chỉ trả về dàn ý ngắn gọn, gắn mỗi luận điểm với mã [REF-...] tương ứng sẽ dùng.
YÊU CẦU GỐC: {task_prompt}
BẰNG CHỨNG:
{evidence_context}
"""
    outline_res = call_gemini(outline_prompt, model=MODEL_LITE, temperature=0.1)
    structured_outline = outline_res if outline_res else "1. Đặt vấn đề và tổng quan\n2. Phân tích kết quả\n3. Bàn luận"

    # ==========================================
    # BƯỚC 3: VIẾT NHÁP CÓ KIỂM SOÁT (Controlled Drafting)
    # ==========================================
    draft_prompt = f"""
{BASE_SYSTEM_RULES}
{context_str}

DÀN Ý ĐÃ ĐƯỢC PHÊ DUYỆT:
{structured_outline}

NHIỆM VỤ GỐC:
{task_prompt}

BẰNG CHỨNG LÂM SÀNG ĐƯỢC PHÉP SỬ DỤNG (ƯU TIÊN TỐI THƯỢNG):
{evidence_context}

YÊU CẦU ĐỊNH DẠNG BẮT BUỘC:
- Bám sát dàn ý trên. Viết thành các đoạn văn xuôi y khoa liền mạch, văn phong khô khan, khách quan, không dùng từ ngữ hoa mỹ.
- Nếu không có dữ liệu, hãy trả lời thẳng "Y văn hiện tại chưa cung cấp số liệu về vấn đề này". Tuyệt đối không tự suy luận.
- PHẢI dùng nguyên vẹn mã [REF-...] được cung cấp ở ngay cuối câu chứa thông tin lấy từ nguồn đó.
"""
    # Dùng mô hình chính để đảm bảo chất lượng, nhiệt độ 0.0 để loại bỏ ảo giác
    raw_output = call_gemini(draft_prompt, temperature=0.0) 
    
    if not raw_output:
        return None, evidence, []

    # ==========================================
    # BƯỚC 4: KIỂM ĐỊNH TRÍCH DẪN (Citation Engine)
    # ==========================================
    final_text, references, invalid_tags = citation_engine.process_vancouver_citations(raw_output)

    if invalid_tags:
        final_text += (
            f"\n\n> ⚠️ CẢNH BÁO KIỂM ĐỊNH: Phát hiện AI tự tạo mã trích dẫn không có "
            f"trong dữ liệu truy xuất: {', '.join(invalid_tags)}. Vui lòng rà soát lại đoạn này."
        )

    return final_text, references, invalid_tags
