import os
import time
import gc
import io
import requests
import streamlit as st
from typing import List, Dict, Any, Tuple, Optional

# CHỈ DÙNG THƯ VIỆN MỚI
from google import genai
from google.genai import types

import key_manager 

# ============================================================
# CẤU HÌNH MODEL MẶC ĐỊNH
# ============================================================
DEFAULT_MODEL = "gemini-3.7-flash"
MODEL_LITE = "gemini-3.5-flash-lite" 

# ============================================================
# 1. QUẢN LÝ API (BYPASS BẰNG REST API TRỰC TIẾP)
# ============================================================

def call_gemini(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_retries: int = 5,
) -> Optional[str]:
    """
    Hàm gọi AI không dùng SDK, dùng trực tiếp Requests để ép Google nhận Key AQ.
    """
    model_name = model if model else DEFAULT_MODEL
    
    for attempt in range(max_retries):
        try:
            # Lấy key từ key_manager
            api_key = key_manager.get_next_key().strip()
            if not api_key:
                st.error("❌ Không tìm thấy API Key nào trong hệ thống!")
                return None
            
            # ÉP BUỘC GỬI KEY QUA URL ĐỂ VƯỢT LỖI CỦA SDK
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": temperature
                },
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ]
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            res_data = response.json()
            
            # Nếu thành công, trích xuất text và trả về
            if response.status_code == 200:
                try:
                    return res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
                except (KeyError, IndexError):
                    return None
            
            # Nếu có lỗi, xử lý theo mã lỗi
            err_msg = str(res_data).lower()
            if any(code in err_msg for code in ["429", "resource_exhausted", "503", "unavailable", "quota"]):
                st.toast("🔄 Key bị quá tải, đang đổi Key mới...")
                time.sleep(1.5)
                continue
            
            if "401" in err_msg or "unauthenticated" in err_msg:
                st.error(f"❌ Key {api_key[:10]}... bị từ chối xác thực. Lỗi: {res_data}")
                return None # Đừng retry nếu key bị từ chối thẳng thừng
                
            if attempt == max_retries - 1:
                st.error(f"❌ Lỗi API Gemini: {res_data}")
                return None
            
            time.sleep(2 ** attempt)
            
        except Exception as exc:
            if attempt == max_retries - 1:
                st.error(f"❌ Lỗi kết nối mạng: {exc}")
                return None
            time.sleep(2 ** attempt)
            
    return None

# ============================================================
# 2. XỬ LÝ TÀI LIỆU
# ============================================================

def extract_text_from_file(uploaded_file):
    # (Giữ nguyên logic của bạn)
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
# 3. HỆ THỐNG PROMPT VÀ WRITING PIPELINE (GIỮ NGUYÊN)
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

def generate_evidence_based(task_prompt, evidence, citation_engine, study_context=None):
    if not evidence:
        return "Tài liệu chưa đủ bằng chứng.", evidence, []

    evidence_context = ""
    for ev in evidence:
        tag = citation_engine.register_evidence(ev["source_id"], ev.get("metadata", {}))
        evidence_context += f"\nTài liệu {tag}: {ev.get('text', '')}\n"

    context_str = ""
    if study_context and any(study_context.values()):
        context_str = "\nBỐI CẢNH ĐỀ TÀI:\n" + "\n".join([f"- {k}: {v}" for k, v in study_context.items()])

    # Bước 1 & 2: Dàn ý
    outline_prompt = f"{BASE_SYSTEM_RULES}\n{context_str}\nNHIỆM VỤ: Lập dàn ý 3-4 luận điểm cho: {task_prompt}\nBẰNG CHỨNG: {evidence_context}"
    outline_res = call_gemini(outline_prompt, model=MODEL_LITE, temperature=0.1)
    structured_outline = outline_res if outline_res else "1. Đặt vấn đề\n2. Phân tích\n3. Kết luận"

    # Bước 3: Viết nháp
    draft_prompt = f"{BASE_SYSTEM_RULES}\n{context_str}\nDÀN Ý: {structured_outline}\nNHIỆM VỤ: {task_prompt}\nBẰNG CHỨNG: {evidence_context}"
    raw_output = call_gemini(draft_prompt, temperature=0.0) 
    
    if not raw_output: return None, evidence, []

    # Bước 4: Citation
    final_text, references, invalid_tags = citation_engine.process_vancouver_citations(raw_output)
    return final_text, references, invalid_tags

def render_writing_chat():
    # (Giữ nguyên toàn bộ logic hiển thị st.chat_message của bạn tại đây)
    pass
