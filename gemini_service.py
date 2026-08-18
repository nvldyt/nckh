import os
import time
import random
import logging
import google.generativeai as genai
from google.api_core import exceptions

# =====================================================================
# CẤU HÌNH MODEL THEO CHUẨN MỚI
# =====================================================================

# Model chính cho các tác vụ suy luận sâu (RAG, Viết luận văn, Diễn giải thống kê)
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

# Model Lite cho các tác vụ nhẹ (Kiểm tra chính tả, Dịch thuật, Style check)
# Đã sửa lỗi: gemini-3.5-flash-lite thay vì 3.6
MODEL_LITE = os.getenv("GEMINI_MODEL_LITE", "gemini-3.5-flash-lite")

# Khởi tạo API Key (Thường được gọi 1 lần khi app khởi động)
def init_gemini(api_key: str):
    if not api_key:
        raise ValueError("Chưa cấu hình Google Gemini API Key.")
    genai.configure(api_key=api_key)

# =====================================================================
# CORE CALLER VỚI EXPONENTIAL BACKOFF VÀ ERROR HANDLING
# =====================================================================

def call_gemini(prompt: str, model_name: str = DEFAULT_MODEL, max_retries: int = 5) -> str:
    """
    Giao tiếp với Gemini API.
    Đã loại bỏ các tham số sampling (temperature, top_k, top_p) bị deprecated.
    Sử dụng Exponential Backoff + Jitter để chống nghẽn mạng (Rate Limit 429).
    """
    # Khởi tạo model
    model = genai.GenerativeModel(model_name)
    
    for attempt in range(max_retries):
        try:
            # Gọi API thuần túy, để model tự quyết định cấu hình sampling tối ưu
            response = model.generate_content(prompt)
            
            if response.text:
                return response.text
            return ""

        # LỖI NHÓM 1: Lỗi từ phía Client (Không Retry)
        except exceptions.InvalidArgument as e:
            # Lỗi 400 - Prompt không hợp lệ, payload sai định dạng
            logging.error(f"[Lỗi 400] Bad Request: {e}")
            return f"❌ Lỗi cấu trúc yêu cầu (400). Chi tiết: {str(e)}"
            
        except (exceptions.PermissionDenied, exceptions.Unauthenticated) as e:
            # Lỗi 401/403 - Lỗi API Key, Quota billing, Auth
            logging.error(f"[Lỗi 401/403] Auth/Permission Error: {e}")
            return "❌ Lỗi xác thực: Vui lòng kiểm tra lại API Key hoặc quyền truy cập."

        # LỖI NHÓM 2: Lỗi từ phía Server hoặc Quá tải mạng (BẮT BUỘC RETRY)
        except (exceptions.ResourceExhausted, exceptions.ServiceUnavailable, exceptions.InternalServerError) as e:
            if attempt == max_retries - 1:
                logging.error(f"Đã thử tối đa {max_retries} lần nhưng vẫn thất bại do Server/Rate Limit.")
                return "⚠️ Hệ thống API đang quá tải (Lỗi 429/50x). Vui lòng thử lại sau ít phút."
            
            # Áp dụng Exponential Backoff kèm Jitter
            # Công thức: min(30, 2^attempt) + random(0, 1)
            sleep_time = min(30, 2 ** attempt) + random.uniform(0, 1)
            logging.warning(f"Server bận (Mã lỗi HTTP cho attempt {attempt+1}). Chờ {sleep_time:.2f}s để thử lại...")
            time.sleep(sleep_time)

        # NHÓM LỖI KHÁC: Các ngoại lệ không lường trước
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"Lỗi không xác định khi gọi Gemini: {e}")
                return f"❌ Đã xảy ra lỗi không xác định: {str(e)}"
            
            sleep_time = min(30, 2 ** attempt) + random.uniform(0, 1)
            time.sleep(sleep_time)

    return ""
