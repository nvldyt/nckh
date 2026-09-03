# File: key_manager.py
import os
import itertools
from pathlib import Path
import streamlit as st

def _get_keys_from_anywhere(secret_name):
    """
    Quét và lấy chuỗi key từ mọi môi trường (Online Streamlit, Local thuần, hoặc OS Env).
    Trả về danh sách (list) các key đã được làm sạch.
    """
    raw_key = None
    
    # 1. Ưu tiên 1: Lấy từ Streamlit Secrets (Khi chạy online hoặc local qua 'streamlit run')
    try:
        if hasattr(st, "secrets") and secret_name in st.secrets:
            raw_key = st.secrets[secret_name]
    except Exception:
        pass

    # 2. Ưu tiên 2: Đọc trực tiếp file .streamlit/secrets.toml (Khi chạy FastAPI / script Python độc lập)
    if not raw_key:
        try:
            secrets_path = Path(".streamlit/secrets.toml")
            if secrets_path.exists():
                import toml # Hoặc dùng 'import tomllib' với Python 3.11+
                with open(secrets_path, "r", encoding="utf-8") as f:
                    data = toml.load(f)
                    raw_key = data.get(secret_name)
        except Exception as e:
            print(f"Bỏ qua đọc file toml cục bộ do lỗi: {e}")

    # 3. Ưu tiên 3: Đọc từ biến môi trường của hệ điều hành
    if not raw_key:
        raw_key = os.getenv(secret_name)

    # Nếu vẫn không có, báo lỗi rõ ràng để dễ gỡ rối
    if not raw_key:
        raise ValueError(f"CRITICAL: Không tìm thấy biến '{secret_name}' ở bất kỳ cấu hình nào!")

    # Xử lý chuỗi (string) thành danh sách (list), cắt bỏ khoảng trắng thừa
    if isinstance(raw_key, str):
        return [k.strip() for k in raw_key.split(",") if k.strip()]
    elif isinstance(raw_key, list):
        return [str(k).strip() for k in raw_key if str(k).strip()]
    
    return [str(raw_key).strip()]


# ==========================================
# 1. CƠ CHẾ XOAY VÒNG KEY GROQ (Tab 1, Tab 5)
# ==========================================
def get_groq_keys():
    return _get_keys_from_anywhere("GROQ_API_KEYS")

@st.cache_resource
def get_groq_cycler():
    return itertools.cycle(get_groq_keys())

def get_next_groq_key():
    """Lấy Key Groq tiếp theo theo vòng lặp"""
    return next(get_groq_cycler())

def get_next_key():
    """Bí danh tương thích ngược cho các tab cũ"""
    return get_next_groq_key()


# ==========================================
# 2. CƠ CHẾ XOAY VÒNG KEY OPENROUTER (Tab 6)
# ==========================================
def get_or_keys():
    return _get_keys_from_anywhere("OPENROUTER_API_KEYS")

@st.cache_resource
def get_or_cycler():
    return itertools.cycle(get_or_keys())

def get_next_or_key():
    """Lấy Key OpenRouter tiếp theo theo vòng lặp"""
    return next(get_or_cycler())
