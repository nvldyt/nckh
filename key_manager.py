# File: key_manager.py
import itertools
import streamlit as st

# ==========================================
# 1. CƠ CHẾ XOAY VÒNG KEY GROQ (Tab 1, Tab 5)
# ==========================================
def get_groq_keys():
    # Đọc từ Secrets, nếu không có thì dùng danh sách dự phòng cứng
    secret_str = st.secrets.get("GROQ_API_KEYS", "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")
    return [k.strip() for k in secret_str.split(",") if k.strip()]

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
    # Đọc từ Secrets, nếu không có dùng key mặc định của bạn
    secret_str = st.secrets.get("OPENROUTER_API_KEYS", "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")
    return [k.strip() for k in secret_str.split(",") if k.strip()]

@st.cache_resource
def get_or_cycler():
    return itertools.cycle(get_or_keys())

def get_next_or_key():
    """Lấy Key OpenRouter tiếp theo theo vòng lặp"""
    return next(get_or_cycler())
