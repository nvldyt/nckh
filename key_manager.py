# File: key_manager.py
import itertools
import os
import streamlit as st

# ==========================================
# 1. DANH SÁCH KEY GROQ (Đã sửa lỗi ngoặc kép)
# ==========================================
MY_KEYS = [
    "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",
    "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",
    "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"
]

# ==========================================
# 2. DANH SÁCH KEY SERP_API (Tìm kiếm bài báo)
# ==========================================
SERP_KEYS = [
    "fd7e53594d38eb2d616337f2316fab9c6f029fcc845931576374a93e8cdc1dcb"
    # Nếu sau này anh có key SerpAPI thứ 2, thứ 3 thì cứ dán tiếp vào đây giống hệt như Groq nhé.
]

# --- CƠ CHẾ XOAY VÒNG KEY GROQ ---
@st.cache_resource
def get_key_cycler():
    return itertools.cycle(MY_KEYS)

def get_next_key():
    """Lấy Key Groq tiếp theo để tránh bị nghẽn (Rate Limit)"""
    return next(get_key_cycler())

# --- CƠ CHẾ XOAY VÒNG KEY SERP_API ---
@st.cache_resource
def get_serp_key_cycler():
    return itertools.cycle(SERP_KEYS)

def get_serpapi_key():
    """Lấy Key SerpAPI tiếp theo để tìm bài báo"""
    return next(get_serp_key_cycler())
