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
    "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",
    "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"
]

# --- CƠ CHẾ XOAY VÒNG KEY GROQ ---
@st.cache_resource
def get_key_cycler():
    return itertools.cycle(MY_KEYS)

def get_next_key():
    """Lấy Key Groq tiếp theo để tránh bị nghẽn (Rate Limit)"""
    return next(get_key_cycler())

