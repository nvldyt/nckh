# File: key_manager.py
import itertools
import streamlit as st

# ==========================================
# 1. DANH SÁCH KEY GROQ (Tab 1 MeSH, Tab 5)
# ==========================================
GROQ_KEYS = [
    "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"
]

@st.cache_resource
def get_groq_cycler():
    return itertools.cycle(GROQ_KEYS)

def get_next_groq_key():
    """Lấy Key Groq tiếp theo"""
    return next(get_groq_cycler())

def get_next_key():
    """Bí danh (alias) giữ lại để các module cũ (như Tab 1 MeSH) gọi không bị lỗi"""
    return get_next_groq_key()


# ==========================================
# 2. DANH SÁCH KEY OPENROUTER (Tab 6 - Qwen 72B)
# ==========================================
OPENROUTER_KEYS = [
    "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"
]

@st.cache_resource
def get_or_cycler():
    return itertools.cycle(OPENROUTER_KEYS)

def get_next_or_key():
    """Lấy Key OpenRouter tiếp theo cho Tab 6"""
    return next(get_or_cycler())
