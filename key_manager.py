# File: key_manager.py
import itertools
import os
import streamlit as st

# 1. Đưa chìa khóa Groq (gsk_...) của anh vào đây
MY_KEYS = [
    "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,
    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,
    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"
]

# 2. Key SerpAPI
SERP_KEY = "2a58c98ff036322c9c40f0154599496b7af2d78a3dd0d1eab0383e479d255cd8"

@st.cache_resource
def get_key_cycler():
    return itertools.cycle(MY_KEYS)

def get_next_key():
    """Hàm lấy Key tiếp theo"""
    return next(get_key_cycler())

def get_serpapi_key():
    """Hàm gọi Key SerpAPI"""
    return SERP_KEY
