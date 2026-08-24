# File: key_manager.py
import itertools
import os
import streamlit as st

# 1. Danh sách 8 Key Gemini
MY_KEYS = [
    "AQ.Ab8RN6J9Ebk6H2rHBFpkSdSBOPXrcZ3TBjkiEX6x-GOnXLII9g"
]

# 2. Key SerpAPI
SERP_KEY = "f99c73f0a83c6e0ec159f8583534aa2d9deabdd339c44511323b83c15c4c6704"

@st.cache_resource
def get_key_cycler():
    return itertools.cycle(MY_KEYS)

def get_next_key():
    """Hàm gọi Key Gemini"""
    return next(get_key_cycler())

def get_serpapi_key():
    """Hàm gọi Key SerpAPI"""
    return SERP_KEY
