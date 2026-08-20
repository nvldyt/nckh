# File: key_manager.py
import itertools
import os
import streamlit as st

# 1. Danh sách 8 Key Gemini
MY_KEYS = [
    "AQ.Ab8RN6JazLovPr7vvTFVBiUS8NKwAVzTxM3theZkK4Bj41MjYA",
    "AQ.Ab8RN6IojyD8oxt2G_QdadzK0cs7MMKOvCfQMEx9K6i-m7hUkg",
    "AQ.Ab8RN6J13twVBkGQlETIl68pTiUC-zs4Yv_zLvbOqjY4FOAU9g",
    "AQ.Ab8RN6If-EN_ZpABL7_YZu8H8Ziwfz5sK94kSaNSJxgRFSeBLg",
    "AQ.Ab8RN6LPAIgE8dbypq2pj9cea2dJDKE2B0hd0ivzCnInLfU3-A",
    "AQ.Ab8RN6KSY6NOw7_M6jBUJEpxTTCueWT4TaBPhQg0VT1w2sW9hA",
    "AQ.Ab8RN6I5cxAGLSXJgy76-zwSulkj1-rjOpKKny_ylvrkOmRWkA",
    "AQ.Ab8RN6JhBJ5w9bnl4pcVuf_NBh8gb2pwRq756ybmvXnar9Q18A"
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
