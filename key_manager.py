# File: key_manager.py
import itertools
import streamlit as st

# Danh sách 8 Key được "đóng gói" trực tiếp vào mã nguồn
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

@st.cache_resource
def get_key_cycler():
    """Tạo một vòng lặp vô tận qua 8 key, lưu vào RAM để không bị reset"""
    return itertools.cycle(MY_KEYS)

def get_next_key():
    """Hàm để các file khác gọi và lấy 1 key mới"""
    cycler = get_key_cycler()
    return next(cycler)
