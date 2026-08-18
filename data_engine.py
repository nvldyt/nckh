# data_engine.py
# ============================================================
# BỘ MÁY XỬ LÝ DỮ LIỆU NGHIÊN CỨU (DATA ENGINE)
# Chuyên trách đọc, dọn dẹp, và chuẩn hóa DataFrame (Excel/CSV)
# ============================================================

import pandas as pd
from typing import Tuple, List

def auto_clean_data(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Hàm tự động dọn rác dữ liệu y khoa từ Excel.
    Tuyệt đối không dùng .capitalize() hay biến đổi case để bảo vệ mã ICD-10, HbA1c, eGFR.
    """
    logs = []
    df_clean = raw_df.copy()
    old_rows = df_clean.shape[0]
    
    # 1. Cắt khoảng trắng thừa ở tên cột
    df_clean.columns = df_clean.columns.str.strip()
    
    # 2. Xóa các cột rác "Unnamed" do phần mềm (như HIS) xuất dư
    unnamed_cols = [c for c in df_clean.columns if "unnamed" in str(c).lower()]
    if unnamed_cols:
        df_clean = df_clean.drop(columns=unnamed_cols)
        logs.append(f"🗑️ Đã xóa {len(unnamed_cols)} cột rác (Unnamed) do phần mềm xuất dư.")
        
    # 3. Xóa các dòng rỗng hoàn toàn (người dùng lỡ tay format dư dòng trong Excel)
    df_clean = df_clean.dropna(how='all')
    if df_clean.shape[0] < old_rows:
        logs.append(f"🗑️ Đã xóa {old_rows - df_clean.shape[0]} dòng trống hoàn toàn.")
        
    # 4. Chuẩn hóa khoảng trắng cho các ô chứa văn bản (strip)
    count_cleaned_cols = 0
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Bỏ qua các cột ngày tháng để tránh làm hỏng format sinh ra lỗi thống kê
            if not any(kw in str(col).lower() for kw in ['ngay', 'ngày', 'date', 'thoi', 'thời']):
                df_clean[col] = df_clean[col].apply(lambda x: str(x).strip() if pd.notnull(x) else x)
                count_cleaned_cols += 1
    
    if count_cleaned_cols > 0:
        logs.append(f"✨ Đã đồng nhất văn bản (cắt khoảng trắng 2 đầu) cho {count_cleaned_cols} cột.")
        
    return df_clean, logs
