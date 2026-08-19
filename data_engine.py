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
    - Bảo vệ tuyệt đối mã ICD-10, HbA1c, eGFR (không đổi case).
    - Tự động phát hiện và chuẩn hóa dấu phẩy thập phân kiểu Việt Nam (VD: 5,2 -> 5.2).
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
        
    # 3. Xóa các dòng rỗng hoàn toàn
    df_clean = df_clean.dropna(how='all')
    if df_clean.shape[0] < old_rows:
        logs.append(f"🗑️ Đã xóa {old_rows - df_clean.shape[0]} dòng trống hoàn toàn.")
        
    # 4. Chuẩn hóa văn bản, xử lý khoảng trắng và dấu phẩy thập phân
    count_cleaned_cols = 0
    count_fixed_decimals = 0
    
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Bỏ qua các cột ngày tháng để tránh làm hỏng format
            if not any(kw in str(col).lower() for kw in ['ngay', 'ngày', 'date', 'thoi', 'thời']):
                # Strip khoảng trắng 2 đầu
                df_clean[col] = df_clean[col].apply(lambda x: str(x).strip() if pd.notnull(x) else x)
                
                # TÍNH NĂNG MỚI: Tự động chuyển dấu phẩy thập phân (VD: 5,2 -> 5.2)
                sample_vals = df_clean[col].dropna().astype(str)
                if not sample_vals.empty:
                    # Kiểm tra xem trên 50% dữ liệu của cột có dạng số dùng dấu phẩy không
                    is_comma_decimal = sample_vals.str.match(r'^-?\d+,\d+$').mean() > 0.5
                    if is_comma_decimal:
                        df_clean[col] = df_clean[col].str.replace(',', '.')
                        # Thử ép sang kiểu số luôn nếu được
                        df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')
                        count_fixed_decimals += 1
                        
                count_cleaned_cols += 1
    
    if count_fixed_decimals > 0:
        logs.append(f"🔢 Đã tự động chuẩn hóa dấu phẩy thành dấu chấm thập phân cho {count_fixed_decimals} cột số liệu.")
    if count_cleaned_cols > 0:
        logs.append(f"✨ Đã đồng nhất văn bản cho {count_cleaned_cols} cột.")
        
    return df_clean, logs
