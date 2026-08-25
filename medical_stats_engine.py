import pandas as pd
import numpy as np
import pingouin as pg
from scipy import stats
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from typing import List, Dict, Any, Tuple, Optional

def generate_table_one(
    df: pd.DataFrame, 
    columns: List[str], 
    categorical: List[str], 
    groupby: str = None, 
    nonnormal: List[str] = None,
    pval: bool = True
) -> str:
    """
    SKILL 05 & 09: Xây dựng Bảng Đặc điểm đối tượng (Table 1) - Phiên bản Nâng cấp.
    Đã sửa lỗi tính p-value cho >2 nhóm, chuẩn hóa vị trí p-value định tính, 
    và dùng thuật toán ANOVA/Kruskal khi có nhiều nhóm.
    """
    if df.empty or not columns:
        raise ValueError("Dữ liệu trống hoặc chưa chọn biến số.")

    df_clean = df.copy()
    rows = []

    # Xác định các nhóm nếu có groupby
    groups = []
    if groupby and groupby in df_clean.columns:
        groups = [str(g) for g in df_clean[groupby].dropna().unique()]
        groups = sorted(groups)

    # Dòng tổng số mẫu (n)
    n_total = len(df_clean)
    n_row = {"Đặc điểm": "<b>n</b>", "Tổng số": str(n_total)}
    if groups:
        for g in groups:
            n_row[g] = str(len(df_clean[df_clean[groupby].astype(str) == g]))
    n_row["p-value"] = ""
    rows.append(n_row)

    for col in columns:
        if col not in df_clean.columns:
            continue
            
        # Phân loại biến
        is_categorical = (col in categorical) or (not pd.api.types.is_numeric_dtype(df_clean[col]))
        
        if is_categorical:
            # 1. TÍNH P-VALUE CHO TOÀN BỘ BIẾN ĐỊNH TÍNH (Đặt ở dòng tiêu đề)
            p_val_str = ""
            if pval and len(groups) >= 2:
                try:
                    contingency = pd.crosstab(df_clean[col], df_clean[groupby])
                    # Nếu là bảng 2x2, có thể cân nhắc dùng Fisher, ở đây dùng Chi-square có hiệu chỉnh
                    chi2, p, dof, ex = stats.chi2_contingency(contingency, correction=True)
                    p_val_str = f"{p:.3f}" if p >= 0.001 else "<0.001"
                except Exception:
                    p_val_str = ""
                    
            # Tiêu đề nhóm biến định tính kèm p-value
            rows.append({"Đặc điểm": f"<b>{col}</b>", "Tổng số": "", **{g: "" for g in groups}, "p-value": p_val_str})
            
            # Liệt kê các giá trị bên trong (không kèm p-value lặp lại)
            vals = df_clean[col].dropna().unique()
            for v in sorted(vals, key=str):
                row_dict = {"Đặc điểm": f"&nbsp;&nbsp;{v}"}
                sub_tot = df_clean[df_clean[col] == v]
                n_v = len(sub_tot)
                pct_v = (n_v / n_total * 100) if n_total > 0 else 0
                row_dict["Tổng số"] = f"{n_v} ({pct_v:.1f}%)"
                
                if groups:
                    for g in groups:
                        sub_g = df_clean[df_clean[groupby].astype(str) == g]
                        n_g = len(sub_g)
                        n_gv = len(sub_g[sub_g[col] == v])
                        pct_gv = (n_gv / n_g * 100) if n_g > 0 else 0
                        row_dict[g] = f"{n_gv} ({pct_gv:.1f}%)"
                row_dict["p-value"] = "" # Để trống để tránh rối mắt
                rows.append(row_dict)
                
        else:
            # 2. XỬ LÝ BIẾN ĐỊNH LƯỢNG
            is_nonnormal = nonnormal and col in nonnormal
            row_dict = {"Đặc điểm": f"<b>{col}</b>"}
            
            # Tính toán Mean/SD hoặc Median/IQR
            if is_nonnormal:
                med = df_clean[col].median()
                q25, q75 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
                row_dict["Tổng số"] = f"{med:.2f} ({q25:.2f} - {q75:.2f})"
                if groups:
                    for g in groups:
                        sub_g = df_clean[df_clean[groupby].astype(str) == g][col].dropna()
                        row_dict[g] = f"{sub_g.median():.2f} ({sub_g.quantile(0.25):.2f} - {sub_g.quantile(0.75):.2f})"
            else:
                mean_val, sd_val = df_clean[col].mean(), df_clean[col].std()
                row_dict["Tổng số"] = f"{mean_val:.2f} ± {sd_val:.2f}"
                if groups:
                    for g in groups:
                        sub_g = df_clean[df_clean[groupby].astype(str) == g][col].dropna()
                        row_dict[g] = f"{sub_g.mean():.2f} ± {sub_g.std():.2f}"
            
            # Tính p-value định lượng thông minh (Hỗ trợ >2 nhóm)
            p_val_str = ""
            if pval and len(groups) >= 2:
                try:
                    g_vals = [df_clean[df_clean[groupby].astype(str) == g][col].dropna() for g in groups]
                    if len(groups) == 2:
                        # 2 Nhóm: T-test hoặc Mann-Whitney U
                        if is_nonnormal:
                            stat, p = stats.mannwhitneyu(g_vals[0], g_vals[1])
                        else:
                            stat, p = stats.ttest_ind(g_vals[0], g_vals[1])
                    else:
                        # >= 3 Nhóm: ANOVA hoặc Kruskal-Wallis
                        if is_nonnormal:
                            stat, p = stats.kruskal(*g_vals)
                        else:
                            stat, p = stats.f_oneway(*g_vals)
                            
                    p_val_str = f"{p:.3f}" if p >= 0.001 else "<0.001"
                except Exception:
                    p_val_str = ""
            row_dict["p-value"] = p_val_str
            rows.append(row_dict)

    out_df = pd.DataFrame(rows)
    return out_df.to_html(index=False, escape=False, classes="table table-striped table-bordered")
