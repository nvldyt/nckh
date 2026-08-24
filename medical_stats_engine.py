import pandas as pd
import numpy as np
import pingouin as pg
from tableone import TableOne
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
) -> pd.DataFrame:
    """
    SKILL 05 & 09: Xây dựng Bảng Đặc điểm đối tượng (Table 1).
    Tự động tính n (%), Mean ± SD (phân bố chuẩn) hoặc Median (IQR) (phân bố không chuẩn).
    """
    if df.empty or not columns:
        raise ValueError("Dữ liệu trống hoặc chưa chọn biến số.")
        
    # Đảm bảo các biến định tính được định dạng đúng là string/category
    for cat in categorical:
        if cat in df.columns:
            df[cat] = df[cat].astype(str)

    try:
        mytable = TableOne(
            df, 
            columns=columns, 
            categorical=categorical, 
            groupby=groupby, 
            nonnormal=nonnormal, # Biến không phân bố chuẩn sẽ tự động dùng Median (IQR)
            pval=pval,
            missing=False,
            decimals={"continuous": 2, "categorical": 1}
        )
        return mytable.table_html
    except Exception as e:
        return f"Lỗi khi tạo Bảng 1: {str(e)}"


def run_advanced_comparison(
    df: pd.DataFrame, 
    target_col: str, 
    group_col: str, 
    test_type: str = 'auto'
) -> pd.DataFrame:
    """
    SKILL 06: Kiểm định so sánh hai hoặc nhiều nhóm.
    Tự động xuất p-value, Khoảng tin cậy (CI 95%) và Cỡ tác động (Effect size).
    """
    # Lọc bỏ các hàng có giá trị khuyết (missing) ở biến mục tiêu hoặc biến phân nhóm
    df_clean = df.dropna(subset=[target_col, group_col])
    
    groups = df_clean[group_col].unique()
    
    if len(groups) == 2:
        # So sánh 2 nhóm (T-test hoặc Mann-Whitney)
        group1 = df_clean[df_clean[group_col] == groups[0]][target_col]
        group2 = df_clean[df_clean[group_col] == groups[1]][target_col]
        
        if test_type == 'ttest' or test_type == 'auto':
            # Kiểm định t-test (Kèm CI 95% và Cohen's d)
            res = pg.ttest(group1, group2)
        else:
            # Kiểm định phi tham số Mann-Whitney U
            res = pg.mwu(group1, group2)
            
    elif len(groups) > 2:
        # So sánh > 2 nhóm (ANOVA hoặc Kruskal-Wallis)
        if test_type == 'anova' or test_type == 'auto':
            res = pg.anova(data=df_clean, dv=target_col, between=group_col)
        else:
            res = pg.kruskal(data=df_clean, dv=target_col, between=group_col)
    else:
        raise ValueError("Biến phân nhóm phải có ít nhất 2 nhóm khác biệt.")

    # Trả về bảng kết quả chi tiết của Pingouin
    return res


def run_survival_analysis(
    df: pd.DataFrame, 
    time_col: str, 
    event_col: str, 
    group_col: Optional[str] = None
) -> Tuple[Any, Any]:
    """
    Phân tích sống còn (Kaplan-Meier) và kiểm định Log-rank.
    Phù hợp đánh giá thời gian đến khi xuất hiện biến cố (VD: Tổn thương thận, khỏi bệnh).
    """
    df_clean = df.dropna(subset=[time_col, event_col])
    kmf = KaplanMeierFitter()
    
    results = {}
    
    if group_col:
        df_clean = df_clean.dropna(subset=[group_col])
        groups = df_clean[group_col].unique()
        
        # Vẽ Kaplan-Meier cho từng nhóm
        for i, group in enumerate(groups):
            idx = (df_clean[group_col] == group)
            kmf.fit(df_clean[time_col][idx], df_clean[event_col][idx], label=str(group))
            results[f"Nhóm {group}"] = kmf.survival_function_
            
        # Kiểm định Log-rank test
        if len(groups) == 2:
            idx1 = (df_clean[group_col] == groups[0])
            idx2 = (df_clean[group_col] == groups[1])
            res_logrank = logrank_test(
                df_clean[time_col][idx1], df_clean[time_col][idx2],
                event_observed_A=df_clean[event_col][idx1], 
                event_observed_B=df_clean[event_col][idx2]
            )
            results['Log-rank p-value'] = res_logrank.p_value
    else:
        # Tổng thể (Không chia nhóm)
        kmf.fit(df_clean[time_col], df_clean[event_col], label="Toàn bộ mẫu")
        results['Tổng thể'] = kmf.survival_function_
        
    return kmf, results
