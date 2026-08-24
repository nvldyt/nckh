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
    SKILL 05 & 09: Xây dựng Bảng Đặc điểm đối tượng (Table 1) bằng Pandas/SciPy thuần túy.
    Tự động tính n (%), Mean ± SD hoặc Median (IQR) kèm p-value đối chiếu.
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
            
        # Phân loại biến định tính hay định lượng
        is_categorical = (col in categorical) or (not pd.api.types.is_numeric_dtype(df_clean[col]))
        
        if is_categorical:
            # Tiêu đề nhóm biến định tính
            rows.append({"Đặc điểm": f"<b>{col}</b>", "Tổng số": "", **{g: "" for g in groups}, "p-value": ""})
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
                
                # Tính p-value (Chi-square) nếu có đúng 2 nhóm
                p_val_str = ""
                if pval and len(groups) == 2:
                    try:
                        contingency = pd.crosstab(df_clean[groupby].astype(str), df_clean[col] == v)
                        chi2, p, dof, ex = stats.chi2_contingency(contingency)
                        p_val_str = f"{p:.3f}" if p >= 0.001 else "<0.001"
                    except Exception:
                        p_val_str = ""
                row_dict["p-value"] = p_val_str
                rows.append(row_dict)
        else:
            # Biến định lượng
            is_nonnormal = nonnormal and col in nonnormal
            row_dict = {"Đặc điểm": f"<b>{col}</b>"}
            
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
            
            # Tính p-value định lượng (T-test hoặc Mann-Whitney U)
            p_val_str = ""
            if pval and len(groups) == 2:
                try:
                    g_vals = [df_clean[df_clean[groupby].astype(str) == g][col].dropna() for g in groups]
                    if is_nonnormal:
                        stat, p = stats.mannwhitneyu(g_vals[0], g_vals[1])
                    else:
                        stat, p = stats.ttest_ind(g_vals[0], g_vals[1])
                    p_val_str = f"{p:.3f}" if p >= 0.001 else "<0.001"
                except Exception:
                    p_val_str = ""
            row_dict["p-value"] = p_val_str
            rows.append(row_dict)

    out_df = pd.DataFrame(rows)
    return out_df.to_html(index=False, escape=False, classes="table table-striped table-bordered")


def run_advanced_comparison(
    df: pd.DataFrame, 
    target_col: str, 
    group_col: str, 
    test_type: str = 'auto'
) -> pd.DataFrame:
    df_clean = df.dropna(subset=[target_col, group_col])
    groups = df_clean[group_col].unique()
    if len(groups) == 2:
        group1 = df_clean[df_clean[group_col] == groups[0]][target_col]
        group2 = df_clean[df_clean[group_col] == groups[1]][target_col]
        res = pg.ttest(group1, group2) if test_type == 'ttest' or test_type == 'auto' else pg.mwu(group1, group2)
    elif len(groups) > 2:
        res = pg.anova(data=df_clean, dv=target_col, between=group_col) if test_type == 'anova' or test_type == 'auto' else pg.kruskal(data=df_clean, dv=target_col, between=group_col)
    else:
        raise ValueError("Biến phân nhóm phải có ít nhất 2 nhóm khác biệt.")
    return res


def run_survival_analysis(
    df: pd.DataFrame, 
    time_col: str, 
    event_col: str, 
    group_col: Optional[str] = None
) -> Tuple[Any, Any]:
    df_clean = df.dropna(subset=[time_col, event_col])
    kmf = KaplanMeierFitter()
    results = {}
    if group_col:
        df_clean = df_clean.dropna(subset=[group_col])
        groups = df_clean[group_col].unique()
        for group in groups:
            idx = (df_clean[group_col] == group)
            kmf.fit(df_clean[time_col][idx], df_clean[event_col][idx], label=str(group))
            results[f"Nhóm {group}"] = kmf.survival_function_
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
        kmf.fit(df_clean[time_col], df_clean[event_col], label="Toàn bộ mẫu")
        results['Tổng thể'] = kmf.survival_function_
    return kmf, results
