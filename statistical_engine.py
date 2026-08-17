import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from typing import Any, Dict, List
import streamlit as st

# ============================================================
# 1. EXCEL – VALIDATION & THỐNG KÊ MÔ TẢ (CÓ CACHE)
# ============================================================

@st.cache_data(show_spinner=False)
def validate_dataframe(df: pd.DataFrame) -> List[str]:
    warnings = []
    if df.empty:
        warnings.append("File không có dòng dữ liệu.")

    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        warnings.append(f"Có {duplicate_rows} dòng trùng hoàn toàn.")

    missing_total = int(df.isna().sum().sum())
    if missing_total:
        warnings.append(f"Tổng số ô thiếu dữ liệu: {missing_total}.")

    duplicated_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicated_columns:
        warnings.append(f"Có tên cột trùng: {duplicated_columns}")

    return warnings

@st.cache_data(show_spinner=False)
def descriptive_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    s = df[column].dropna()
    counts = s.value_counts(dropna=False)
    total = len(s)

    result = pd.DataFrame({"Phân loại": counts.index.astype(str), "n": counts.values})
    result["%"] = (result["n"] / total * 100).round(2)
    return result

@st.cache_data(show_spinner=False)
def numeric_summary(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    s = pd.to_numeric(df[column], errors="coerce").dropna()
    if len(s) == 0:
        return {}
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "sd": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
        "median": float(s.median()),
        "q1": float(s.quantile(0.25)),
        "q3": float(s.quantile(0.75)),
        "min": float(s.min()),
        "max": float(s.max()),
    }

# ============================================================
# 2. CROSSTAB + CHI-SQUARE / FISHER (CÓ CACHE)
# ============================================================

@st.cache_data(show_spinner=False)
def crosstab_test(df: pd.DataFrame, independent: str, dependent: str) -> Dict[str, Any]:
    tmp = df[[independent, dependent]].dropna()
    table = pd.crosstab(tmp[independent], tmp[dependent])

    result = {
        "table": table, "test": None, "statistic": None,
        "p_value": None, "expected": None, "warning": None,
    }

    chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
    result["test"] = "Pearson Chi-square"
    result["statistic"] = float(chi2)
    result["p_value"] = float(p)
    result["expected"] = expected

    if table.shape == (2, 2) and (expected < 5).any():
        oddsratio, fisher_p = stats.fisher_exact(table)
        result["test"] = "Fisher's exact test"
        result["statistic"] = float(oddsratio)
        result["p_value"] = float(fisher_p)
        result["warning"] = "Một số tần số kỳ vọng <5; sử dụng Fisher's exact."
    elif (expected < 5).any():
        result["warning"] = "Có ô có tần số kỳ vọng <5. Cần cân nhắc gộp nhóm hoặc phương pháp kiểm định phù hợp hơn."

    return result

# ============================================================
# 3. SO SÁNH 2 NHÓM (CÓ CACHE)
# ============================================================

@st.cache_data(show_spinner=False)
def compare_two_groups(df: pd.DataFrame, group_col: str, value_col: str) -> Dict[str, Any]:
    tmp = df[[group_col, value_col]].dropna()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna()

    groups = tmp[group_col].unique().tolist()
    if len(groups) != 2:
        raise ValueError(f"Biến nhóm phải có đúng 2 mức; hiện có {len(groups)}.")

    g1 = tmp[tmp[group_col] == groups[0]][value_col]
    g2 = tmp[tmp[group_col] == groups[1]][value_col]

    def normal_ok(s):
        if 3 <= len(s) <= 5000:
            try:
                return stats.shapiro(s).pvalue > 0.05
            except Exception:
                return False
        return False

    is_normal = normal_ok(g1) and normal_ok(g2)

    if is_normal:
        stat, p = stats.ttest_ind(g1, g2, equal_var=False)
        test_name = "Independent t-test (Welch)"
    else:
        stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        test_name = "Mann-Whitney U"

    def describe(s):
        return {
            "n": int(len(s)),
            "mean": float(s.mean()),
            "sd": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
            "median": float(s.median()),
            "q1": float(s.quantile(0.25)),
            "q3": float(s.quantile(0.75)),
        }

    return {
        "group_names": [str(groups[0]), str(groups[1])],
        "group1_stats": describe(g1),
        "group2_stats": describe(g2),
        "test": test_name,
        "statistic": float(stat),
        "p_value": float(p),
        "normal_distribution_assumed": is_normal,
    }

# ============================================================
# 4. HỒI QUY LOGISTIC – LƯU TOÀN BỘ MÔ HÌNH (CÓ CACHE)
# ============================================================

@st.cache_data(show_spinner=False)
def binary_logistic_regression(df: pd.DataFrame, outcome: str, predictors: List[str]):
    cols = [outcome] + predictors
    tmp = df[cols].dropna().copy()

    if tmp.empty:
        raise ValueError("Không còn dữ liệu sau khi loại missing.")

    y_levels = tmp[outcome].dropna().unique().tolist()
    if len(y_levels) != 2:
        raise ValueError(f"Biến kết cục phải có đúng 2 mức; hiện có {len(y_levels)}.")

    mapping = {y_levels[0]: 0, y_levels[1]: 1}
    tmp["_Y_"] = tmp[outcome].map(mapping)

    formula_parts = []
    for p in predictors:
        if pd.api.types.is_numeric_dtype(tmp[p]):
            formula_parts.append(p)
        else:
            formula_parts.append(f"C(Q('{p}'))")

    formula = "_Y_ ~ " + " + ".join(formula_parts)
    model = smf.logit(formula=formula, data=tmp).fit(disp=False)

    conf = model.conf_int()
    params = model.params

    # Trả về toàn bộ các biến trong mô hình, không lọc bỏ biến nào theo p-value
    output = pd.DataFrame({
        "Biến": params.index,
        "OR": np.exp(params.values),
        "CI95% thấp": np.exp(conf.iloc[:, 0].values),
        "CI95% cao": np.exp(conf.iloc[:, 1].values),
        "p-value": model.pvalues.values,
    })

    return output, model.summary().as_text()
