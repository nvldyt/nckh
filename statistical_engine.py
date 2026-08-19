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
        
    if df.shape[0] < 30:
        warnings.append(f"Cỡ mẫu khá nhỏ (n={df.shape[0]}). Kết quả thống kê có thể thiếu độ tin cậy.")

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
# 2. CROSSTAB + CHI-SQUARE / FISHER + OR & CI95% (CÓ CACHE)
# ============================================================

@st.cache_data(show_spinner=False)
def crosstab_test(df: pd.DataFrame, independent: str, dependent: str) -> Dict[str, Any]:
    tmp = df[[independent, dependent]].dropna()
    table = pd.crosstab(tmp[independent], tmp[dependent])

    if table.empty or table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError(f"Không đủ dữ liệu tạo bảng chéo cho {independent} và {dependent}")

    result = {
        "table": None, "test": None, "statistic": None,
        "p_value": None, "expected": None, "effect_size": None
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
    elif (expected < 5).any():
        result["test"] = "Chi-square (Cảnh báo: Có ô kỳ vọng < 5)"

    # TÍNH EFFECT SIZE (OR và 95% CI) cho bảng 2x2
    effect_size_str = "Không áp dụng (Bảng > 2x2)"
    if table.shape == (2, 2):
        try:
            a, b = table.iloc[0, 0], table.iloc[0, 1]
            c, d = table.iloc[1, 0], table.iloc[1, 1]
            if b == 0 or c == 0:
                effect_size_str = "Không thể tính OR (có ô bằng 0)"
            else:
                or_val = (a * d) / (b * c)
                se_ln_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
                ci_lower = np.exp(np.log(or_val) - 1.96 * se_ln_or)
                ci_upper = np.exp(np.log(or_val) + 1.96 * se_ln_or)
                effect_size_str = f"OR = {or_val:.2f} (95% CI: {ci_lower:.2f} - {ci_upper:.2f})"
        except Exception as e:
            effect_size_str = f"Lỗi tính OR: {str(e)}"
    
    result["effect_size"] = effect_size_str

    # Tạo bảng hiển thị đẹp kèm tỷ lệ %
    table_perc = pd.crosstab(tmp[independent], tmp[dependent], normalize='index') * 100
    display_df = table.astype(str) + " (" + table_perc.round(1).astype(str) + "%)"
    result["table"] = display_df.reset_index()

    return result

# ============================================================
# 3. SO SÁNH 2 NHÓM + MEAN/MEDIAN DIFFERENCE (CÓ CACHE)
# ============================================================

@st.cache_data(show_spinner=False)
def compare_two_groups(df: pd.DataFrame, group_col: str, value_col: str) -> Dict[str, Any]:
    tmp = df[[group_col, value_col]].dropna()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna()

    groups = tmp[group_col].unique().tolist()
    if len(groups) != 2:
        raise ValueError(f"Biến nhóm '{group_col}' phải có đúng 2 mức; hiện có {len(groups)}.")

    g1 = tmp[tmp[group_col] == groups[0]][value_col]
    g2 = tmp[tmp[group_col] == groups[1]][value_col]
    
    if len(g1) < 3 or len(g2) < 3:
        raise ValueError("Một trong hai nhóm có cỡ mẫu quá nhỏ (< 3) để so sánh.")

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
        
        # Tính Mean Difference và 95% CI
        mean_diff = g1.mean() - g2.mean()
        se_diff = np.sqrt(g1.var(ddof=1)/len(g1) + g2.var(ddof=1)/len(g2))
        ci_low = mean_diff - 1.96 * se_diff
        ci_high = mean_diff + 1.96 * se_diff
        effect_str = f"Mean Diff = {mean_diff:.2f} (95% CI: {ci_low:.2f} - {ci_high:.2f})"
        
        g1_stat = f"{g1.mean():.2f} ± {g1.std(ddof=1):.2f}"
        g2_stat = f"{g2.mean():.2f} ± {g2.std(ddof=1):.2f}"
    else:
        stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        test_name = "Mann-Whitney U"
        
        # Tính Median Difference
        median_diff = g1.median() - g2.median()
        effect_str = f"Median Diff = {median_diff:.2f}"
        
        g1_stat = f"{g1.median():.2f} ({g1.quantile(0.25):.2f}-{g1.quantile(0.75):.2f})"
        g2_stat = f"{g2.median():.2f} ({g2.quantile(0.25):.2f}-{g2.quantile(0.75):.2f})"

    return {
        "group_names": [str(groups[0]), str(groups[1])],
        "group1_stats": g1_stat,
        "group2_stats": g2_stat,
        "test": test_name,
        "statistic": float(stat),
        "p_value": float(p),
        "effect_size": effect_str,
    }

# ============================================================
# 4. HỒI QUY LOGISTIC – BẮT LỖI MẠNH MẼ (CÓ CACHE)
# ============================================================

@st.cache_data(show_spinner=False)
def binary_logistic_regression(df: pd.DataFrame, outcome: str, predictors: List[str]) -> Tuple[pd.DataFrame, str]:
    import statsmodels.formula.api as smf
    import numpy as np
    import pandas as pd

    # 1. Chuẩn bị dữ liệu: Lọc bỏ dòng chứa dữ liệu trống (Missing)
    tmp = df[[outcome] + predictors].dropna().copy()
    if tmp.empty:
        raise ValueError("Không còn dữ liệu sau khi loại bỏ ô trống (Missing).")

    # 2. Kiểm tra biến kết cục phải là nhị phân (2 mức)
    y_levels = tmp[outcome].unique()
    if len(y_levels) != 2:
        raise ValueError(f"Biến kết cục '{outcome}' phải có đúng 2 mức (ví dụ: Có/Không, 0/1). Hiện tại có: {len(y_levels)} mức.")

    # 3. Xây dựng công thức hồi quy (Formula)
    # Tự động nhận diện biến định lượng (không bọc C) và biến phân loại (bọc C)
    formula_parts = []
    for p in predictors:
        if pd.api.types.is_numeric_dtype(tmp[p]):
            formula_parts.append(f"Q('{p}')")
        else:
            formula_parts.append(f"C(Q('{p}'))") # C() chỉ định đây là biến phân loại cho statsmodels

    formula = f"Q('{outcome}') ~ " + " + ".join(formula_parts)
    
    # 4. Chạy mô hình và BẮT LỖI THỐNG KÊ (Robust Error Handling)
    try:
        model = smf.logit(formula=formula, data=tmp).fit(disp=0)
    except Exception as e:
        err = str(e).lower()
        if "singular" in err or "linalg" in err:
            raise ValueError("Lỗi ma trận (Singular Matrix): Đa cộng tuyến quá mạnh hoặc cỡ mẫu quá nhỏ cho số biến đã chọn.")
        if "separation" in err:
            raise ValueError("Lỗi Perfect Separation: Một biến dự báo phân tách hoàn hảo kết cục (ví dụ: mọi bệnh nhân nhóm A đều khỏi bệnh).")
        raise ValueError(f"Lỗi hệ thống khi chạy mô hình: {str(e)}")

    # 5. Trích xuất kết quả OR, 95% CI và P-value
    conf = model.conf_int()
    params = model.params
    
    # Tạo DataFrame kết quả
    output = pd.DataFrame({
        "Biến": params.index,
        "OR": np.exp(params.values).round(2),
        "CI_Lower": np.exp(conf.iloc[:, 0].values),
        "CI_Upper": np.exp(conf.iloc[:, 1].values),
        "P-value": model.pvalues.values,
    })
    
    # Định dạng cột 95% CI
    output["95% CI"] = output.apply(lambda row: f"{row['CI_Lower']:.2f} - {row['CI_Upper']:.2f}", axis=1)
    output = output.drop(columns=["CI_Lower", "CI_Upper"])
    
    # Loại bỏ dòng Intercept (không ý nghĩa lâm sàng)
    output = output[~output["Biến"].str.contains("Intercept")].reset_index(drop=True)
    
    # Làm đẹp tên biến trong bảng kết quả (xóa bớt ký tự thừa do formula tạo ra)
    output["Biến"] = output["Biến"].str.replace("Q('", "").str.replace("')", "").str.replace("C(", "").str.replace(")", "")

    return output, "Mô hình hồi quy Logistic đa biến thành công."

# ============================================================
# 5. TỔNG HỢP BẢNG ĐẶC ĐIỂM CHUNG CHUẨN LUẬN VĂN (BASELINE)
# ============================================================

@st.cache_data(show_spinner=False)
def create_clinical_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tự động quét các cột lâm sàng phổ biến (Tuổi, BMI, CrCl) 
    và tạo ra các cột phân nhóm chuẩn y khoa nếu dữ liệu gốc chưa có.
    """
    df = df.copy()
    
    # Phân nhóm Tuổi
    age_col = next((c for c in df.columns if str(c).strip().lower() in ['tuổi', 'tuoi', 'age']), None)
    if age_col and pd.api.types.is_numeric_dtype(df[age_col]):
        bins = [0, 18, 60, float('inf')]
        labels = ["< 18 tuổi", "18 - 60 tuổi", "> 60 tuổi"]
        df["Nhóm tuổi"] = pd.cut(df[age_col], bins=bins, labels=labels, right=False)

    # Phân nhóm chức năng thận (CrCl / Độ thanh thải)
    crcl_col = next((c for c in df.columns if any(k in str(c).strip().lower() for k in ['crcl', 'thanh thải', 'clearance'])), None)
    if crcl_col and pd.api.types.is_numeric_dtype(df[crcl_col]):
        bins = [0, 30, 50, float('inf')]
        labels = ["< 30 mL/min", "30 - < 50 mL/min", ">= 50 mL/min"]
        df["Nhóm chức năng thận (CrCl)"] = pd.cut(df[crcl_col], bins=bins, labels=labels, right=False)
        
    # Phân nhóm BMI (Chuẩn Châu Á)
    bmi_col = next((c for c in df.columns if 'bmi' in str(c).strip().lower()), None)
    if bmi_col and pd.api.types.is_numeric_dtype(df[bmi_col]):
        bins = [0, 18.5, 23, 25, float('inf')]
        labels = ["Gầy (< 18.5)", "Bình thường (18.5 - < 23)", "Tiền béo phì (23 - < 25)", "Béo phì (>= 25)"]
        df["Phân loại BMI"] = pd.cut(df[bmi_col], bins=bins, labels=labels, right=False)

    return df

@st.cache_data(show_spinner=False)
def generate_baseline_table(df: pd.DataFrame, cat_vars: list, num_vars: list) -> pd.DataFrame:
    """
    Tổng hợp bảng Đặc điểm chung chuẩn luận văn:
    - Tự động chạy Shapiro-Wilk chọn Mean ± SD hoặc Median [IQR].
    - Tính tỷ lệ % trên mẫu số hợp lệ (loại bỏ NaN).
    - Gom nhóm hiếm < 5% vào mục "Khác".
    - Định dạng Plain Text viết hoa tên biến (chống lỗi Markdown trong Word).
    """
    rows = []
    
    # 1. XỬ LÝ BIẾN ĐỊNH LƯỢNG
    for var in num_vars:
        if var in df.columns:
            clean_series = df[var].dropna()
            valid_n = len(clean_series)
            if valid_n == 0: continue
            
            # Kiểm định phân phối chuẩn (Shapiro-Wilk)
            is_normal = False
            if valid_n >= 3:
                try:
                    stat, p = stats.shapiro(clean_series)
                    if p >= 0.05:
                        is_normal = True
                except:
                    pass
            
            if is_normal:
                mean_val = clean_series.mean()
                sd_val = clean_series.std(ddof=1)
                rows.append({
                    "Đặc điểm": f"{var.upper()} (Mean ± SD)",
                    "Số lượng / Thống kê": f"n = {valid_n}",
                    "Tỷ lệ (%) / Giá trị": f"{mean_val:.2f} ± {sd_val:.2f}"
                })
            else:
                median_val = clean_series.median()
                q1 = clean_series.quantile(0.25)
                q3 = clean_series.quantile(0.75)
                rows.append({
                    "Đặc điểm": f"{var.upper()} (Median [IQR])",
                    "Số lượng / Thống kê": f"n = {valid_n}",
                    "Tỷ lệ (%) / Giá trị": f"{median_val:.2f} [{q1:.2f} - {q3:.2f}]"
                })

    # 2. XỬ LÝ BIẾN PHÂN LOẠI
    for var in cat_vars:
        if var in df.columns:
            clean_series = df[var].dropna()
            valid_n = len(clean_series)
            if valid_n == 0: continue
            
            # Tên biến (Viết hoa để nổi bật trong Word thay vì dùng **)
            rows.append({
                "Đặc điểm": str(var).upper(),
                "Số lượng / Thống kê": f"N = {valid_n}",
                "Tỷ lệ (%) / Giá trị": ""
            })
            
            value_counts = clean_series.value_counts()
            
            # Gom nhóm hiếm < 5% vào "Khác"
            other_count = 0
            main_categories = []
            for val, count in value_counts.items():
                pct = (count / valid_n) * 100
                # Chỉ gộp nếu biến có nhiều hơn 3 nhóm (để tránh gộp sai biến nhị phân như Giới tính)
                if pct < 5.0 and len(value_counts) > 3:
                    other_count += count
                else:
                    main_categories.append((val, count, pct))
            
            for val, count, pct in main_categories:
                rows.append({
                    "Đặc điểm": f"  - {val}",
                    "Số lượng / Thống kê": str(count),
                    "Tỷ lệ (%) / Giá trị": f"{pct:.1f}%"
                })
            
            if other_count > 0:
                other_pct = (other_count / valid_n) * 100
                rows.append({
                    "Đặc điểm": "  - Khác",
                    "Số lượng / Thống kê": str(other_count),
                    "Tỷ lệ (%) / Giá trị": f"{other_pct:.1f}%"
                })

    return pd.DataFrame(rows)
