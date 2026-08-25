# statistics_tab.py
import streamlit as st
import pandas as pd
import gc
from docx import Document
import io

def render_statistics_tab(
    ui_key, 
    safe_read_excel, 
    auto_clean_data, 
    render_chat_assistant, 
    validate_dataframe, 
    create_word_document, 
    StudyObjective, 
    TableSelectionEngine, 
    NarrativePlanner, 
    assemble_results_and_discussion_chapter, 
    format_numbered_citations, 
    get_citation_engine, 
    citation_bibliography_wrapper, 
    descriptive_table, 
    numeric_summary, 
    crosstab_test, 
    large_batch_ok, 
    compare_two_groups, 
    binary_logistic_regression, 
    upsert_candidate, 
    call_gemini, 
    BASE_SYSTEM_RULES, 
    DEFAULT_MODEL
):
    st.header("📊 Phân tích số liệu bệnh án (SPSS Mini)")
    excel_file = st.file_uploader("Tải file Excel", type=["xlsx", "xls"], key=ui_key("excel_uploader_tab4"))
    
    if excel_file is not None:
        with st.spinner("Đang dọn dẹp và nạp dữ liệu..."):
            raw = safe_read_excel(excel_file)
        if not raw.empty:
            st.success(f"✅ Nạp thành công: {raw.shape[0]} dòng và {raw.shape[1]} cột.")
            st.dataframe(raw.head(50), use_container_width=True)
            st.session_state["excel_data"] = raw
            try:
                with st.spinner("Đang dọn dẹp và chuẩn hóa dữ liệu bằng Data Engine..."): 
                    df, logs = auto_clean_data(raw)
                st.session_state["excel_df"] = df
                st.session_state["clean_logs"] = logs
            except Exception as exc: 
                st.error(f"Lỗi khi chuẩn hóa dữ liệu: {exc}")
                
    render_chat_assistant()
    df = st.session_state.get("excel_df")
    
    if df is not None and not df.empty:
        if st.session_state.get("clean_logs"):
            with st.expander("🛠️ Xem nhật ký tự động dọn dẹp dữ liệu", expanded=True):
                for log in st.session_state["clean_logs"]: st.write(log)
                
        for item in validate_dataframe(df) or []: st.warning(item)
        with st.expander("Xem dữ liệu sau khi chuẩn hóa"): st.dataframe(df, use_container_width=True)
        
        st.write("---")
        st.markdown(f"### 🛒 Giỏ kết quả: **{len(st.session_state.get('result_cart', []))}** bảng đã lưu")
        st.info("💡 Mỗi khi anh bấm các nút thống kê bên dưới, kết quả tự động được nạp vào Giỏ này để lát nữa AI tuyển chọn hoặc xuất ra Word.")
        
        gc1, gc2 = st.columns(2)
        with gc1:
            if st.button("🗑️ Xóa toàn bộ Giỏ kết quả", use_container_width=True, key=ui_key("clear_result_cart")):
                st.session_state["result_cart"] = []
                st.session_state["saved_tables"] = {}
                st.session_state["selection_decisions"] = []
                st.session_state["narrative_plan"] = {}
                gc.collect()
                st.rerun()
        with gc2:
            saved = st.session_state.get("saved_tables", {})
            if saved:
                md = []
                for tid, dft in saved.items():
                    h = "| " + " | ".join(map(str, dft.columns)) + " |"
                    sep = "|" + "|".join(["---"] * len(dft.columns)) + "|"
                    rows = ["| " + " | ".join(map(str, row.values)) + " |" for _, row in dft.iterrows()]
                    md.append(f"### Kết quả Thống kê: {tid}\n\n" + "\n".join([h, sep] + rows))
                wd = create_word_document("Phụ lục Số liệu Thống kê (Xuất từ Giỏ kết quả)", "\n\n".join(md))
                st.download_button("📥 Tải TẤT CẢ bảng ra file Word", data=wd, file_name="Phu_luc_So_lieu_Thong_ke.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True, key=ui_key("download_all_tables"))
            else: 
                st.button("📥 Tải TẤT CẢ bảng ra file Word", disabled=True, use_container_width=True, key=ui_key("download_all_tables_disabled"))

        st.write("---")
        st.subheader("📋 Bộ máy tuyển chọn & Sắp xếp bảng cho Chương Kết quả")
        with st.expander("🎯 Khai báo Mục tiêu nghiên cứu"):
            o1 = st.text_input("Mục tiêu 1", value="ĐẶC ĐIỂM BỆNH NHÂN NGHIÊN CỨU", key=ui_key("obj_1"))
            o2 = st.text_input("Mục tiêu 2", value="PHÂN TÍCH THỰC TRẠNG SỬ DỤNG THUỐC", key=ui_key("obj_2"))
            objectives = [
                StudyObjective(id="MT1", title=o1, keywords=["tuổi", "tuoi", "giới", "gioi", "bệnh", "benh", "đặc điểm", "nhân khẩu", "bmi", "SoBHYT", "NgaySinh"]),
                StudyObjective(id="MT2", title=o2, keywords=["thuốc", "thuoc", "phù hợp", "phu hop", "liều", "lieu", "chỉ định", "chi dinh", "hoạt chất", "icd", "TenHang"])
            ]
            
        if st.button("🚀 Chạy Table Selection Engine & Lập mạch kể chuyện", type="primary", key=ui_key("run_engine")):
            if not st.session_state["result_cart"]: 
                st.error("❌ Giỏ kết quả đang trống! Anh cần bấm các nút thống kê để nạp số liệu vào Giỏ trước.")
            else:
                try:
                    dec = TableSelectionEngine(objectives, st.session_state["result_cart"]).run()
                    st.session_state["selection_decisions"] = dec
                    st.session_state["narrative_plan"] = NarrativePlanner.build(dec)
                    st.success("✅ Đã hoàn thành tuyển chọn, lọc trùng và sắp xếp cấu trúc Chương Kết quả!")
                except Exception as exc: 
                    st.error(f"❌ Table Selection Engine lỗi: {exc}")
                    
        if st.session_state.get("selection_decisions"):
            rows = []
            for d in st.session_state["selection_decisions"]: 
                rows.append({"Thứ tự": d.recommended_order or "Phụ lục", "Mức độ": d.priority.value, "Hình thức": d.presentation.value, "Điểm": d.total_score, "Tiêu đề bảng": d.title, "Lý do đề xuất": d.reason})
            st.write("### 📊 Bảng tổng hợp đề xuất cấu trúc Chương 3")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.write("### 📖 Mạch kể chuyện (Result Story / Narrative Plan)")
            st.json(st.session_state["narrative_plan"])

        st.write("---")
        st.subheader("🚀 Trình lắp ráp tự động Chương Kết quả & Bàn luận (Auto-Assembler Agent)")
        st.info("💡 Hệ thống sẽ tự động kích hoạt Loop Agent: Duyệt qua từng bảng theo mạch kể chuyện, tự viết nhận xét, tự tìm bằng chứng đối chiếu và lắp ráp thành toàn bộ bản thảo 2 chương lớn.")
        
        if st.button("🪄 Tự động lập bản thảo toàn bộ Chương 3 & Chương 4", type="primary", key=ui_key("btn_auto_assemble")):
            if not st.session_state.get("selection_decisions") or not st.session_state.get("saved_tables"):
                st.warning("⚠️ Bạn cần chạy 'Table Selection Engine' để thiết lập mạch kể chuyện và lưu bảng vào Giỏ trước!")
            else:
                try:
                    with st.spinner("Agent đang tự động quét, viết nhận xét, truy xuất y văn và lắp ráp hai chương..."):
                        a3, a4 = assemble_results_and_discussion_chapter(
                            selection_decisions=st.session_state["selection_decisions"], 
                            saved_tables=st.session_state["saved_tables"], 
                            chunks=st.session_state.get("chunks", []), 
                            embeddings=st.session_state.get("embeddings"), 
                            bm25=st.session_state.get("bm25"), 
                            citation_engine=get_citation_engine(), 
                            study_context=st.session_state.get("study_context", {})
                        )
                        full_draft = a3 + "\n\n---SPLIT_CHAPTER---\n\n" + a4
                        clean_draft = format_numbered_citations(full_draft)
                        clean_a3, clean_a4 = clean_draft.split("\n\n---SPLIT_CHAPTER---\n\n")
                        
                    st.session_state["assembled_ch3"] = clean_a3
                    st.session_state["assembled_ch4"] = clean_a4
                    st.success("🎉 Đã lắp ráp thành công toàn bộ bản thảo hai chương!")
                except Exception as exc: 
                    st.error(f"❌ Auto-Assembler lỗi: {exc}")
                    
        if st.session_state.get("assembled_ch3") and st.session_state.get("assembled_ch4"):
            t3, t4 = st.tabs(["📄 Chương 3: Kết quả", "📄 Chương 4: Bàn luận"])
            with t3:
                st.markdown(st.session_state["assembled_ch3"])
                d3 = create_word_document("Chương 3: Kết quả nghiên cứu", st.session_state["assembled_ch3"])
                st.download_button("📥 Tải Chương 3 ra file Word", data=d3, file_name="Chuong_3_Ket_qua.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", key=ui_key("dl_ch3"))
            with t4:
                st.markdown(st.session_state["assembled_ch4"])
                d4 = create_word_document("Chương 4: Bàn luận", st.session_state["assembled_ch4"], citation_bibliography_wrapper())
                st.download_button("📥 Tải Chương 4 ra file Word (kèm TLTK)", data=d4, file_name="Chuong_4_Ban_luan.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", key=ui_key("dl_ch4"))

        # =====================================================================
        # BẢNG 1 & THỐNG KÊ MÔ TẢ CHI TIẾT
        # =====================================================================
        st.write("---")
        st.subheader("⭐ ĐẶC BIỆT: Tự động lập Bảng 1 (Đặc điểm đối tượng nghiên cứu)")
        st.info("💡 Bảng sẽ tự động tính n (%), Mean ± SD cho biến chuẩn và Median (IQR) cho biến không chuẩn.")
        
        all_cols = df.columns.tolist()
        c1, c2 = st.columns(2)
        with c1:
            columns_to_show = st.multiselect("📍 Chọn các biến số cần đưa vào Bảng 1:", options=all_cols, key=ui_key("t1_cols"))
            categorical_vars = st.multiselect("🏷️ Chọn biến Định tính (Tính n, %):", options=columns_to_show, key=ui_key("t1_cats"))
        with c2:
            non_normal_vars = st.multiselect("📉 Chọn biến Định lượng phân bố KHÔNG chuẩn (Tính Median, IQR):", options=[c for c in columns_to_show if c not in categorical_vars], key=ui_key("t1_nonnorm"))
            groupby_var = st.selectbox("✂️ Biến chia nhóm (Tùy chọn tính p-value):", options=["Không chia nhóm"] + all_cols, key=ui_key("t1_group"))
        
        if st.button("🚀 Trích xuất Bảng 1 (Tự động nạp vào Giỏ)", type="primary", key=ui_key("btn_table_one")):
            if not columns_to_show: 
                st.warning("⚠️ Vui lòng chọn ít nhất 1 biến số để hiển thị!")
            else:
                with st.spinner("Đang chạy thuật toán thống kê y khoa..."):
                    try:
                        from medical_stats_engine import generate_table_one
                        actual_groupby = None if groupby_var == "Không chia nhóm" else groupby_var
                        
                        html_table = generate_table_one(
                            df=df, columns=columns_to_show, categorical=categorical_vars,
                            groupby=actual_groupby, nonnormal=non_normal_vars
                        )
                        
                        st.markdown(f"<div style='background-color: white; padding: 15px; border-radius: 8px; color: black; overflow-x: auto;'>{html_table}</div>", unsafe_allow_html=True)
                        
                        parsed_df = pd.read_html(io.StringIO(html_table))[0] 
                        rid = "TABLE_1_BASELINE"
                        st.session_state["saved_tables"][rid] = parsed_df
                        upsert_candidate(rid, "Đặc điểm chung của đối tượng nghiên cứu (Bảng 1)", "baseline", columns_to_show, 5.0, 5.0, 5.0)
                        
                        st.success("✅ Đã khởi tạo và nạp Bảng 1 vào Giỏ kết quả thành công!")
                    except Exception as e:
                        st.error(f"⚠️ Lỗi phân tích: {e}")

        st.write("---")
        st.subheader("1. Thống kê mô tả (biến phân loại)")
        desc_vars = st.multiselect("Chọn biến phân loại", all_cols, key="tab7_unique_desc_vars_key")
        if st.button("Tính tần số và tỷ lệ (Tự động nạp vào Giỏ)", key=ui_key("calc_desc")):
            if not desc_vars: st.warning("Vui lòng chọn ít nhất 1 biến.")
            else:
                n = 0
                for var in desc_vars:
                    try:
                        r = descriptive_table(df, var)
                        if r is not None and not r.empty:
                            n += 1
                            st.markdown(f"**► Biến: {var}**")
                            st.dataframe(r, use_container_width=True)
                            rid = f"DESC_{var}"
                            st.session_state["saved_tables"][rid] = r
                            upsert_candidate(rid, f"Đặc điểm phân bố của biến {var}", "demographic", [var], 3.5, 4.0, 3.0)
                    except Exception as exc: 
                        st.error(f"⚠️ Không thể phân tích [{var}]: {exc}")
                st.success(f"✅ Đã nạp {n} bảng mô tả vào Giỏ!")

        st.write("---")
        st.subheader("2. Biến định lượng — Mô tả")
        numeric_candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_candidates:
            num_vars = st.multiselect("Chọn biến định lượng", numeric_candidates, key=ui_key("num_vars"))
            if st.button("Tính Mean/SD và Median/IQR (Tự động nạp vào Giỏ)", key=ui_key("calc_num")):
                if not num_vars: st.warning("Vui lòng chọn ít nhất 1 biến.")
                else:
                    n = 0
                    for var in num_vars:
                        try:
                            s = numeric_summary(df, var)
                            if s:
                                n += 1
                                num_df = pd.DataFrame([{"N": s["n"], "Mean ± SD": f"{s['mean']:.2f} ± {s['sd']:.2f}", "Median (IQR)": f"{s['median']:.2f} ({s['q1']:.2f} - {s['q3']:.2f})", "Min-Max": f"{s['min']:.2f} - {s['max']:.2f}"}])
                                st.markdown(f"**► Biến: {var}**")
                                st.dataframe(num_df, use_container_width=True)
                                rid = f"NUM_{var}"
                                st.session_state["saved_tables"][rid] = num_df
                                upsert_candidate(rid, f"Đặc điểm định lượng của biến {var}", "baseline", [var], 3.5, 4.0, 3.0)
                        except Exception as exc: 
                            st.error(f"⚠️ Không thể tính [{var}]: {exc}")
                    st.success(f"✅ Đã nạp {n} bảng định lượng vào Giỏ!")

        st.write("---")
        st.subheader("3. Bảng chéo và kiểm định (Chi-square / Fisher / OR / CI95)")
        a, b = st.columns(2)
        with a:
            all_d = st.checkbox("✅ Chọn tất cả biến phụ thuộc", key=ui_key("chk_all_deps"))
            dkey = ui_key("cross_deps_all") if all_d else ui_key("cross_deps_manual")
            deps = st.multiselect("Các biến phụ thuộc", all_cols, default=all_cols if all_d else [], key=dkey)
        with b:
            all_i = st.checkbox("✅ Chọn tất cả biến độc lập", key=ui_key("chk_all_indeps"))
            ikey = ui_key("cross_indeps_all") if all_i else ui_key("cross_indeps_manual")
            indeps = st.multiselect("Các biến độc lập cần đối chiếu", all_cols, default=all_cols if all_i else [], key=ikey)
            
        if st.button("Quét Crosstab + Kiểm định (Nạp TẤT CẢ vào Giỏ)", key=ui_key("calc_cross")):
            if not deps or not indeps: st.warning("Vui lòng chọn ít nhất 1 biến ở mỗi mục.")
            elif not large_batch_ok("Crosstab + Kiểm định", len({frozenset([d, i]) for d in deps for i in indeps if d != i})): st.stop()
            else:
                done = 0
                seen = set()
                with st.spinner("Đang cày xới toàn bộ ma trận số liệu..."):
                    for dep in deps:
                        for indep in indeps:
                            if dep == indep or frozenset([dep, indep]) in seen: continue
                            seen.add(frozenset([dep, indep]))
                            try:
                                r = crosstab_test(df, indep, dep)
                                pv = r.get("p_value")
                                done += 1
                                st.markdown(f"**► Mối liên quan giữa: [{indep}] & [{dep}] — {'🟢 CÓ Ý NGHĨA' if pv is not None and pv<0.05 else '⚪ KHÔNG Ý NGHĨA'}**")
                                st.dataframe(r["table"], use_container_width=True)
                                st.write(f"- **Kiểm định:** {r.get('test','')} | **p-value:** `{pv:.4g}`" if pv is not None else f"- **Kiểm định:** {r.get('test','')}")
                                if "effect_size" in r: st.write(f"- **Chỉ số (Effect Size / OR):** `{r['effect_size']}`")
                                rid = f"CROSS_{indep}_{dep}"
                                st.session_state["saved_tables"][rid] = r["table"]
                                upsert_candidate(rid, f"Mối liên quan giữa {indep} và {dep}", "association", [indep, dep], 4.5, 4.5, 5.0, pv)
                            except Exception as exc: 
                                st.error(f"⚠️ Lỗi phân tích chéo [{indep} & {dep}]: {exc}")
                st.success(f"✅ Đã phân tích và nạp {done} bảng kiểm định vào Giỏ!")
                gc.collect()

        st.write("---")
        st.subheader("4. So sánh biến định lượng giữa 2 nhóm (T-test / Mann-Whitney)")
        a, b = st.columns(2)
        with a:
            sag = st.checkbox("✅ Chọn tất cả biến nhóm", key=ui_key("chk_all_groups"))
            gkey = ui_key("group_vars_all") if sag else ui_key("group_vars_manual")
            groups = st.multiselect("Biến nhóm (Tự lọc biến 2 mức)", all_cols, default=all_cols if sag else [], key=gkey)
        with b:
            sav = st.checkbox("✅ Chọn tất cả biến định lượng", key=ui_key("chk_all_vals"))
            vkey = ui_key("val_vars_all") if sav else ui_key("val_vars_manual")
            vals = st.multiselect("Biến định lượng cần so sánh", numeric_candidates or all_cols, default=(numeric_candidates or all_cols) if sav else [], key=vkey)
            
        if st.button("Quét kiểm định so sánh (Nạp TẤT CẢ vào Giỏ)", key=ui_key("run_group_compare")):
            if not groups or not vals: st.warning("Vui lòng chọn ít nhất 1 biến ở mỗi mục.")
            elif not large_batch_ok("So sánh 2 nhóm", sum(g != v for g in groups for v in vals)): st.stop()
            else:
                done = 0
                with st.spinner("Đang rà soát và tính toán..."):
                    for gv in groups:
                        for vv in vals:
                            if gv == vv: continue
                            try:
                                r = compare_two_groups(df, gv, vv)
                                pv = r.get("p_value")
                                done += 1
                                g1, g2 = r["group_names"]
                                st.markdown(f"**► Sự phân bố của [{vv}] giữa 2 nhóm [{gv}] — {'🟢 KHÁC BIỆT' if pv is not None and pv<0.05 else '⚪ TƯƠNG ĐỒNG'}**")
                                st.write(f"- **Kiểm định:** {r.get('test','')} | **p-value:** `{pv:.4g}`" if pv is not None else f"- **Kiểm định:** {r.get('test','')}")
                                if "effect_size" in r: st.write(f"- **Chỉ số (Effect Size):** `{r['effect_size']}`")
                                comp = pd.DataFrame({g1: [r["group1_stats"]], g2: [r["group2_stats"]]}, index=["Giá trị"])
                                st.dataframe(comp, use_container_width=True)
                                rid = f"COMP_{gv}_{vv}"
                                st.session_state["saved_tables"][rid] = comp
                                upsert_candidate(rid, f"Sự khác biệt của biến {vv} giữa các nhóm {gv}", "association", [gv, vv], 4.5, 4.5, 5.0, pv)
                            except Exception as exc: 
                                st.error(f"⚠️ Lỗi so sánh [{gv} & {vv}]: {exc}")
                st.success(f"✅ Đã nạp {done} kết quả so sánh vào Giỏ!")
                gc.collect()

        st.write("---")
        st.subheader("5. Hồi quy logistic nhị phân (OR và 95% CI)")
        outcomes_all = [c for c in all_cols if df[c].dropna().nunique() == 2]
        forbidden = ["unnamed", "ngay", "ngày", "ten", "tên", "ma", "mã", "sobenhan", "id"]
        predictors_all = [c for c in all_cols if not any(k in str(c).lower() for k in forbidden) and df[c].dropna().nunique() > 1]
        
        a, b = st.columns([1, 2])
        with a:
            sao = st.checkbox("✅ Chọn tất cả biến kết cục", key=ui_key("chk_all_outcomes"))
            okey = ui_key("log_outcomes_all") if sao else ui_key("log_outcomes_manual")
            outcomes = st.multiselect("Biến kết cục (Nhị phân)", outcomes_all, default=outcomes_all if sao else [], key=okey)
        with b:
            sap = st.checkbox("✅ Chọn tất cả yếu tố dự báo", key=ui_key("chk_all_predictors"))
            pkey = ui_key("log_predictors_all") if sap else ui_key("log_predictors_manual")
            predictors = st.multiselect("Yếu tố dự báo", predictors_all, default=predictors_all if sap else [], key=pkey)
            
        if st.button("Chạy Logistic Regression đa biến (Nạp vào Giỏ)", key=ui_key("run_logistic")):
            if not outcomes or not predictors: st.warning("Chọn ít nhất một biến ở mỗi bên.")
            elif not large_batch_ok("Logistic Regression", len(outcomes)): st.stop()
            else:
                done = 0
                with st.spinner("Đang xây dựng mô hình hồi quy..."):
                    for out in outcomes:
                        preds = [p for p in predictors if p != out]
                        if not preds: continue
                        try:
                            r, summary = binary_logistic_regression(df, out, preds)
                            done += 1
                            st.markdown(f"**► MÔ HÌNH HỒI QUY ĐA BIẾN CHO KẾT CỤC: [{out}]**")
                            st.info(summary)
                            st.dataframe(r, use_container_width=True)
                            rid = f"LOG_{out}"
                            st.session_state["saved_tables"][rid] = r
                            upsert_candidate(rid, f"Mô hình hồi quy logistic đánh giá yếu tố liên quan đến {out}", "regression", [out] + preds, 5.0, 5.0, 5.0)
                        except Exception as exc: 
                            st.error(f"⚠️ Không thể xây dựng mô hình cho [{out}]: {exc}")
                st.success(f"✅ Đã nạp {done} mô hình hồi quy vào Giỏ!")
                gc.collect()

        st.write("---")
        st.subheader("8. Diễn giải kết quả bằng AI (Nhận xét bảng chuẩn khoa học)")
        st.info("💡 Anh có thể chọn một bảng riêng lẻ hoặc chọn **'🌟 Chọn TẤT CẢ các bảng trong Giỏ'** để AI tổng hợp nhận xét toàn bộ số liệu.")
        saved = st.session_state.get("saved_tables", {})
        options = ["-- Chỉ dùng số liệu dán tay bên dưới --", "🌟 Chọn TẤT CẢ các bảng trong Giỏ"] + list(saved.keys())
        choice = st.selectbox("Lựa chọn bảng hoặc nguồn dữ liệu:", options, key=ui_key("select_ai_table"))
        extra = st.text_area("Số liệu bổ sung hoặc yêu cầu cụ thể (nếu có):", height=100, key=ui_key("interpretation_request"))
        
        if st.button("🤖 AI Viết Nhận Xét Bảng", type="primary", key=ui_key("ai_interpret")):
            final = ""
            if choice == "🌟 Chọn TẤT CẢ các bảng trong Giỏ":
                if not saved: st.warning("⚠️ Giỏ kết quả đang trống, chưa có bảng nào được lưu!")
                for k, v in saved.items(): final += f"### BẢNG: {k}\n" + v.to_markdown(index=False) + "\n\n"
            elif choice != "-- Chỉ dùng số liệu dán tay bên dưới --": 
                final += f"### BẢNG: {choice}\n" + saved[choice].to_markdown(index=False) + "\n\n"
            if extra.strip(): final += f"SỐ LIỆU / YÊU CẦU BỔ SUNG:\n{extra.strip()}"
            if not final.strip(): st.warning("⚠️ Anh chưa chọn bảng nào hoặc chưa dán số liệu!")
            else:
                prompt = f"""{BASE_SYSTEM_RULES}
Nhiệm vụ của bạn là viết phần **'Nhận xét'** cho các bảng số liệu thống kê trong luận văn Dược lâm sàng.
QUY TẮC VÀNG BẮT BUỘC:
1. CHỈ ĐƯA RA SỐ LIỆU: Chỉ diễn giải các số liệu, tần số, tỷ lệ % nổi bật có trong bảng.
2. VĂN PHONG KHOA HỌC: Câu văn logic, ngắn gọn, dễ hiểu, mạch lạc.
3. TUYỆT ĐỐI KHÔNG BÀN LUẬN: Không giải thích nguyên nhân, không suy diễn cơ chế lâm sàng, không so sánh với các nghiên cứu khác.
4. Trình bày thành các đoạn văn xuôi y khoa liền mạch, chuẩn mực.
DỮ LIỆU ĐẦU VÀO:\n{final}"""
                try:
                    with st.spinner("AI đang phân tích số liệu và soạn nhận xét chuyên sâu..."): 
                        out = call_gemini(prompt, model=DEFAULT_MODEL)
                    if out: 
                        st.markdown("### 📝 Kết quả Nhận xét Bảng:")
                        st.markdown(out)
                except Exception as exc: 
                    st.error(f"Lỗi gọi AI: {exc}")
