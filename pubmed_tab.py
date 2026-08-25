# pubmed_tab.py
import streamlit as st

def render_pubmed_tab(
    ui_key,
    render_evidence_database_status,
    translate_query_to_mesh,
    search_pubmed,
    normalize_pubmed_query,
    ingest_pubmed_article,
    rebuild_index
):
    st.header("🔍 PubMed")
    st.info("Nhập tên đề tài bằng tiếng Việt. Hệ thống tự dịch sang từ khoá MeSH để tra cứu tài liệu y khoa chuẩn quốc tế trên PubMed.")
    render_evidence_database_status()

    cs, cb = st.columns([4, 1])
    with cs: 
        t3_query = st.text_input("Tên đề tài nghiên cứu (tiếng Việt):", key=ui_key("t3_query_input"))
    with cb: 
        # ĐÃ ÉP CỨNG: Tối thiểu 20 bài, mặc định 20 bài (có thể kéo lên tối đa 100 bài)
        max_res = st.number_input("Số bài/nguồn", min_value=20, max_value=100, value=20, key=ui_key("t3_max_res"))

    if st.button("🚀 Tìm kiếm trên PubMed", type="primary", key=ui_key("t3_btn_search")):
        if not t3_query.strip(): 
            st.warning("Vui lòng nhập tên đề tài nghiên cứu!")
        else:
            st.session_state["t3_query"] = t3_query
            with st.spinner("Đang dịch & chuẩn hoá từ khoá sang MeSH (PubMed)..."):
                en_query = translate_query_to_mesh(t3_query)
                st.session_state["t3_en_keyword"] = en_query

            with st.spinner(f"Đang tìm & tải {max_res} Abstract từ PubMed..."):
                st.session_state["t3_pm_data"] = search_pubmed(normalize_pubmed_query(en_query, t3_query), max_res)

    # HIỂN THỊ KẾT QUẢ PUBMED
    if st.session_state.get("t3_pm_data"):
        st.write("---")
        st.markdown("### 🌍 Kết quả tra cứu PubMed")
        if st.session_state.get("t3_en_keyword"): 
            st.success(f"🔑 Từ khoá MeSH: **{st.session_state['t3_en_keyword']}**")

        for i, art in enumerate(st.session_state.get("t3_pm_data", [])):
            with st.container(border=True):
                st.markdown(f"**[{art.get('title', '')}]({art.get('url', '#')})**")
                st.caption(f"✍️ {art.get('authors', '')} ({art.get('year', '')}) — {art.get('journal', '')}")
                with st.expander("Xem tóm tắt (Abstract)"): 
                    st.write(art.get("abstract", ""))

                if st.button("➕ Nạp vào Evidence Database", key=ui_key(f"pm_ingest_{i}")):
                    try:
                        if ingest_pubmed_article(art): 
                            rebuild_index()
                            st.success("Đã nạp vào Evidence Database.")
                            st.rerun()
                        else: 
                            st.info("Nguồn này đã có trong Evidence Database.")
                    except Exception as exc: 
                        st.error(f"❌ Lỗi nạp nguồn: {exc}")

        st.write("---")
        if st.button("➕ Nạp TẤT CẢ kết quả ở trên vào Evidence Database", key=ui_key("t3_ingest_all")):
            count = 0
            with st.spinner("Đang nạp tất cả nguồn và cập nhật index..."):
                for art in st.session_state.get("t3_pm_data", []):
                    try: 
                        count += 1 if ingest_pubmed_article(art) else 0
                    except Exception: 
                        pass
                if count: 
                    rebuild_index()
            st.success(f"Đã nạp {count} nguồn mới vào Evidence Database.")
