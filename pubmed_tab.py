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
    st.info("Hệ thống sẽ dịch đề tài sang từ khoá MeSH. Bạn có thể kiểm tra và chỉnh sửa lại từ khóa trước khi chính thức tra cứu PubMed.")
    render_evidence_database_status()

    # BƯỚC 1: NHẬP ĐỀ TÀI & DỊCH MESH
    t3_query = st.text_input("Tên đề tài nghiên cứu (tiếng Việt):", key=ui_key("t3_query_input"))
    
    if st.button("🧠 1. Dịch sang từ khóa MeSH", key=ui_key("t3_btn_translate")):
        if not t3_query.strip(): 
            st.warning("Vui lòng nhập tên đề tài nghiên cứu!")
        else:
            st.session_state["t3_query"] = t3_query
            with st.spinner("Đang phân tích ngữ nghĩa y khoa và dịch sang MeSH..."):
                en_query = translate_query_to_mesh(t3_query)
                st.session_state["t3_en_keyword"] = en_query

    # BƯỚC 2: CHỈNH SỬA TỪ KHÓA & TÌM KIẾM
    if "t3_en_keyword" in st.session_state:
        st.markdown("### 🔑 Từ khóa tra cứu PubMed")
        # Cho phép người dùng tự tay tinh chỉnh từ khóa MeSH nếu AI dịch chưa sát
        editable_mesh = st.text_input(
            "Bạn có thể thêm/bớt từ khóa ở ô dưới đây trước khi tìm kiếm:", 
            value=st.session_state["t3_en_keyword"], 
            key=ui_key("t3_editable_mesh")
        )

        cs, cb = st.columns([4, 1])
        with cb: 
            max_res = st.number_input("Số bài/nguồn", min_value=10, max_value=100, value=20, key=ui_key("t3_max_res"))
        with cs:
            st.write("") # Căn lề cho nút bấm ngang hàng với ô nhập số
            st.write("")
            search_clicked = st.button("🚀 2. Tìm kiếm trên PubMed", type="primary", key=ui_key("t3_btn_search"))

        if search_clicked:
            st.session_state["t3_en_keyword"] = editable_mesh # Cập nhật lại từ khóa nếu có chỉnh sửa
            with st.spinner(f"Đang tìm & tải {max_res} Abstract từ PubMed..."):
                final_query = normalize_pubmed_query(editable_mesh, t3_query)
                st.session_state["t3_pm_data"] = search_pubmed(final_query, max_res)

    # BƯỚC 3: HIỂN THỊ KẾT QUẢ & NẠP DỮ LIỆU (UX TỐI ƯU)
    if st.session_state.get("t3_pm_data"):
        st.write("---")
        st.markdown("### 🌍 Kết quả tra cứu PubMed")
        
        # ĐƯA NÚT NẠP TẤT CẢ LÊN ĐẦU ĐỂ KHÔNG PHẢI CUỘN CHUỘT
        if st.button("➕ Nạp TẤT CẢ kết quả bên dưới vào Evidence Database", type="primary", key=ui_key("t3_ingest_all_top")):
            count = 0
            failed_count = 0
            with st.spinner("Đang nạp tất cả nguồn và cập nhật index (vector database)..."):
                for art in st.session_state.get("t3_pm_data", []):
                    try: 
                        if ingest_pubmed_article(art):
                            count += 1
                    except Exception as e: 
                        failed_count += 1
                
                if count > 0: 
                    rebuild_index()
            
            st.success(f"✅ Đã nạp thành công {count} nguồn mới vào Evidence Database.")
            if failed_count > 0:
                st.warning(f"⚠️ Có {failed_count} bài bị lỗi nạp hoặc trùng lặp. Vui lòng kiểm tra lại.")

        st.write("---")

        for i, art in enumerate(st.session_state.get("t3_pm_data", [])):
            with st.container(border=True):
                st.markdown(f"**[{art.get('title', '')}]({art.get('url', '#')})**")
                st.caption(f"✍️ {art.get('authors', '')} ({art.get('year', '')}) — {art.get('journal', '')}")
                with st.expander("Xem tóm tắt (Abstract)"): 
                    st.write(art.get("abstract", ""))

                # SỬ DỤNG PMID LÀM KEY THAY VÌ INDEX ĐỂ TRÁNH LỖI DUPLICATE KEY CỦA STREAMLIT
                pmid = art.get('pmid', str(i)) 
                
                if st.button("➕ Nạp vào Evidence Database", key=ui_key(f"pm_ingest_{pmid}")):
                    try:
                        if ingest_pubmed_article(art): 
                            rebuild_index()
                            st.success("Đã nạp vào Evidence Database.")
                            st.rerun()
                        else: 
                            st.info("Nguồn này đã có trong Evidence Database.")
                    except Exception as exc: 
                        st.error(f"❌ Lỗi nạp nguồn: {exc}")
