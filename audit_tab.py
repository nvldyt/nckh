# audit_tab.py (Cập nhật Error Handling cho đầu vào)
import streamlit as st
import re
import pandas as pd

def render_audit_tab(
    ui_key,
    Audit_generated_text_wrapper,
    internal_overlap_Audit_wrapper,
    check_internet_plagiarism, 
    call_gemini,
    BASE_SYSTEM_RULES,
    MODEL_LITE
):
    st.header("🔎 Audit luận văn toàn diện")
    st.markdown('<div class="warning-box">⚠️ <b>Giới hạn cần biết:</b> Công cụ chỉ báo nguy cơ. Quét Internet có thể mất vài giây.</div>', unsafe_allow_html=True)
    
    # 1. Nhận văn bản đầu vào
    text = st.text_area("Dán đoạn văn cần Audit vào đây:", height=250, key=ui_key("Audit_text"))
    
    # 2. XỬ LÝ LỖI (ERROR HANDLING) VÀ KIỂM TRA CHẤT LƯỢNG VĂN BẢN ĐẦU VÀO
    # Khởi tạo cờ kiểm tra hợp lệ
    is_valid_input = False
    
    if not text.strip():
        st.info("ℹ️ Vui lòng dán văn bản luận văn/báo cáo vào ô trống phía trên để bắt đầu phân tích.")
    else:
        # Kiểm tra độ dài tối thiểu (ví dụ: cần ít nhất 30 từ để phân tích có ý nghĩa)
        word_count = len(text.strip().split())
        # Đếm số lượng câu (dựa vào dấu chấm, hỏi, than)
        sentence_count = len(re.split(r'[.!?]+', text.strip())) - 1 

        if word_count < 30:
            st.warning("⚠️ VĂN BẢN QUÁ NGẮN: Vui lòng nhập đoạn văn dài hơn (ít nhất 30 từ) để thuật toán có đủ dữ liệu nhận diện văn phong AI và quét đạo văn.")
        elif sentence_count < 2:
            st.warning("⚠️ THIẾU CẤU TRÚC CÂU: Đoạn văn có vẻ giống một danh sách liệt kê hơn là văn xuôi (không có đủ dấu chấm câu). Hãy chuyển dữ liệu thành văn bản mô tả (Prose) để AI có thể phân tích Burstiness và Perplexity chính xác.")
        else:
            is_valid_input = True # Đã đạt chuẩn

    # Bố cục nút bấm
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    st.write("---")
    box = st.container()
    
    with c1:
        if st.button("🔢 Số liệu", use_container_width=True, key=ui_key("Audit_numbers")):
            if not is_valid_input:
                st.error("Vui lòng nhập văn bản có cấu trúc câu đầy đủ và đủ dài để phân tích.")
            else:
                try: 
                    r = Audit_generated_text_wrapper(text)
                except Exception as exc: 
                    r = {"exact_matches": [], "derived_matches": [], "warnings": [], "evidence_used": []}
                    st.error(f"❌ Không thể Audit số liệu: {exc}")
                
                with box:
                    st.markdown("### 🔢 Kết quả Audit Số liệu")
                    st.success("**Level 1 (Khớp chính xác):** " + (", ".join(r.get("exact_matches", [])) or "Không có"))
                    st.info("**Level 2 (Khớp phái sinh):** " + (", ".join(r.get("derived_matches", [])) or "Không có"))
                    
                    if r.get("warnings"): 
                        st.error("**Level 3 (⚠️ SỐ LIỆU LẠ):** " + ", ".join(r["warnings"]))
                    else: 
                        st.success("**Level 3:** Không phát hiện số liệu lạ!")
                    
                    with st.expander("📄 Xem bằng chứng đối chiếu"):
                        for e in r.get("evidence_used", []): 
                            st.write(f"> {e.get('text', '')}")
                            
    with c2:
        if st.button("📚 Trích dẫn", use_container_width=True, key=ui_key("Audit_citation")):
            if not is_valid_input:
                st.error("Vui lòng nhập văn bản có cấu trúc câu đầy đủ và đủ dài để phân tích.")
            else:
                cites = re.findall(r"\[(\d+)\]", text)
                refs = {str(x.get("vancouver_index")): x for x in st.session_state.get("current_references", [])}
                with box:
                    st.markdown("### 📚 Kết quả Audit Citation")
                    fake = [x for x in cites if x not in refs]
                    if fake: 
                        st.error("❌ Phát hiện trích dẫn ẢO: " + ", ".join(f"[{x}]" for x in fake))
                    elif cites: 
                        st.success("✅ Toàn bộ trích dẫn khớp!")
                    else: 
                        st.info("Không tìm thấy trích dẫn [n].")
                        
    with c3:
        btn_overlap_internal = st.button("🔍 Nội bộ", use_container_width=True, key=ui_key("Audit_overlap"), help="Quét trùng lặp với các tài liệu đã tải lên")
        btn_overlap_internet = st.button("🌐 Internet", use_container_width=True, key=ui_key("Audit_internet"), help="Dò tìm đạo văn trên Google/DuckDuckGo")
        
        if btn_overlap_internal:
            if not is_valid_input:
                st.error("Vui lòng nhập văn bản có cấu trúc câu đầy đủ và đủ dài để phân tích.")
            else:
                try: 
                    ov = internal_overlap_Audit_wrapper(text)
                except Exception as exc: 
                    ov = []
                    st.error(f"❌ Không thể quét trùng lặp: {exc}")
                
                with box:
                    st.markdown("### 🔍 Báo cáo Trùng lặp Nội bộ (Internal Overlap)")
                    if not ov: 
                        st.success("✅ Không tìm thấy đoạn văn trùng lặp đáng kể nào trong tài liệu nội bộ.")
                    else:
                        st.warning(f"⚠️ Phát hiện {len(ov)} đoạn có dấu hiệu trùng lặp cao.")
                        df_ov = pd.DataFrame(ov)
                        if not df_ov.empty:
                            df_ov["% Trùng lặp"] = (df_ov["similarity"] * 100).round(1).astype(str) + "%"
                            df_ov["Tài liệu gốc"] = df_ov["file"] + " (Trang " + df_ov["page"].astype(str) + ")"
                            display_df = df_ov[["% Trùng lặp", "Tài liệu gốc", "text"]]
                            display_df.rename(columns={"text": "Nội dung trùng khớp"}, inplace=True)
                            st.dataframe(display_df, use_container_width=True, hide_index=True)

        if btn_overlap_internet:
            if not is_valid_input:
                st.error("Vui lòng nhập văn bản có cấu trúc câu đầy đủ và đủ dài để phân tích.")
            else:
                with st.spinner("🌐 Đang kết nối Internet và đối chiếu dữ liệu (có thể mất 5-10 giây)..."):
                    try: 
                        ext_results = check_internet_plagiarism(text)
                    except Exception as exc: 
                        ext_results = []
                        st.error(f"❌ Lỗi kết nối Internet: {exc}")
                
                with box:
                    st.markdown("### 🌐 Báo cáo Đạo văn Internet (External Plagiarism)")
                    if not ext_results: 
                        st.success("✅ Tuyệt vời! Thuật toán dò tìm không phát hiện câu văn này bị sao chép trực tiếp từ Internet.")
                    else:
                        st.error(f"🚨 CẢNH BÁO ĐẠO VĂN: Tìm thấy {len(ext_results)} nguồn trên mạng chứa nguyên văn câu chữ này!")
                        
                        df_ext = pd.DataFrame(ext_results)
                        if not df_ext.empty:
                            st.dataframe(
                                df_ext, 
                                column_config={
                                    "Nguồn (URL)": st.column_config.LinkColumn("Link Website (Click để xem)"),
                                },
                                use_container_width=True,
                                hide_index=True
                            )
                        
    with c4:
        if st.button("🔤 Chính tả", use_container_width=True, key=ui_key("Audit_spelling")):
            if not is_valid_input:
                 st.error("Vui lòng nhập văn bản có cấu trúc câu đầy đủ và đủ dài để phân tích.")
            else:
                p = f"{BASE_SYSTEM_RULES}\nRà soát đoạn văn bản sau để tìm lỗi chính tả/thuật ngữ. ĐOẠN VĂN: {text}"
                try: 
                    res = call_gemini(p, model=MODEL_LITE)
                except Exception as exc: 
                    res = f"Lỗi gọi Gemini: {exc}"
                with box: 
                    st.markdown("### 🔤 Chính tả & Thuật ngữ\n" + str(res))
                    
    with c5:
        if st.button("🤖 Check văn AI", use_container_width=True, key=ui_key("Audit_ai_style")):
            if not is_valid_input:
                 st.error("Vui lòng nhập văn bản có cấu trúc câu đầy đủ và đủ dài để phân tích.")
            else:
                AI_DETECT_PROMPT = f"""{BASE_SYSTEM_RULES}
Bạn là một hệ thống phân tích ngôn ngữ học chuyên nghiệp (AI Text Detector).
Hãy phân tích đoạn văn sau và đánh giá xác suất nó được viết bởi AI hay con người.

Tiêu chí phân tích (BẮT BUỘC):
1. Perplexity (Độ lúng túng): Tính dễ đoán của từ vựng.
2. Burstiness (Độ bùng nổ): Sự đa dạng về cấu trúc và độ dài câu. 

Hãy trả về kết quả theo cấu trúc Markdown sau:
### 📊 Tỷ lệ rủi ro AI: [Điền % từ 0-100%]

**1. Phân tích Perplexity:** [Đánh giá...]
**2. Phân tích Burstiness:** [Đánh giá...]
**3. Bằng chứng:** [Trích dẫn câu có vẻ máy móc nhất]

ĐOẠN VĂN:
{text}
"""
                try: 
                    res = call_gemini(AI_DETECT_PROMPT) 
                except Exception as exc: 
                    res = f"Lỗi gọi Gemini: {exc}"
                with box: 
                    st.markdown("### 🤖 Báo cáo phân tích văn phong AI")
                    st.markdown(str(res))
                    
    with c6:
        if st.button("⚖️ Phản biện", use_container_width=True, key=ui_key("logic_review")):
            if not is_valid_input:
                 st.error("Vui lòng nhập văn bản có cấu trúc câu đầy đủ và đủ dài để phân tích. Bạn cũng nên bổ sung quan điểm cá nhân để thuật toán có cơ sở phản biện lại.")
            else:
                p = f"{BASE_SYSTEM_RULES}\nĐóng vai phản biện luận văn CKI Dược lâm sàng. Chỉ ra điểm yếu logic: thiếu bằng chứng, tương quan/nhân quả, vượt giới hạn thiết kế nghiên cứu. ĐOẠN VĂN: {text}"
                try: 
                    res = call_gemini(p)
                except Exception as exc: 
                    res = f"Lỗi gọi Gemini: {exc}"
                with box: 
                    st.markdown("### ⚖️ Kết quả Phản biện\n" + str(res))
