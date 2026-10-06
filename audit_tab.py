# audit_tab.py
import streamlit as st
import re
import pandas as pd

def render_audit_tab(
    ui_key,
    Audit_generated_text_wrapper,
    internal_overlap_Audit_wrapper,
    call_gemini,
    BASE_SYSTEM_RULES,
    MODEL_LITE
):
    st.header("🔎 Audit luận văn toàn diện")
    st.markdown('<div class="warning-box">⚠️ <b>Giới hạn cần biết:</b> Công cụ chỉ báo nguy cơ.</div>', unsafe_allow_html=True)
    
    text = st.text_area("Dán đoạn văn cần Audit vào đây:", height=250, key=ui_key("Audit_text"))
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    st.write("---")
    box = st.container()
    
    with c1:
        if st.button("🔢 Số liệu", use_container_width=True, key=ui_key("Audit_numbers")):
            if not text.strip(): 
                st.warning("Chưa có văn bản.")
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
            if not text.strip(): 
                st.warning("Chưa có văn bản.")
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
        if st.button("🔍 Trùng lặp", use_container_width=True, key=ui_key("Audit_overlap")):
            if not text.strip(): 
                st.warning("Chưa có văn bản.")
            else:
                try: 
                    ov = internal_overlap_Audit_wrapper(text)
                except Exception as exc: 
                    ov = []
                    st.error(f"❌ Không thể quét trùng lặp: {exc}")
                
                with box:
                    st.markdown("### 🔍 Báo cáo Trùng lặp (Plagiarism & Overlap)")
                    if not ov: 
                        st.success("✅ Tuyệt vời! Không tìm thấy đoạn văn trùng lặp đáng kể nào trong tài liệu nội bộ.")
                    else:
                        st.warning(f"⚠️ Phát hiện {len(ov)} đoạn có dấu hiệu trùng lặp cao.")
                        
                        df_ov = pd.DataFrame(ov)
                        
                        if not df_ov.empty:
                            df_ov["% Trùng lặp"] = (df_ov["similarity"] * 100).round(1).astype(str) + "%"
                            df_ov["Tài liệu gốc"] = df_ov["file"] + " (Trang " + df_ov["page"].astype(str) + ")"
                            
                            display_df = df_ov[["% Trùng lặp", "Tài liệu gốc", "text"]]
                            display_df.rename(columns={"text": "Nội dung trùng khớp"}, inplace=True)
                            
                            st.dataframe(
                                display_df, 
                                use_container_width=True,
                                hide_index=True
                            )
                        
    with c4:
        if st.button("🔤 Chính tả", use_container_width=True, key=ui_key("Audit_spelling")):
            if not text.strip(): 
                st.warning("Chưa có văn bản.")
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
            if not text.strip(): 
                st.warning("Chưa có văn bản.")
            else:
                AI_DETECT_PROMPT = f"""{BASE_SYSTEM_RULES}
Bạn là một hệ thống phân tích ngôn ngữ học chuyên nghiệp (AI Text Detector) tương tự như Originality.ai hay ZeroGPT.
Hãy phân tích đoạn văn sau và đánh giá xác suất nó được viết bởi AI (như ChatGPT, Claude, Gemini) hay con người.

Tiêu chí phân tích (BẮT BUỘC):
1. Perplexity (Độ lúng túng): Tính dễ đoán của từ vựng. AI thường dùng từ rất phổ biến, dễ đoán, khuôn sáo (Perplexity thấp). Con người dùng từ vựng đa dạng, có thể có từ lóng, thuật ngữ chuyên ngành hẹp hoặc cách dùng từ độc đáo (Perplexity cao).
2. Burstiness (Độ bùng nổ): Sự đa dạng về cấu trúc và độ dài câu. Con người viết câu lúc ngắn lúc dài, lúc phức tạp lúc đơn giản, nhịp điệu không đều (Burstiness cao). AI thường viết các câu có độ dài và cấu trúc rất đều đặn, câu văn xuôi mượt mà nhưng rập khuôn (Burstiness thấp).

Hãy trả về kết quả theo cấu trúc Markdown sau:
### 📊 Tỷ lệ rủi ro AI: [Điền % từ 0-100%]
* (>70%: Chắc chắn AI, 30-70%: Có thể chỉnh sửa từ AI, <30%: Khả năng cao là người viết)*

**1. Phân tích Perplexity (Lựa chọn từ vựng):**
[Đánh giá chi tiết của bạn về cách dùng từ...]

**2. Phân tích Burstiness (Cấu trúc & Độ dài câu):**
[Đánh giá chi tiết của bạn về cấu trúc câu...]

**3. Bằng chứng cụ thể:**
[Trích dẫn 1-2 câu trong bài có văn phong đậm chất "máy móc" nhất nếu có, và giải thích tại sao]

ĐOẠN VĂN CẦN KIỂM TRA:
{text}
"""
                try: 
                    # Sử dụng mô hình mặc định (Pro/Flash) mạnh hơn thay vì LITE để phân tích ngữ nghĩa chính xác
                    res = call_gemini(AI_DETECT_PROMPT) 
                except Exception as exc: 
                    res = f"Lỗi gọi Gemini: {exc}"
                with box: 
                    st.markdown("### 🤖 Báo cáo phân tích văn phong AI")
                    st.info("💡 **Lưu ý:** AI Detection dựa trên thuật toán ngữ nghĩa, có thể xảy ra dương tính giả (false positive) nếu văn bản học thuật được viết với cấu trúc quá khuôn mẫu.")
                    st.markdown(str(res))
                    
    with c6:
        if st.button("⚖️ Phản biện", use_container_width=True, key=ui_key("logic_review")):
            if not text.strip(): 
                st.warning("Chưa có văn bản.")
            else:
                p = f"{BASE_SYSTEM_RULES}\nĐóng vai phản biện luận văn CKI Dược lâm sàng. Chỉ ra điểm yếu logic: thiếu bằng chứng, tương quan/nhân quả, vượt giới hạn thiết kế nghiên cứu. ĐOẠN VĂN: {text}"
                try: 
                    res = call_gemini(p)
                except Exception as exc: 
                    res = f"Lỗi gọi Gemini: {exc}"
                with box: 
                    st.markdown("### ⚖️ Kết quả Phản biện\n" + str(res))
