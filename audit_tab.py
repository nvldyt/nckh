# audit_tab.py
import streamlit as st
import re

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
                    st.markdown("### 🔍 Trùng lặp nội bộ")
                    if not ov: 
                        st.info("Không tìm thấy đoạn trùng đáng kể.")
                    for x in ov: 
                        st.markdown(f"**{x.get('file', '')} – trang {x.get('page', '')}**\nSimilarity: **{x.get('similarity', 0)}**\n> {x.get('text', '')}")
                        
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
                p = f"{BASE_SYSTEM_RULES}\nSoi khắt khe các dấu hiệu văn bản do AI viết. ĐOẠN VĂN: {text}"
                try: 
                    res = call_gemini(p, model=MODEL_LITE)
                except Exception as exc: 
                    res = f"Lỗi gọi Gemini: {exc}"
                with box: 
                    st.markdown("### 🤖 Chỉ báo nguy cơ AI viết\n" + str(res))
                    
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
