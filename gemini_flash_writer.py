import os
import time
import random
import streamlit as st
from google import genai

def get_gemini_key() -> str:
    """Tự động xoay vòng lấy API Key từ danh sách trong st.secrets hoặc biến môi trường."""
    try:
        if "GEMINI" in st.secrets and "API_KEYS" in st.secrets["GEMINI"]:
            keys = st.secrets["GEMINI"]["API_KEYS"]
            if keys and isinstance(keys, list):
                return random.choice(keys).strip()
            elif isinstance(keys, str):
                # Hỗ trợ cả trường hợp user khai báo nhầm thành chuỗi (ngăn cách bởi dấu phẩy)
                return random.choice([k.strip() for k in keys.split(",") if k.strip()])
    except Exception:
        pass
        
    return os.getenv("GEMINI_API_KEY", "")

def render_gemini_flash_tab():
    st.markdown("### 🚀 Trợ lý Viết Luận văn (Gemini 3.8 Flash)")
    st.caption("Ứng dụng mô hình Gemini 3.8 Flash với bộ nhớ siêu dài (1M Token) và công nghệ phản hồi tức thì (Streaming).")

    # ==========================================
    # 1. QUẢN LÝ KHÓA API
    # ==========================================
    active_key = get_gemini_key()
    
    if not active_key:
        active_key = st.text_input(
            "🔑 Nhập Google Gemini API Key:", 
            type="password", 
            key="input_gemini_key_tab6",
            help="Hệ thống không tìm thấy khóa mặc định trong secrets. Vui lòng nhập thủ công."
        )

    if "gemini_writer_messages" not in st.session_state:
        st.session_state["gemini_writer_messages"] = []

    # ==========================================
    # 2. NẠP TOÀN BỘ DỮ LIỆU BỐI CẢNH (RAG LONG-CONTEXT)
    # ==========================================
    with st.expander("🔍 Dữ liệu bối cảnh và danh mục tham khảo đang nạp", expanded=False):
        context_blocks = []
        docs = st.session_state.get("documents", {})
        chunks = st.session_state.get("chunks", [])
        ref_counter = 1
        
        if docs and chunks:
            doc_list_text = []
            doc_mapping = {}
            
            # 2.1. Xây dựng Danh mục
            for sid, meta in docs.items():
                doc_mapping[sid] = ref_counter
                title = meta.get("title") or meta.get("file_name") or sid
                authors = meta.get("authors", "Không rõ tác giả")
                year = meta.get("year", "")
                doc_list_text.append(f"[{ref_counter}] {authors}. {title}. {year}")
                ref_counter += 1
                
            context_blocks.append("DANH MỤC TÀI LIỆU GỐC:\n" + "\n".join(doc_list_text))
            
            # 2.2. Nhồi Chunks
            ev_lines = []
            for c in chunks:
                sid = c.get("source_id")
                doc_idx = doc_mapping.get(sid, "?")
                text = c.get("text", "")
                if text.strip():
                    ev_lines.append(f"[Trích đoạn từ tài liệu {doc_idx}]:\n{text}")
            context_blocks.append("NỘI DUNG CHI TIẾT TỪ CÁC TÀI LIỆU:\n" + "\n---\n".join(ev_lines))
            
        # 2.3. Nạp Text từ Word (Nếu có từ Tab 7/Tab khác chuyển sang)
        analyzed_data = st.session_state.get("analyzed_data", {"dataframes": {}, "word_texts": []})
        if analyzed_data.get("word_texts"):
            context_blocks.extend(analyzed_data["word_texts"])
            
        # 2.4. Nạp Tóm tắt & Bảng SPSS
        summary = st.session_state.get("cached_summary", "")
        if summary:
            context_blocks.append(f"TÓM TẮT ĐỀ TÀI (Nguồn nội bộ [{ref_counter}]):\n{summary}")
            ref_counter += 1

        saved_tables = st.session_state.get("saved_tables", {})
        if saved_tables:
            table_info = "".join([f"Bảng {name}:\n{df.to_markdown()}\n\n" for name, df in saved_tables.items()])
            context_blocks.append(f"BẢNG SỐ LIỆU NGHIÊN CỨU:\n{table_info}")

        for name, df in analyzed_data.get("dataframes", {}).items():
            context_blocks.append(f"DỮ LIỆU BẢNG EXCEL [{name}]:\n{df.to_markdown()}")

        compiled_context = "\n\n".join(context_blocks)
        
        if compiled_context:
            st.success(f"✅ Đã nạp TOÀN BỘ dữ liệu vào siêu bộ nhớ Flash (Tổng số nguồn: {ref_counter - 1}). Sẵn sàng viết luận văn!")
        else:
            st.info("ℹ️ Chưa có dữ liệu nền nào được nạp từ các Tab trước.")

    # ==========================================
    # 3. HIỂN THỊ LỊCH SỬ CHAT
    # ==========================================
    for msg in st.session_state["gemini_writer_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ==========================================
    # 4. XỬ LÝ LỆNH GỌI AI (VỚI CƠ CHẾ STREAMING SIÊU TỐC)
    # ==========================================
    user_query = st.chat_input("Yêu cầu AI viết (VD: Viết phần bàn luận về kết quả kiểm soát huyết áp và chèn trích dẫn [1], [2])...")

    if user_query:
        if not active_key:
            st.error("❌ Vui lòng cung cấp Gemini API Key để tiếp tục.")
            return

        # Hiển thị câu hỏi của User
        st.session_state["gemini_writer_messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Chuẩn bị gọi AI
        with st.chat_message("assistant"):
            # CẤU TRÚC PROMPTING CHUYÊN SÂU DÀNH CHO LUẬN VĂN Y KHOA (RAG)
            system_instruction = (
                "Bạn là một chuyên gia Dược lâm sàng xuất sắc, hỗ trợ nghiên cứu viên viết luận văn Chuyên khoa I. "
                "Bạn đang được cung cấp một khối lượng y văn lớn. YÊU CẦU LẬP LUẬN BẮT BUỘC:\n"
                "1. ĐỌC HIỂU ĐA NGUỒN: Phân tích và tổng hợp điểm tương đồng/khác biệt giữa các tài liệu trước khi viết.\n"
                "2. TRÍCH DẪN GỘP: Nếu nhiều tài liệu cùng ủng hộ một quan điểm, BẮT BUỘC gộp trích dẫn ở cuối câu dạng ngoặc vuông, ví dụ: [2, 5, 12].\n"
                "3. TÍNH CHUẨN XÁC: Thông tin, con số xuất phát từ tài liệu số mấy phải khớp 100% với nội dung tài liệu đó. Tuyệt đối không bịa số liệu.\n"
                "4. VĂN PHONG: Khách quan, khoa học, lập luận liền mạch theo chuẩn Y khoa.\n\n"
                f"=== DỮ LIỆU BỐI CẢNH (RAG) ===\n{compiled_context}"
            )

            # Xây dựng ngữ cảnh hội thoại ngắn gọn để truyền vào contents
            conversation_history = ""
            for m in st.session_state["gemini_writer_messages"][-5:-1]: # Lấy 4 tin nhắn gần nhất
                sender = "Người dùng" if m["role"] == "user" else "Trợ lý AI"
                conversation_history += f"{sender}: {m['content']}\n\n"

            chat_prompt = f"LỊCH SỬ TRAO ĐỔI:\n{conversation_history}\nYÊU CẦU MỚI: {user_query}"

            max_retries = 3
            
            for attempt in range(max_retries):
                try:
                    # Xoay vòng lấy key mới ở mỗi lần thử nghiệm nếu bị Rate Limit
                    current_key = get_gemini_key() if attempt > 0 else active_key
                    client = genai.Client(api_key=current_key)
                    
                    # SỬ DỤNG STREAMING ĐỂ NHẢ CHỮ TỨC THÌ
                    response_stream = client.models.generate_content_stream(
                        model="gemini-3.8-flash",
                        contents=chat_prompt,
                        config={
                            "system_instruction": system_instruction, # Đẩy ngữ cảnh khổng lồ vào System để API xử lý cực nhanh
                            "temperature": 0.2, # Giữ mức sáng tạo thấp để bám sát số liệu y khoa
                            "thinking_level": "low" # Ép AI phản hồi nhanh chóng, loại bỏ độ trễ
                        }
                    )
                    
                    # Hàm yield để Streamlit render hiệu ứng gõ chữ
                    def stream_generator():
                        for chunk in response_stream:
                            if chunk.text:
                                yield chunk.text
                    
                    # st.write_stream tự động in ra màn hình và trả về toàn bộ chuỗi khi kết thúc
                    final_response_text = st.write_stream(stream_generator())
                    
                    if not final_response_text:
                        raise ValueError("Máy chủ trả về kết quả rỗng.")

                    # Lưu kết quả vào lịch sử
                    st.session_state["gemini_writer_messages"].append({"role": "assistant", "content": final_response_text})
                    break # Thành công -> Thoát vòng lặp retry
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        st.warning(f"⏳ Cổng Gemini đang bận, đang tự động đổi Key và kết nối lại... ({e})")
                        time.sleep(2)
                    else:
                        st.error(f"❌ Các API Key đều gặp lỗi hoặc đang quá tải. Chi tiết: {e}")
                        if st.session_state["gemini_writer_messages"]:
                            st.session_state["gemini_writer_messages"].pop() # Xóa câu hỏi của user nếu AI không trả lời được
