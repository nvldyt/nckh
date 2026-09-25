import os
import time
import streamlit as st
from google import genai
import random

def get_gemini_key() -> str:
    """
    Tự động xoay vòng lấy API Key từ danh sách trong st.secrets.
    Yêu cầu cấu hình st.secrets (ví dụ trong file .streamlit/secrets.toml hoặc trên Streamlit Cloud):
    [GEMINI]
    API_KEYS = [
        "AIzaSyxxxxxxxxxxxxxxxxx",
        "AIzaSyyyyyyyyyyyyyyyyyy",
        "AIzaSzzzzzzzzzzzzzzzzzz"
    ]
    """
    try:
        if "GEMINI" in st.secrets and "API_KEYS" in st.secrets["GEMINI"]:
            keys = st.secrets["GEMINI"]["API_KEYS"]
            if keys and isinstance(keys, list):
                # Cơ chế xoay vòng: Bốc ngẫu nhiên một khóa để cân bằng tải
                return random.choice(keys).strip()
            elif isinstance(keys, str):
                return keys.strip()
    except Exception:
        pass
        
    # Cứu cánh cuối cùng: lấy từ biến môi trường
    return os.getenv("GEMINI_API_KEY", "")

def render_gemini_flash_tab():
    st.markdown("### 🚀 Trợ lý Viết Luận văn (Gemini 3.8 Flash)")
    st.caption("Ứng dụng mô hình Gemini 3.8 Flash thế hệ mới với bộ nhớ siêu dài, tối ưu cho tổng hợp tài liệu và viết luận văn chuyên sâu.")

    # 1. QUẢN LÝ KHÓA API
    active_key = get_gemini_key()
    
    # Cho phép người dùng nhập khóa riêng nếu st.secrets không có
    if not active_key:
        active_key = st.text_input(
            "🔑 Nhập Google Gemini API Key:", 
            type="password", 
            key="input_gemini_key_tab6",
            help="Hệ thống không tìm thấy khóa mặc định trong secrets. Vui lòng nhập thủ công."
        )
    else:
        # Tùy chọn: Hiển thị thông báo nhỏ báo hiệu đang dùng key từ hệ thống
        pass 

    if "gemini_writer_messages" not in st.session_state:
        st.session_state["gemini_writer_messages"] = []

    # 2. NẠP TOÀN BỘ DỮ LIỆU BỐI CẢNH (LONG-CONTEXT PROMPTING)
    with st.expander("🔍 Dữ liệu bối cảnh và danh mục tham khảo đang nạp", expanded=False):
        context_blocks = []
        
        # Kéo toàn bộ PDF và Bài báo đã nạp từ Tab 1 & Tab 2
        docs = st.session_state.get("documents", {})
        chunks = st.session_state.get("chunks", [])
        
        ref_counter = 1
        
        if docs and chunks:
            doc_list_text = []
            doc_mapping = {} # Bản đồ ánh xạ ID -> Số thứ tự [1], [2]
            
            # 2.1. Xây dựng Danh mục tài liệu tham khảo cho AI
            for sid, meta in docs.items():
                doc_mapping[sid] = ref_counter
                title = meta.get("title") or meta.get("file_name") or sid
                authors = meta.get("authors", "Không rõ tác giả")
                year = meta.get("year", "")
                doc_list_text.append(f"[{ref_counter}] {authors}. {title}. {year}")
                ref_counter += 1
                
            context_blocks.append("DANH MỤC TÀI LIỆU GỐC:\n" + "\n".join(doc_list_text))
            
            # 2.2. Nhồi TOÀN BỘ các đoạn văn (Chunks) vào bộ nhớ Flash
            ev_lines = []
            for c in chunks:
                sid = c.get("source_id")
                doc_idx = doc_mapping.get(sid, "?")
                text = c.get("text", "")
                if text.strip():
                    # Đóng dấu số [1], [2] vào từng đoạn để AI biết trích dẫn từ đâu
                    ev_lines.append(f"[Trích đoạn từ tài liệu {doc_idx}]:\n{text}")
                    
            context_blocks.append("NỘI DUNG CHI TIẾT TỪ CÁC TÀI LIỆU:\n" + "\n---\n".join(ev_lines))
            
        # 2.3. Nạp thêm Tóm tắt (nếu có)
        summary = st.session_state.get("cached_summary", "")
        if summary:
            context_blocks.append(f"TÓM TẮT ĐỀ TÀI (Nguồn nội bộ [{ref_counter}]):\n{summary}")
            ref_counter += 1

        # 2.4. Nạp thêm Bảng số liệu từ SPSS (nếu có)
        saved_tables = st.session_state.get("saved_tables", {})
        if saved_tables:
            table_info = "".join([f"Bảng {name}:\n{df.to_markdown()}\n\n" for name, df in saved_tables.items()])
            context_blocks.append(f"BẢNG SỐ LIỆU NGHIÊN CỨU:\n{table_info}")

        compiled_context = "\n\n".join(context_blocks)
        
        if compiled_context:
            st.success(f"✅ Đã nạp TOÀN BỘ dữ liệu vào siêu bộ nhớ Flash (Tổng số nguồn: {ref_counter - 1}). Sẵn sàng viết luận văn!")
        else:
            st.info("ℹ️ Chưa có dữ liệu nền nào được nạp từ các Tab trước.")

    # 3. HIỂN THỊ LỊCH SỬ CHAT
    for msg in st.session_state["gemini_writer_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 4. XỬ LÝ LỆNH GỌI AI
    user_query = st.chat_input("Yêu cầu AI viết (VD: Viết phần bàn luận về kết quả kiểm soát huyết áp và chèn trích dẫn [1], [2])...")

    if user_query:
        if not active_key:
            st.error("❌ Vui lòng cung cấp Gemini API Key để tiếp tục.")
            return

        st.session_state["gemini_writer_messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            system_instruction = (
                "Bạn là một chuyên gia Dược lâm sàng xuất sắc, hỗ trợ nghiên cứu viên viết luận văn Chuyên khoa I. "
                "YÊU CẦU BẮT BUỘC VỀ TRÍCH DẪN: "
                "1. Khi sử dụng thông tin, số liệu, hoặc kết luận từ các tài liệu được cung cấp, bạn PHẢI đính kèm số thứ tự tài liệu tham khảo dạng ngoặc vuông ở cuối câu (ví dụ: [1], [2]). "
                "2. Các số trích dẫn phải tuân thủ đúng thứ tự xuất hiện của nguồn tài liệu trong ngữ cảnh bên dưới. "
                "3. Tuyệt đối không bịa đặt số liệu hoặc tự ý gán nguồn sai sự thật. "
                "4. Văn phong: Khách quan, khoa học, chuẩn mực y khoa, lập luận liền mạch.\n\n"
                f"=== DỮ LIỆU ĐỀ TÀI VÀ NGUỒN THAM KHẢO ===\n{compiled_context}"
            )

            # Xây dựng ngữ cảnh hội thoại
            conversation_history = ""
            for m in st.session_state["gemini_writer_messages"][-4:]:
                sender = "Người dùng" if m["role"] == "user" else "Trợ lý AI"
                conversation_history += f"{sender}: {m['content']}\n\n"

            full_prompt = (
                f"{system_instruction}\n\n"
                f"=== LỊCH SỬ TRAO ĐỔI GẦN ĐÂY ===\n{conversation_history}"
                f"YÊU CẦU HIỆN TẠI TỪ NGHIÊN CỨU VIÊN: {user_query}\n"
                "TRẢ LỜI CỦA TRỢ LÝ AI:"
            )

            max_retries = 3
            success = False
            
            for attempt in range(max_retries):
                try:
                    with st.spinner(f"🚀 Gemini 3.8 Flash đang phân tích y văn và viết bản thảo (Lần thử {attempt + 1})..."):
                        # Xoay vòng lấy key mới ở mỗi lần thử nghiệm (trường hợp key cũ bị giới hạn)
                        current_key = get_gemini_key() if attempt > 0 else active_key
                        client = genai.Client(api_key=current_key)
                        
                        try:
                            # CẤU HÌNH MỚI: Thêm generation_config để giảm độ trễ (Thinking Level)
                            interaction = client.interactions.create(
                                model="gemini-3.8-flash",
                                input=full_prompt,
                                generation_config={
                                    "thinking_level": "low"  # Ép AI phản hồi nhanh, bỏ qua suy luận thừa
                                }
                            )
                            response_text = interaction.output_text
                        except AttributeError:
                            # Fallback cho phiên bản thư viện cũ hơn
                            res = client.models.generate_content(
                                model="gemini-3.8-flash",
                                contents=full_prompt
                            )
                            response_text = res.text

                        if not response_text:
                            raise ValueError("Phản hồi nhận được từ máy chủ rỗng.")

                        st.markdown(response_text)
                        st.session_state["gemini_writer_messages"].append({"role": "assistant", "content": response_text})
                        success = True
                        break # Thoát vòng lặp retry nếu thành công
                        
                except Exception as e:
                    if attempt < max_retries - 1:
                        st.warning(f"⏳ Cổng Gemini đang bận hoặc giới hạn token, đang tự động đổi Key và thử lại... ({e})")
                        time.sleep(2) # Nghỉ 2 giây trước khi thử lại
                    else:
                        st.error(f"❌ Toàn bộ các API Key đều gặp lỗi hoặc đang quá tải. Chi tiết: {e}")
                        if st.session_state["gemini_writer_messages"]:
                            st.session_state["gemini_writer_messages"].pop()
