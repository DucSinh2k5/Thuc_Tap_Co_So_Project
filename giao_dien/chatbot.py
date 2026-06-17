from __future__ import annotations

import html

import streamlit as st

from dich_vu.chatbot import (
    goi_groq_chat,
    lay_groq_api_key,
    lay_groq_model,
    tao_system_message,
)


KHOA_TIN_NHAN = "car_chat_messages"
KHOA_MO_CHAT = "car_chat_open"


def khoi_tao_chatbot() -> None:
    if KHOA_TIN_NHAN not in st.session_state:
        st.session_state[KHOA_TIN_NHAN] = [
            {
                "role": "assistant",
                "content": (
                    "Xin chào, mình là trợ lý AI cho dự án định giá ô tô cũ. "
                    "Bạn có thể hỏi về XGBoost, YOLO, ConvNeXt-Tiny hoặc nhờ mình "
                    "giải thích kết quả dự đoán hiện tại."
                ),
            }
        ]
    st.session_state.setdefault(KHOA_MO_CHAT, False)


def dat_lai_chatbot() -> None:
    st.session_state[KHOA_TIN_NHAN] = [
        {
            "role": "assistant",
            "content": "Đã xoá lịch sử chat. Bạn muốn hỏi gì tiếp nào?",
        }
    ]


def gui_tin_nhan_chatbot(user_prompt: str, ket_qua_cuoi: dict | None = None) -> None:
    prompt = user_prompt.strip()
    if not prompt:
        return

    st.session_state[KHOA_TIN_NHAN].append({"role": "user", "content": prompt})
    lich_su = [
        {"role": "system", "content": tao_system_message(ket_qua_cuoi)},
        *st.session_state[KHOA_TIN_NHAN][-10:],
    ]

    try:
        cau_tra_loi = goi_groq_chat(lich_su, model_name=lay_groq_model())
    except Exception as exc:
        cau_tra_loi = f"Không gọi được Groq API: {exc}"

    st.session_state[KHOA_TIN_NHAN].append(
        {"role": "assistant", "content": cau_tra_loi}
    )


def _chen_css_chatbot() -> None:
    st.markdown(
        """
        <style>
        .st-key-car_chatbot_launcher,
        div[data-testid="stElementContainer"]:has(.st-key-car_chatbot_launcher) {
            position: fixed !important;
            right: 24px !important;
            bottom: 24px !important;
            width: 76px !important;
            max-width: 76px !important;
            z-index: 1000000 !important;
            display: flex !important;
            justify-content: flex-end !important;
            pointer-events: auto !important;
        }
        .st-key-car_chatbot_launcher button {
            width: 64px;
            height: 64px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.62);
            background: #0f766e;
            color: #ffffff;
            font-size: 17px;
            font-weight: 800;
            box-shadow: 0 18px 42px rgba(15, 118, 110, 0.34);
        }
        .st-key-car_chatbot_launcher button:hover {
            background: #115e59;
            border-color: rgba(255,255,255,0.9);
        }
        .st-key-car_chatbot_panel,
        div[data-testid="stElementContainer"]:has(.st-key-car_chatbot_panel) {
            position: fixed !important;
            right: 24px !important;
            bottom: 100px !important;
            width: min(430px, calc(100vw - 32px)) !important;
            max-height: min(74vh, 680px);
            overflow-y: auto;
            z-index: 999999 !important;
            background: #f8fafc;
            border: 1px solid #b7d8d4;
            border-radius: 8px;
            box-shadow: 0 24px 68px rgba(15, 23, 42, 0.30);
            padding: 16px 16px 14px 16px;
        }
        .st-key-car_chatbot_panel [data-testid="stVerticalBlock"] {
            gap: 0.55rem;
        }
        .car-chat-title {
            font-size: 24px;
            line-height: 1.08;
            font-weight: 800;
            color: #134e4a;
            margin: 0;
        }
        .car-chat-status {
            color: #475569;
            font-size: 13px;
            font-weight: 600;
            margin-top: 2px;
        }
        .car-chat-divider {
            height: 1px;
            background: #cbd5e1;
            margin: 8px -16px 10px -16px;
        }
        .car-message-row {
            display: flex;
            margin: 8px 0;
        }
        .car-message-row.user {
            justify-content: flex-end;
        }
        .car-message-row.assistant {
            justify-content: flex-start;
        }
        .car-message-bubble {
            max-width: 88%;
            border-radius: 8px;
            padding: 10px 13px;
            font-size: 14px;
            line-height: 1.45;
            border: 1px solid #d7e3e1;
            word-break: break-word;
        }
        .car-message-row.assistant .car-message-bubble {
            background: #ffffff;
            color: #0f172a;
        }
        .car-message-row.user .car-message-bubble {
            background: #0f766e;
            color: #ffffff;
            border-color: #0f766e;
        }
        .car-chat-hint {
            color: #64748b;
            font-size: 12px;
            margin-top: 2px;
        }
        .st-key-car_chatbot_panel button {
            border-radius: 6px;
            font-weight: 700;
        }
        .st-key-car_chatbot_panel div[data-testid="stForm"] {
            background: #e6f3f1;
            border-radius: 8px;
            padding: 12px;
            border: 1px solid #c7dedb;
        }
        @media (max-width: 640px) {
            .st-key-car_chatbot_panel,
            div[data-testid="stElementContainer"]:has(.st-key-car_chatbot_panel) {
                right: 12px !important;
                bottom: 88px !important;
                width: calc(100vw - 24px) !important;
            }
            .st-key-car_chatbot_launcher,
            div[data-testid="stElementContainer"]:has(.st-key-car_chatbot_launcher) {
                right: 16px !important;
                bottom: 16px !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _hien_thi_tin_nhan() -> None:
    for tin_nhan in st.session_state[KHOA_TIN_NHAN][-8:]:
        role = tin_nhan.get("role", "assistant")
        noi_dung = html.escape(str(tin_nhan.get("content", ""))).replace("\n", "<br>")
        row_role = "user" if role == "user" else "assistant"
        st.markdown(
            f"""
            <div class="car-message-row {row_role}">
                <div class="car-message-bubble">{noi_dung}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _hien_goi_y_nhanh(ket_qua_cuoi: dict | None) -> None:
    goi_y = [
        ("Giải thích giá", "Hãy giải thích kết quả định giá hiện tại theo cách em có thể nói khi báo cáo."),
        ("XGBoost", "XGBoost đang đóng vai trò gì trong dự án dự đoán giá ô tô cũ này?"),
        ("YOLO", "YOLO phát hiện hư hỏng như thế nào và output gồm những gì?"),
        ("ConvNeXt", "ConvNeXt-Tiny phân loại mức độ hư hỏng trong dự án này như thế nào?"),
    ]

    cot = st.columns(2)
    for idx, (nhan, prompt) in enumerate(goi_y):
        with cot[idx % 2]:
            if st.button(nhan, key=f"car_quick_prompt_{idx}", use_container_width=True):
                with st.spinner("Chatbot đang trả lời..."):
                    gui_tin_nhan_chatbot(prompt, ket_qua_cuoi)
                st.rerun()


def hien_chatbot_noi(ket_qua_cuoi: dict | None = None) -> None:
    khoi_tao_chatbot()
    _chen_css_chatbot()

    if st.session_state[KHOA_MO_CHAT]:
        with st.container(key="car_chatbot_panel"):
            cot_tieu_de, cot_dong = st.columns([0.82, 0.18], vertical_alignment="center")
            with cot_tieu_de:
                trang_thai = "Sẵn sàng" if lay_groq_api_key() else "Thiếu API key"
                st.markdown(
                    f"""
                    <div class="car-chat-title">Car Assistant</div>
                    <div class="car-chat-status">{trang_thai} · {html.escape(lay_groq_model())}</div>
                    """,
                    unsafe_allow_html=True,
                )
            with cot_dong:
                if st.button("x", key="car_chatbot_close", help="Đóng chatbot"):
                    st.session_state[KHOA_MO_CHAT] = False
                    st.rerun()

            st.markdown('<div class="car-chat-divider"></div>', unsafe_allow_html=True)

            if not lay_groq_api_key():
                st.warning("Thiếu GROQ_API_KEY trong file .env. Thêm key rồi restart Streamlit.")

            _hien_thi_tin_nhan()
            _hien_goi_y_nhanh(ket_qua_cuoi)

            with st.form("car_floating_chat_form", clear_on_submit=True):
                user_prompt = st.text_input(
                    "Nhập tin nhắn",
                    max_chars=1000,
                    placeholder="Ví dụ: Vì sao xe bị trừ giá?",
                )
                cot_gui, cot_xoa = st.columns([1, 1])
                with cot_gui:
                    submitted = st.form_submit_button(
                        "Gửi",
                        type="primary",
                        use_container_width=True,
                    )
                with cot_xoa:
                    clear_chat = st.form_submit_button(
                        "Xóa chat",
                        use_container_width=True,
                    )

            if clear_chat:
                dat_lai_chatbot()
                st.rerun()
            if submitted and user_prompt.strip():
                with st.spinner("Chatbot đang trả lời..."):
                    gui_tin_nhan_chatbot(user_prompt, ket_qua_cuoi)
                st.rerun()

            st.markdown(
                '<div class="car-chat-hint">Không nhập API key, mật khẩu hoặc dữ liệu quá nhạy cảm.</div>',
                unsafe_allow_html=True,
            )

    with st.container(key="car_chatbot_launcher"):
        nhan_nut = "x" if st.session_state[KHOA_MO_CHAT] else "AI"
        if st.button(nhan_nut, key="car_chatbot_toggle", help="Mở chatbot AI"):
            st.session_state[KHOA_MO_CHAT] = not st.session_state[KHOA_MO_CHAT]
            st.rerun()
