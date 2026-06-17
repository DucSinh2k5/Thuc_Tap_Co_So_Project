from __future__ import annotations

import os
from pathlib import Path

from tien_ich.dinh_dang import dinh_dang_phan_tram, dinh_dang_vnd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

try:
    from dotenv import load_dotenv

    load_dotenv(ENV_PATH)
except ImportError:
    # Neu venv chua cai python-dotenv, van doc .env bang parser nho nay.
    if ENV_PATH.exists():
        for dong in ENV_PATH.read_text(encoding="utf-8").splitlines():
            dong = dong.strip()
            if not dong or dong.startswith("#") or "=" not in dong:
                continue
            khoa, gia_tri = dong.split("=", 1)
            khoa = khoa.strip()
            gia_tri = gia_tri.strip().strip('"').strip("'")
            if khoa and khoa not in os.environ:
                os.environ[khoa] = gia_tri


CHATBOT_SYSTEM_PROMPT = """
Bạn là chatbot AI hỗ trợ dự án dự đoán giá ô tô cũ của sinh viên.
Trả lời bằng tiếng Việt có dấu, thân thiện, rõ ràng và ưu tiên cách giải thích
phù hợp để sinh viên có thể dùng khi báo cáo với giảng viên.

Bối cảnh dự án:
- XGBoost dự đoán giá cơ sở từ dữ liệu bảng của xe.
- YOLOv8s phát hiện vùng hư hỏng ngoại thất trên ảnh xe.
- ConvNeXt-Tiny phân loại mức độ hư hỏng ảnh thành minor, moderate, severe.
- Lớp rule-based adjustment kết hợp giá cơ sở, loại hư hỏng, độ tin cậy,
  diện tích bounding box và severity để tính giá cuối.

Nguyên tắc trả lời:
- Không bịa số liệu nếu ngữ cảnh hiện tại không cung cấp.
- Nếu người dùng hỏi về kết quả demo hiện tại, hãy dựa vào phần ngữ cảnh kết quả
  được cung cấp trong system message.
- Nếu người dùng hỏi kiến thức, hãy giải thích bản chất, công thức nếu cần,
  và liên hệ với dự án ô tô cũ.
- Không tiết lộ, yêu cầu hoặc nhắc lại API key.
""".strip()


def lay_groq_api_key() -> str:
    return (os.getenv("GROQ_API_KEY") or os.getenv("GROG_API_KEY") or "").strip()


def lay_groq_model() -> str:
    return (os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile").strip()


def _lay(thong_tin: dict | None, khoa: str, mac_dinh="N/A"):
    if not isinstance(thong_tin, dict):
        return mac_dinh
    gia_tri = thong_tin.get(khoa, mac_dinh)
    if gia_tri in (None, ""):
        return mac_dinh
    return gia_tri


def tao_ngu_canh_ket_qua(ket_qua_cuoi: dict | None) -> str:
    if not ket_qua_cuoi:
        return "Chưa có kết quả dự đoán trong phiên Streamlit hiện tại."

    thong_tin_xe = ket_qua_cuoi.get("car_info", {})
    ket_qua_gia = ket_qua_cuoi.get("pricing", {})
    tong_hop = ket_qua_cuoi.get("severity_summary", {})
    danh_sach_phat_hien = ket_qua_cuoi.get("detections", [])

    dong = [
        "Ngữ cảnh kết quả demo hiện tại:",
        (
            "- Xe: "
            f"{_lay(thong_tin_xe, 'brand')} {_lay(thong_tin_xe, 'model')}, "
            f"năm {_lay(thong_tin_xe, 'year')}, "
            f"{_lay(thong_tin_xe, 'fuel_type')} / {_lay(thong_tin_xe, 'transmission')}, "
            f"đã đi {_lay(thong_tin_xe, 'km_driven')} km."
        ),
        (
            "- Giá: "
            f"base={dinh_dang_vnd(ket_qua_gia.get('base_price', 0))}, "
            f"deduction={dinh_dang_vnd(ket_qua_gia.get('deduction_amount', 0))}, "
            f"final={dinh_dang_vnd(ket_qua_gia.get('final_price', 0))}."
        ),
        (
            "- Hư hỏng: "
            f"total={tong_hop.get('total_damages', 0)}, "
            f"max_severity={tong_hop.get('max_severity', 'none')}, "
            f"average_severity_score={tong_hop.get('average_severity_score', 0)}."
        ),
        (
            "- Lý do trừ giá nổi bật: "
            f"{ket_qua_gia.get('top_deduction_reason', 'N/A')}; "
            f"tỷ lệ trừ ước tính={dinh_dang_phan_tram(ket_qua_gia.get('estimated_deduction_rate', 0))}."
        ),
    ]

    if danh_sach_phat_hien:
        dong.append("- Một số detection từ YOLO:")
        for phat_hien in danh_sach_phat_hien[:8]:
            dong.append(
                "  + "
                f"ảnh={phat_hien.get('image_name', 'N/A')}, "
                f"class={phat_hien.get('class', 'N/A')}, "
                f"confidence={phat_hien.get('confidence', 0):.2f}, "
                f"area_ratio={phat_hien.get('area_ratio', 0):.4f}."
            )

    return "\n".join(dong)


def tao_system_message(ket_qua_cuoi: dict | None = None) -> str:
    return CHATBOT_SYSTEM_PROMPT + "\n\n" + tao_ngu_canh_ket_qua(ket_qua_cuoi)


def goi_groq_chat(
    messages: list[dict[str, str]],
    model_name: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 900,
) -> str:
    api_key = lay_groq_api_key()
    if not api_key:
        return "Chưa cấu hình GROQ_API_KEY trong file .env."

    try:
        from groq import Groq
    except ImportError:
        return "Thiếu thư viện groq. Hãy cài dependencies trong requirements.txt rồi chạy lại app."

    client = Groq(api_key=api_key)
    chat_completion = client.chat.completions.create(
        model=model_name or lay_groq_model(),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return chat_completion.choices[0].message.content or "Mình chưa tạo được câu trả lời."
