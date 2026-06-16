from datetime import datetime
import math

import streamlit as st

from dich_vu.dinh_gia import du_doan_gia_co_ban, tinh_dieu_chinh_gia
from giao_dien.bo_cuc import chen_css_co_ban, hien_tieu_de_dau_trang, hien_tieu_de_muc
from giao_dien.thanh_phan import (
    doc_anh_da_tai,
    hien_form_thong_tin_xe,
    hien_ket_qua_gia,
    hien_ket_qua_muc_do,
    hien_ket_qua_phat_hien,
    hien_tai_anh_len,
    hien_xem_truoc_anh,
)
from tien_ich.du_lieu_mau import HOP_SO, LOAI_NHIEN_LIEU, THONG_TIN_XE_MAC_DINH
from tien_ich.trang_thai import khoi_tao_trang_thai


def tong_hop_muc_do_rong():
    return {
        "total_damages": 0,
        "num_dents": 0,
        "num_scratches": 0,
        "num_cracks": 0,
        "max_severity": "none",
        "average_severity_score": 0.0,
    }


def chay_pipeline(thong_tin_xe, danh_sach_anh):
    """Chạy pipeline dự đoán giá từ thông tin xe và ảnh."""
    thong_tin_xe_model = chuan_hoa_thong_tin_xe(thong_tin_xe)
    gia_co_ban = du_doan_gia_co_ban(thong_tin_xe_model)

    if danh_sach_anh:
        from dich_vu.muc_do_hu_hong import phan_loai_muc_do, tong_hop_muc_do
        from dich_vu.phat_hien_hu_hong import phat_hien_hu_hong, ve_bbox_anh

        danh_sach_phat_hien = phat_hien_hu_hong(danh_sach_anh)
        anh_voi_bbox = ve_bbox_anh(danh_sach_anh, danh_sach_phat_hien)
        danh_sach_muc_do = phan_loai_muc_do(danh_sach_anh)
        tong_hop_muc_do_kq = tong_hop_muc_do(danh_sach_phat_hien, danh_sach_muc_do)
    else:
        danh_sach_phat_hien = []
        anh_voi_bbox = {}
        danh_sach_muc_do = []
        tong_hop_muc_do_kq = tong_hop_muc_do_rong()

    ket_qua_gia = tinh_dieu_chinh_gia(gia_co_ban, danh_sach_phat_hien, danh_sach_muc_do)

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "car_info": thong_tin_xe,
        "car_info_model": thong_tin_xe_model,
        "images": danh_sach_anh,
        "detections": danh_sach_phat_hien,
        "annotated_images": anh_voi_bbox,
        "severities": danh_sach_muc_do,
        "severity_summary": tong_hop_muc_do_kq,
        "pricing": ket_qua_gia,
    }


def kiem_tra_dau_vao(thong_tin_xe, danh_sach_anh):
    """Kiểm tra dữ liệu đầu vào và trả về danh sách cảnh báo."""
    canh_bao = []
    if not thong_tin_xe.get("brand", "").strip():
        canh_bao.append("Brand is required.")
    if not thong_tin_xe.get("model", "").strip():
        canh_bao.append("Model is required.")
    if not thong_tin_xe.get("fuel_type", "").strip():
        canh_bao.append("Fuel type is required.")
    if not thong_tin_xe.get("transmission", "").strip():
        canh_bao.append("Transmission is required.")
    return canh_bao


FUEL_TYPE_MAP = {
    "Petrol": 0,
    "Diesel": 1,
    "CNG": 2,
    "LPG": 3,
    "Electric": 4,
    "Hybrid": 5,
}

TRANSMISSION_MAP = {
    "Manual": 0,
    "Automatic": 1,
    "CVT": 1,
    "Semi-Automatic": 1,
}


def _to_float_or_nan(value):
    if value is None or value == "":
        return float("nan")
    return float(value)


def chuan_hoa_thong_tin_xe(thong_tin_xe):
    brand = (thong_tin_xe.get("brand") or "").strip()
    model = (thong_tin_xe.get("model") or "").strip()
    ten_xe = f"{brand} {model}".strip()

    year = thong_tin_xe.get("year") or datetime.now().year
    tuoi_xe = max(datetime.now().year - int(year), 1)

    km_driven = _to_float_or_nan(thong_tin_xe.get("km_driven"))
    if math.isnan(km_driven):
        km_driven = 0.0

    km_moi_nam = km_driven / tuoi_xe
    log_quang_duong = math.log1p(max(km_driven, 0.0))
    chay_nhieu = 1 if km_moi_nam > 15000 else 0

    loai_nhien_lieu = FUEL_TYPE_MAP.get(thong_tin_xe.get("fuel_type"))
    if loai_nhien_lieu is None:
        loai_nhien_lieu = float("nan")

    hop_so = TRANSMISSION_MAP.get(thong_tin_xe.get("transmission"))
    if hop_so is None:
        hop_so = float("nan")

    quyen_so_huu = _to_float_or_nan(thong_tin_xe.get("owner_count"))

    hang_xe = brand or (ten_xe.split()[0] if ten_xe else "Other")
    if ten_xe.lower().startswith("land rover") or brand.lower() == "land rover":
        hang_xe = "Land Rover"

    return {
        "Loai_nhien_lieu": loai_nhien_lieu,
        "Hop_so": hop_so,
        "Quyen_so_huu": quyen_so_huu,
        "Muc_tieu_hao(km/l)": _to_float_or_nan(thong_tin_xe.get("fuel_consumption")),
        "Dung_tich(cc)": _to_float_or_nan(thong_tin_xe.get("engine_cc")),
        "Cong_suat_toi_da": _to_float_or_nan(thong_tin_xe.get("max_power")),
        "So_cho_ngoi": _to_float_or_nan(thong_tin_xe.get("num_seats")),
        "Tuoi_xe": float(tuoi_xe),
        "Hang_xe": hang_xe,
        "Km_moi_nam": km_moi_nam,
        "Chay_nhieu": chay_nhieu,
        "log_Quang_duong_da_di(km)": log_quang_duong,
        "Top_xe": ten_xe or "Other",
    }


def chay_ung_dung():
    st.set_page_config(
        page_title="Used Car Price AI Demo",
        page_icon="🚗",
        layout="wide",
    )

    chen_css_co_ban()
    khoi_tao_trang_thai(THONG_TIN_XE_MAC_DINH)

    hien_tieu_de_dau_trang()

    hien_tieu_de_muc("Input", "Provide car details. Upload images if you want damage-aware pricing.")
    thong_tin_xe = hien_form_thong_tin_xe(THONG_TIN_XE_MAC_DINH, LOAI_NHIEN_LIEU, HOP_SO)
    tep_da_tai = hien_tai_anh_len()
    danh_sach_anh = doc_anh_da_tai(tep_da_tai)

    hien_xem_truoc_anh(danh_sach_anh)

    bam_du_doan = st.button("Analyze Car", type="primary", use_container_width=True)
    if bam_du_doan:
        canh_bao = kiem_tra_dau_vao(thong_tin_xe, danh_sach_anh)
        if canh_bao:
            for thong_bao in canh_bao:
                st.warning(thong_bao)
        else:
            with st.spinner("Running prediction pipeline..."):
                ket_qua = chay_pipeline(thong_tin_xe, danh_sach_anh)
            st.session_state["ket_qua_cuoi"] = ket_qua
            st.success("Prediction complete.")

    st.divider()

    if st.session_state.get("ket_qua_cuoi"):
        ket_qua = st.session_state["ket_qua_cuoi"]

        if ket_qua["images"]:
            from dich_vu.muc_do_hu_hong import ghep_chi_tiet_muc_do

            hien_tieu_de_muc("Detection Results")
            hien_ket_qua_phat_hien(
                ket_qua["images"],
                ket_qua["detections"],
                ket_qua["annotated_images"],
            )

            hien_tieu_de_muc("Severity Results")
            bang_muc_do = ghep_chi_tiet_muc_do(ket_qua["severities"])
            hien_ket_qua_muc_do(ket_qua["severity_summary"], bang_muc_do)
        else:
            st.info("No images uploaded. Final price is based on the XGBoost tabular prediction only.")

        hien_tieu_de_muc("Pricing Results")
        hien_ket_qua_gia(ket_qua["pricing"], ket_qua["severity_summary"])

    else:
        st.info("Provide inputs to see results.")


if __name__ == "__main__":
    chay_ung_dung()
