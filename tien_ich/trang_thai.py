import streamlit as st

from tien_ich.du_lieu_mau import THONG_TIN_XE_MAC_DINH

KHOA_FORM = [
    "brand",
    "model",
    "year",
    "num_seats",
    "km_driven",
    "fuel_type",
    "transmission",
    "owner_count",
    "fuel_consumption",
    "engine_cc",
    "max_power",
]


def khoi_tao_trang_thai(gia_tri_mac_dinh=None):
    """Khởi tạo session state cho ứng dụng."""
    gia_tri_mac_dinh = gia_tri_mac_dinh or THONG_TIN_XE_MAC_DINH

    st.session_state.setdefault("ket_qua_cuoi", None)
    for khoa in KHOA_FORM:
        if khoa not in st.session_state:
            st.session_state[khoa] = gia_tri_mac_dinh.get(khoa)


def dat_lai_form(gia_tri_mac_dinh=None):
    """Đặt lại form về giá trị mặc định."""
    gia_tri_mac_dinh = gia_tri_mac_dinh or THONG_TIN_XE_MAC_DINH
    for khoa in KHOA_FORM:
        st.session_state[khoa] = gia_tri_mac_dinh.get(khoa)
