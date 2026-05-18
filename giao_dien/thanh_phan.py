import json
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image

from tien_ich.dinh_dang import dinh_dang_phan_tram, dinh_dang_vnd


def _chi_so_an_toan(lua_chon, gia_tri):
    if gia_tri in lua_chon:
        return lua_chon.index(gia_tri)
    return 0


def hien_form_thong_tin_xe(
    gia_tri_mac_dinh,
    loai_nhien_lieu,
    hop_so,
):
    # """Hiển thị form thông tin xe và trả về dữ liệu."""
    st.subheader("Car Information Form")

    cot1, cot2 = st.columns(2)
    hang_xe = cot1.text_input("Brand", value=gia_tri_mac_dinh.get("brand", ""), key="brand")
    dong_xe = cot2.text_input("Model", value=gia_tri_mac_dinh.get("model", ""), key="model")

    nam_sx = cot1.number_input(
        "Year",
        min_value=1980,
        max_value=datetime.now().year,
        value=int(gia_tri_mac_dinh.get("year", datetime.now().year)),
        step=1,
        key="year",
    )
    so_ghe = cot2.number_input(
        "Number of Seats",
        min_value=2,
        max_value=12,
        value=int(gia_tri_mac_dinh.get("num_seats", 5)),
        step=1,
        key="num_seats",
    )

    km_da_di = cot1.number_input(
        "KM Driven",
        min_value=0,
        max_value=1_000_000,
        value=int(gia_tri_mac_dinh.get("km_driven", 50_000)),
        step=1_000,
        key="km_driven",
    )
    nhien_lieu = cot2.selectbox(
        "Fuel Type",
        options=loai_nhien_lieu,
        index=_chi_so_an_toan(loai_nhien_lieu, gia_tri_mac_dinh.get("fuel_type", loai_nhien_lieu[0])),
        key="fuel_type",
    )

    hop_so_xe = cot1.selectbox(
        "Transmission",
        options=hop_so,
        index=_chi_so_an_toan(hop_so, gia_tri_mac_dinh.get("transmission", hop_so[0])),
        key="transmission",
    )
    so_chu = cot2.number_input(
        "Owner Count",
        min_value=1,
        max_value=10,
        value=int(gia_tri_mac_dinh.get("owner_count", 1)),
        step=1,
        key="owner_count",
    )

    muc_tieu_hao = cot1.number_input(
        "Fuel Consumption (km/l)",
        min_value=0.0,
        max_value=100.0,
        value=float(gia_tri_mac_dinh.get("fuel_consumption", 0.0)),
        step=0.1,
        key="fuel_consumption",
    )
    dung_tich = cot2.number_input(
        "Engine Displacement (cc)",
        min_value=0.0,
        max_value=10000.0,
        value=float(gia_tri_mac_dinh.get("engine_cc", 0.0)),
        step=50.0,
        key="engine_cc",
    )
    cong_suat = cot1.number_input(
        "Max Power",
        min_value=0.0,
        max_value=2000.0,
        value=float(gia_tri_mac_dinh.get("max_power", 0.0)),
        step=5.0,
        key="max_power",
    )

    return {
        "brand": hang_xe,
        "model": dong_xe,
        "year": nam_sx,
        "num_seats": so_ghe,
        "km_driven": km_da_di,
        "fuel_type": nhien_lieu,
        "transmission": hop_so_xe,
        "owner_count": so_chu,
        "fuel_consumption": muc_tieu_hao,
        "engine_cc": dung_tich,
        "max_power": cong_suat,
    }


def hien_tai_anh_len():
    # """Hiển thị khu vực tải ảnh và trả về danh sách tệp."""
    tep_da_tai = st.file_uploader(
        "Upload car images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="uploaded_images",
    )
    return tep_da_tai or []


def doc_anh_da_tai(tep_da_tai):
    # """Đọc các ảnh đã tải và trả về danh sách dict (tên, ảnh)."""
    danh_sach_anh = []
    for tep in tep_da_tai:
        tep.seek(0)
        anh = Image.open(tep).convert("RGB")
        danh_sach_anh.append({"name": tep.name, "image": anh})
    return danh_sach_anh


def hien_xem_truoc_anh(danh_sach_anh):
    # """Hiển thị xem trước các ảnh đã tải."""
    if not danh_sach_anh:
        st.info("Upload images to see a preview.")
        return

    st.subheader("Image Preview")
    cot = st.columns(3)
    for chi_so, muc_anh in enumerate(danh_sach_anh):
        cot[chi_so % 3].image(
            muc_anh["image"],
            caption=muc_anh["name"],
            use_container_width=True,
        )


def hien_ket_qua_phat_hien(
    danh_sach_anh,
    danh_sach_phat_hien,
    anh_voi_bbox,
):
    # """Hiển thị ảnh gốc, ảnh bbox và bảng phát hiện hư hỏng."""
    if not danh_sach_anh:
        st.info("No images to display.")
        return

    st.subheader("Damage Detection Viewer")
    for muc_anh in danh_sach_anh:
        cot1, cot2 = st.columns(2)
        cot1.image(
            muc_anh["image"],
            caption=f"{muc_anh['name']} (original)",
            use_container_width=True,
        )
        anh_bbox = anh_voi_bbox.get(muc_anh["name"], muc_anh["image"])
        cot2.image(
            anh_bbox,
            caption=f"{muc_anh['name']} (damage overlay)",
            use_container_width=True,
        )

    st.subheader("Detected Damages")
    if not danh_sach_phat_hien:
        st.info("No damages detected in this run.")
        return

    hang = []
    for phat_hien in danh_sach_phat_hien:
        hang.append(
            {
                "image_name": phat_hien.get("image_name"),
                "class": phat_hien.get("class"),
                "confidence": f"{phat_hien.get('confidence', 0):.2f}",
                "bbox": phat_hien.get("bbox"),
                "area_ratio": dinh_dang_phan_tram(phat_hien.get("area_ratio", 0.0)),
            }
        )
    bang = pd.DataFrame(hang)
    st.dataframe(bang, use_container_width=True, hide_index=True)


def hien_ket_qua_muc_do(tong_hop_muc_do, bang_muc_do):
    # """Hiển thị tóm tắt mức độ và bảng chi tiết."""
    st.subheader("Severity Analysis")

    cot1, cot2, cot3, cot4 = st.columns(4)
    cot1.metric("Total Damages", tong_hop_muc_do.get("total_damages", 0))
    cot2.metric("Max Severity", tong_hop_muc_do.get("max_severity", "none").title())
    cot3.metric(
        "Average Severity Score",
        f"{tong_hop_muc_do.get('average_severity_score', 0.0):.2f}",
    )
    cot4.metric("Dents", tong_hop_muc_do.get("num_dents", 0))

    bang_tom_tat = [
        {"Metric": "Total damages", "Value": tong_hop_muc_do.get("total_damages", 0)},
        {"Metric": "Num dents", "Value": tong_hop_muc_do.get("num_dents", 0)},
        {"Metric": "Num scratches", "Value": tong_hop_muc_do.get("num_scratches", 0)},
        {"Metric": "Num cracks", "Value": tong_hop_muc_do.get("num_cracks", 0)},
        {"Metric": "Max severity", "Value": tong_hop_muc_do.get("max_severity", "none")},
        {
            "Metric": "Average severity score",
            "Value": f"{tong_hop_muc_do.get('average_severity_score', 0.0):.2f}",
        },
    ]
    st.dataframe(pd.DataFrame(bang_tom_tat), use_container_width=True, hide_index=True)

    st.subheader("Severity Details")
    if not bang_muc_do:
        st.info("No severity results to display.")
        return

    st.dataframe(pd.DataFrame(bang_muc_do), use_container_width=True, hide_index=True)


def hien_ket_qua_gia(ket_qua_gia, tong_hop_muc_do):
    # """Hiển thị chỉ số giá và giải thích."""
    cot1, cot2, cot3 = st.columns(3)
    cot1.metric("Base Price", dinh_dang_vnd(ket_qua_gia.get("base_price", 0.0)))
    cot2.metric("Damage Deduction", dinh_dang_vnd(ket_qua_gia.get("deduction_amount", 0.0)))
    cot3.metric("Final Adjusted Price", dinh_dang_vnd(ket_qua_gia.get("final_price", 0.0)))

    st.markdown("**Pricing Explanation**")
    st.write(f"Top deduction reason: {ket_qua_gia.get('top_deduction_reason', 'N/A')}")
    st.write(f"Total damage score: {dinh_dang_phan_tram(ket_qua_gia.get('damage_score', 0.0))}")

    bang_tom_tat = [
        {
            "Metric": "Number of damages",
            "Value": tong_hop_muc_do.get("total_damages", 0),
        },
        {
            "Metric": "Highest severity",
            "Value": tong_hop_muc_do.get("max_severity", "none"),
        },
        {
            "Metric": "Estimated deduction rate",
            "Value": dinh_dang_phan_tram(ket_qua_gia.get("damage_score", 0.0)),
        },
        {
            "Metric": "Final adjusted price",
            "Value": dinh_dang_vnd(ket_qua_gia.get("final_price", 0.0)),
        },
    ]
    st.dataframe(pd.DataFrame(bang_tom_tat), use_container_width=True, hide_index=True)


def hien_bang_lich_su(lich_su):
    # """Hiển thị lịch sử dự đoán (không dùng hiện tại)."""
    if not lich_su:
        st.info("No history yet.")
        return

    bang = pd.DataFrame(lich_su)
    if bang.empty:
        st.info("No history yet.")
        return

    bang["base_price"] = bang["base_price"].apply(dinh_dang_vnd)
    bang["final_price"] = bang["final_price"].apply(dinh_dang_vnd)
    st.dataframe(bang, use_container_width=True, hide_index=True)


def hien_nut_xuat(ket_qua_xuat, lich_su):
    # """Hiển thị nút xuất JSON/CSV (không dùng hiện tại)."""
    if not ket_qua_xuat and not lich_su:
        return

    cot1, cot2 = st.columns(2)

    if ket_qua_xuat:
        du_lieu_json = json.dumps(ket_qua_xuat, indent=2)
        cot1.download_button(
            label="Export latest result (JSON)",
            data=du_lieu_json,
            file_name="prediction_result.json",
            mime="application/json",
            use_container_width=True,
        )

    if lich_su:
        bang_lich_su = pd.DataFrame(lich_su)
        cot2.download_button(
            label="Export history (CSV)",
            data=bang_lich_su.to_csv(index=False),
            file_name="prediction_history.csv",
            mime="text/csv",
            use_container_width=True,
        )
