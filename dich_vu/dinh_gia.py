import joblib
import pandas as pd
import requests
import xml.etree.ElementTree as ET

TY_GIA_URL = "https://portal.vietcombank.com.vn/Usercontrols/TVPortal.TyGia/pXML.aspx"


def _tai_ty_gia_inr_vnd():
    response = requests.get(
        TY_GIA_URL,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10,
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)

    for item in root.findall("Exrate"):
        if item.attrib.get("CurrencyCode") == "INR":
            ty_gia = (
                item.attrib.get("Transfer")
                or item.attrib.get("Sell")
                or item.attrib.get("Buy")
            )
            if not ty_gia:
                raise ValueError("Missing INR exchange rate in XML response.")
            return float(ty_gia.replace(",", ""))

    raise ValueError("INR not found in XML exchange rates.")


def _lay_lakh_inr_sang_vnd():
    return _tai_ty_gia_inr_vnd() * 100_000


try:
    LAKH_INR_SANG_VND = _lay_lakh_inr_sang_vnd()
except Exception:
    LAKH_INR_SANG_VND = 30_000_000

TRONG_SO_LOP = {
    "dent": 1.0,
    "scratch": 0.85,
    "crack": 1.1,
    "lamp_broken": 1.2,
    "glass_broken": 1.15,
    "rust": 0.9,
}

mo_hinh = joblib.load(r"F:\Documents\CODE\TTCS\Thuc_Tap_Co_So_Project\Models\model.pkl")

def _tao_du_lieu_du_doan(thong_tin_xe, mo_hinh):
    if isinstance(thong_tin_xe, pd.DataFrame):
        du_lieu = thong_tin_xe.copy()
    else:
        du_lieu = pd.DataFrame([thong_tin_xe])

    cot_dau_vao = mo_hinh["candidate_features"]
    thieu_cot = [cot for cot in cot_dau_vao if cot not in du_lieu.columns]
    if thieu_cot:
        raise ValueError(f"Input is missing required features: {thieu_cot}")
    return du_lieu[cot_dau_vao]


def _du_doan_lakh(mo_hinh, du_lieu):
    bien_doi = mo_hinh["preprocessor"]
    chi_so_chon = mo_hinh["selected_indices"]
    x = bien_doi.transform(du_lieu[mo_hinh["candidate_features"]])
    x_chon = x[:, chi_so_chon]

    if "model_xgb" in mo_hinh:
        du_doan = mo_hinh["model_xgb"].predict(x_chon)
    elif "model_rf" in mo_hinh:
        du_doan = mo_hinh["model_rf"].predict(x_chon)
    else:
        du_doan = mo_hinh["model_lr"].predict(mo_hinh["scaler_lr"].transform(x_chon))

    return max(0.0, float(du_doan[0]))


def du_doan_gia_co_ban(thong_tin_xe):
    """Dự đoán giá cơ bản bằng model tabular trong Models/model.pkl.

    thong_tin_xe phải chứa các feature đã được chuẩn hóa giống pipeline huấn luyện.
    Input must already be formatted; no cleaning is applied here.
    """
    du_lieu = _tao_du_lieu_du_doan(thong_tin_xe, mo_hinh)
    gia_lakh = _du_doan_lakh(mo_hinh, du_lieu)
    gia_vnd = gia_lakh * LAKH_INR_SANG_VND
    return round(gia_vnd / 100_000) * 100_000


def tinh_dieu_chinh_gia(
    gia_co_ban,
    danh_sach_phat_hien,
    danh_sach_muc_do,
):
    """Tính toán điều chỉnh giá dựa trên hư hỏng và mức độ."""
    if not danh_sach_phat_hien:
        return {
            "base_price": gia_co_ban,
            "deduction_amount": 0.0,
            "final_price": gia_co_ban,
            "damage_score": 0.0,
            "estimated_deduction_rate": 0.0,
            "top_deduction_reason": "No detected damages",
        }

    ban_do_muc_do = {s["damage_id"]: s for s in danh_sach_muc_do}
    diem_theo_lop = {}
    dem_theo_lop = {}
    tong_diem = 0.0
    diem_cao_nhat = 0
    muc_do_cao_nhat = "minor"

    for phat_hien in danh_sach_phat_hien:
        muc_do = ban_do_muc_do.get(phat_hien["damage_id"], {"severity_score": 1, "severity": "minor"})
        diem_muc_do = muc_do["severity_score"]
        trong_so = TRONG_SO_LOP.get(phat_hien["class"], 1.0)
        ti_le = phat_hien.get("area_ratio", 0.0)
        he_so_dien_tich = 0.2 + min(ti_le, 0.25) * 2.5
        diem = diem_muc_do * trong_so * he_so_dien_tich

        tong_diem += diem
        diem_theo_lop[phat_hien["class"]] = diem_theo_lop.get(phat_hien["class"], 0.0) + diem
        dem_theo_lop[phat_hien["class"]] = dem_theo_lop.get(phat_hien["class"], 0) + 1

        if diem_muc_do > diem_cao_nhat:
            diem_cao_nhat = diem_muc_do
            muc_do_cao_nhat = muc_do["severity"]

    diem_hu_hong = min(0.3, tong_diem * 0.03)
    tien_tru = round((gia_co_ban * diem_hu_hong) / 100_000) * 100_000
    tien_tru = min(tien_tru, gia_co_ban)
    gia_sau = max(gia_co_ban - tien_tru, 0.0)

    lop_noi_bat = max(diem_theo_lop, key=diem_theo_lop.get)
    so_luong = dem_theo_lop.get(lop_noi_bat, 0)
    ly_do = f"{so_luong} {lop_noi_bat} damage(s), max severity {muc_do_cao_nhat}"

    return {
        "base_price": gia_co_ban,
        "deduction_amount": tien_tru,
        "final_price": gia_sau,
        "damage_score": diem_hu_hong,
        "estimated_deduction_rate": diem_hu_hong,
        "top_deduction_reason": ly_do,
    }
