import joblib
import pandas as pd

LAKH_INR_SANG_VND = 30_000_000

TRONG_SO_LOP = {
    "scratch": 0.75,
    "dent": 1.0,
    "crack": 1.15,
    "tire_flat": 1.1,
    "glass_broken": 1.25,
    "lamp_broken": 1.35,
}

HE_SO_DIEM_SANG_TI_LE = 0.012
MUC_GIAM_TOI_DA = 0.3

mo_hinh = joblib.load(r"F:\Documents\CODE\TTCS\Thuc_Tap_Co_So_Project\Models\model.pkl")


def _tinh_he_so_dien_tich(ti_le_dien_tich):
    ti_le = max(0.0, float(ti_le_dien_tich or 0.0))
    if ti_le < 0.02:
        return 0.9
    if ti_le < 0.08:
        return 1.0
    if ti_le < 0.16:
        return 1.1
    return 1.2


def _tinh_he_so_so_luong(so_lan_da_gap):
    if so_lan_da_gap <= 0:
        return 1.0
    if so_lan_da_gap == 1:
        return 0.75
    return 0.5


def _tinh_he_so_tin_cay(confidence):
    do_tin_cay = max(0.0, min(float(confidence or 0.0), 1.0))
    return 0.75 + do_tin_cay * 0.25

def _tao_du_lieu_du_doan(thong_tin_xe, mo_hinh):
    if isinstance(thong_tin_xe, pd.DataFrame):
        du_lieu = thong_tin_xe.copy()
    else:
        du_lieu = pd.DataFrame([thong_tin_xe])

    cot_dau_vao = mo_hinh["candidate_features"]
    for cot in cot_dau_vao:
        if cot not in du_lieu.columns:
            du_lieu[cot] = float("nan")
    return du_lieu[cot_dau_vao]


def _du_doan_lakh(mo_hinh, du_lieu):
    bien_doi = mo_hinh["preprocessor"]
    chi_so_chon = mo_hinh["selected_indices"]
    x = bien_doi.transform(du_lieu[mo_hinh["candidate_features"]])
    x_chon = x[:, chi_so_chon]
    du_doan = mo_hinh["model_xgb"].predict(x_chon)
    

    return max(0.0, float(du_doan[0]))


def du_doan_gia_co_ban(thong_tin_xe):
    # Dự đoán giá cơ bản bằng model tabular trong Models/model.pkl.

    # thong_tin_xe phải chứa các feature đã được chuẩn hóa giống pipeline huấn luyện.
    # Input must already be formatted; no cleaning is applied here.
    
    du_lieu = _tao_du_lieu_du_doan(thong_tin_xe, mo_hinh)
    gia_lakh = _du_doan_lakh(mo_hinh, du_lieu)
    gia_vnd = gia_lakh * LAKH_INR_SANG_VND
    return round(gia_vnd / 100_000) * 100_000


def tinh_dieu_chinh_gia(gia_co_ban, danh_sach_phat_hien,danh_sach_muc_do,):
    #Tính toán điều chỉnh giá dựa trên hư hỏng và mức độ.
    if not danh_sach_phat_hien:
        return {
            "base_price": gia_co_ban,
            "deduction_amount": 0.0,
            "final_price": gia_co_ban,
            "damage_score": 0.0,
            "estimated_deduction_rate": 0.0,
            "top_deduction_reason": "No detected damages",
        }

    ban_do_muc_do_theo_anh = {s["image_name"]: s for s in danh_sach_muc_do if s.get("image_name")}
    muc_do_mac_dinh = max(
        danh_sach_muc_do,
        key=lambda muc_do: muc_do.get("severity_score", 1),
        default={"severity_score": 1, "severity": "minor"},
    )
    diem_theo_lop = {}
    dem_theo_lop = {}
    tong_diem = 0.0
    diem_cao_nhat = 0
    muc_do_cao_nhat = "minor"

    for phat_hien in danh_sach_phat_hien:
        muc_do = ban_do_muc_do_theo_anh.get(phat_hien.get("image_name"), muc_do_mac_dinh)
        lop_hu_hong = phat_hien.get("class", "unknown")
        diem_muc_do = muc_do["severity_score"]
        trong_so = TRONG_SO_LOP.get(lop_hu_hong, 1.0)
        he_so_dien_tich = _tinh_he_so_dien_tich(phat_hien.get("area_ratio", 0.0))
        he_so_so_luong = _tinh_he_so_so_luong(dem_theo_lop.get(lop_hu_hong, 0))
        he_so_tin_cay = _tinh_he_so_tin_cay(phat_hien.get("confidence", 1.0))
        diem = diem_muc_do * trong_so * he_so_dien_tich * he_so_so_luong * he_so_tin_cay

        tong_diem += diem
        diem_theo_lop[lop_hu_hong] = diem_theo_lop.get(lop_hu_hong, 0.0) + diem
        dem_theo_lop[lop_hu_hong] = dem_theo_lop.get(lop_hu_hong, 0) + 1

        if diem_muc_do > diem_cao_nhat:
            diem_cao_nhat = diem_muc_do
            muc_do_cao_nhat = muc_do["severity"]

    diem_hu_hong = min(MUC_GIAM_TOI_DA, tong_diem * HE_SO_DIEM_SANG_TI_LE)
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
