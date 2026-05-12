from functools import lru_cache
from pathlib import Path

from PIL import Image
import torch
from torchvision import models
from torchvision.models import ResNet18_Weights

CAC_MUC_DO = ["minor", "moderate", "severe"]
DIEM_MUC_DO = {"minor": 1, "moderate": 2, "severe": 3}
DUONG_DAN_MO_HINH = r"F:\Documents\CODE\TTCS\Thuc_Tap_Co_So_Project\Models\cnn_car.pkl"
THIET_BI = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BIEN_DOI_DANH_GIA = ResNet18_Weights.DEFAULT.transforms()


@lru_cache(maxsize=1)
def _tai_mo_hinh_muc_do():
    if not Path(DUONG_DAN_MO_HINH).exists():
        raise FileNotFoundError(f"Missing severity model at {DUONG_DAN_MO_HINH}")
    try:
        doi_tuong_tai = torch.load(DUONG_DAN_MO_HINH, map_location=THIET_BI)
    except Exception:
        doi_tuong_tai = None

    if isinstance(doi_tuong_tai, torch.nn.Module):
        mo_hinh = doi_tuong_tai
    elif isinstance(doi_tuong_tai, dict):
        trang_thai = (
            doi_tuong_tai.get("model_state_dict")
            or doi_tuong_tai.get("state_dict")
            or doi_tuong_tai
        )
        mo_hinh = models.resnet18(weights=None)
        mo_hinh.fc = torch.nn.Linear(mo_hinh.fc.in_features, len(CAC_MUC_DO))
        mo_hinh.load_state_dict(trang_thai, strict=True)
    else:
        try:
            mo_hinh = torch.jit.load(str(DUONG_DAN_MO_HINH), map_location=THIET_BI)
        except Exception as exc:
            raise RuntimeError(
                "Không thể tải cnn_car.pkl bằng torch. Hãy export torchscript hoặc lưu state_dict."
            ) from exc

    mo_hinh.to(THIET_BI)
    mo_hinh.eval()
    return mo_hinh


def _cat_theo_bbox(phat_hien, ban_do_anh):
    anh = ban_do_anh.get(phat_hien.get("image_name", ""))
    if anh is None:
        return Image.new("RGB", (224, 224), (0, 0, 0))

    x1, y1, x2, y2 = phat_hien.get("bbox", [0, 0, anh.width, anh.height])
    x1 = max(0, min(anh.width - 1, int(x1)))
    y1 = max(0, min(anh.height - 1, int(y1)))
    x2 = max(1, min(anh.width, int(x2)))
    y2 = max(1, min(anh.height, int(y2)))

    if x2 <= x1 or y2 <= y1:
        cat = anh
    else:
        cat = anh.crop((x1, y1, x2, y2))

    return cat.convert("RGB")


def _tien_xu_ly_cat(danh_sach_cat):
    tensor_list = []
    for cat in danh_sach_cat:
        tensor_list.append(BIEN_DOI_DANH_GIA(cat))
    return torch.stack(tensor_list, dim=0)


def _du_doan_nhan(mo_hinh, batch):
    batch = batch.to(THIET_BI)
    with torch.no_grad():
        dau_ra = mo_hinh(batch)

    if isinstance(dau_ra, (tuple, list)):
        dau_ra = dau_ra[0]
    if isinstance(dau_ra, dict):
        dau_ra = dau_ra.get("logits") or dau_ra.get("output") or next(iter(dau_ra.values()))

    if not isinstance(dau_ra, torch.Tensor):
        raise RuntimeError("Đầu ra mô hình mức độ không phải tensor.")

    chi_so = torch.argmax(dau_ra, dim=1).detach().cpu().tolist()
    return [CAC_MUC_DO[int(idx)] for idx in chi_so]


def phan_loai_muc_do(danh_sach_phat_hien, danh_sach_anh=None):
    """Dự đoán mức độ hư hỏng bằng mô hình cnn_car.pkl."""
    if not danh_sach_phat_hien:
        return []

    if not danh_sach_anh:
        raise RuntimeError("Mô hình mức độ cần ảnh để cắt theo bbox.")

    mo_hinh = _tai_mo_hinh_muc_do()
    ban_do_anh = {item["name"]: item["image"] for item in danh_sach_anh}
    danh_sach_cat = [_cat_theo_bbox(phat_hien, ban_do_anh) for phat_hien in danh_sach_phat_hien]
    batch = _tien_xu_ly_cat(danh_sach_cat)
    nhan = _du_doan_nhan(mo_hinh, batch)

    danh_sach_muc_do = []
    for phat_hien, nhan_muc_do in zip(danh_sach_phat_hien, nhan):
        danh_sach_muc_do.append(
            {
                "damage_id": phat_hien["damage_id"],
                "severity": nhan_muc_do,
                "severity_score": DIEM_MUC_DO.get(nhan_muc_do, 1),
            }
        )

    return danh_sach_muc_do


def ghep_chi_tiet_muc_do(danh_sach_phat_hien, danh_sach_muc_do):
    """Gộp phát hiện và mức độ thành từng dòng chi tiết."""
    ban_do_muc_do = {s["damage_id"]: s for s in danh_sach_muc_do}
    hang = []

    for phat_hien in danh_sach_phat_hien:
        muc_do = ban_do_muc_do.get(phat_hien["damage_id"], {"severity": "minor", "severity_score": 1})
        hang.append(
            {
                "damage_id": phat_hien["damage_id"],
                "image_name": phat_hien["image_name"],
                "class": phat_hien["class"],
                "confidence": phat_hien["confidence"],
                "area_ratio": phat_hien.get("area_ratio", 0.0),
                "severity": muc_do["severity"],
                "severity_score": muc_do["severity_score"],
            }
        )

    return hang


def tong_hop_muc_do(danh_sach_phat_hien, danh_sach_muc_do):
    """Tóm tắt kết quả mức độ để hiển thị."""
    tong_hop = {
        "total_damages": len(danh_sach_phat_hien),
        "num_dents": 0,
        "num_scratches": 0,
        "num_cracks": 0,
        "max_severity": "none",
        "average_severity_score": 0.0,
    }

    if not danh_sach_phat_hien:
        return tong_hop

    ban_do_muc_do = {s["damage_id"]: s for s in danh_sach_muc_do}
    diem_cao_nhat = 0
    tong_diem = 0

    for phat_hien in danh_sach_phat_hien:
        ten_lop = phat_hien.get("class", "")
        if ten_lop == "dent":
            tong_hop["num_dents"] += 1
        if ten_lop == "scratch":
            tong_hop["num_scratches"] += 1
        if ten_lop == "crack":
            tong_hop["num_cracks"] += 1

        muc_do = ban_do_muc_do.get(phat_hien["damage_id"], {"severity": "minor", "severity_score": 1})
        diem = muc_do["severity_score"]
        tong_diem += diem
        if diem > diem_cao_nhat:
            diem_cao_nhat = diem
            tong_hop["max_severity"] = muc_do["severity"]

    tong_hop["average_severity_score"] = round(tong_diem / len(danh_sach_phat_hien), 2)
    return tong_hop
