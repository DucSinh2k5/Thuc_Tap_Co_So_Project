from functools import lru_cache
from pathlib import Path

import torch
from torchvision import models
from torchvision.models import ConvNeXt_Tiny_Weights


CAC_MUC_DO = ["minor", "moderate", "severe"]
DIEM_MUC_DO = {"minor": 1, "moderate": 2, "severe": 3}
DUONG_DAN_MO_HINH = Path(__file__).resolve().parents[1] / "Models" / "ConvNeXt.pkl"
THIET_BI = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BIEN_DOI_DANH_GIA = ConvNeXt_Tiny_Weights.DEFAULT.transforms()


def _lay_state_dict(doi_tuong_tai):
    if isinstance(doi_tuong_tai, dict):
        for khoa in ("state_dict", "model_state_dict", "model", "net", "weights"):
            gia_tri = doi_tuong_tai.get(khoa)
            if isinstance(gia_tri, dict):
                return gia_tri
        return doi_tuong_tai
    return None


def _bo_tien_to(state_dict, tien_to):
    if state_dict and all(khoa.startswith(tien_to) for khoa in state_dict.keys()):
        return {khoa[len(tien_to):]: gia_tri for khoa, gia_tri in state_dict.items()}
    return state_dict


def _tao_convnext_tiny():
    mo_hinh = models.convnext_tiny(weights=None)
    so_dac_trung = mo_hinh.classifier[2].in_features
    mo_hinh.classifier[2] = torch.nn.Linear(so_dac_trung, len(CAC_MUC_DO))
    return mo_hinh


@lru_cache(maxsize=1)
def _tai_mo_hinh_muc_do():
    doi_tuong_tai = torch.load(DUONG_DAN_MO_HINH, map_location=THIET_BI)

    if isinstance(doi_tuong_tai, torch.nn.Module):
        mo_hinh = doi_tuong_tai
    else:
        state_dict = _lay_state_dict(doi_tuong_tai)
        state_dict = _bo_tien_to(state_dict, "module.")
        state_dict = _bo_tien_to(state_dict, "model.")
        state_dict = _bo_tien_to(state_dict, "net.")

        mo_hinh = _tao_convnext_tiny()
        mo_hinh.load_state_dict(state_dict, strict=True)

    mo_hinh.to(THIET_BI)
    mo_hinh.eval()
    return mo_hinh


def _tien_xu_ly_anh_full(danh_sach_anh):
    tensor_list = []
    for muc_anh in danh_sach_anh:
        tensor_list.append(BIEN_DOI_DANH_GIA(muc_anh["image"].convert("RGB")))
    return torch.stack(tensor_list, dim=0)


def _lay_logits(dau_ra):
    if isinstance(dau_ra, (tuple, list)):
        return dau_ra[0]
    if isinstance(dau_ra, dict):
        for khoa in ("logits", "output"):
            if khoa in dau_ra:
                return dau_ra[khoa]
        return next(iter(dau_ra.values()))
    return dau_ra


def _du_doan_muc_do(mo_hinh, batch):
    batch = batch.to(THIET_BI)
    with torch.no_grad():
        logits = _lay_logits(mo_hinh(batch))
        xac_suat = torch.softmax(logits, dim=1).detach().cpu()

    chi_so = torch.argmax(xac_suat, dim=1).tolist()
    ket_qua = []
    for idx, probs in zip(chi_so, xac_suat):
        nhan = CAC_MUC_DO[int(idx)]
        ket_qua.append(
            {
                "severity": nhan,
                "severity_score": DIEM_MUC_DO.get(nhan, 1),
                "confidence": round(float(probs[int(idx)].item()), 4),
                "probabilities": {
                    CAC_MUC_DO[i]: round(float(probs[i].item()), 4)
                    for i in range(len(CAC_MUC_DO))
                },
            }
        )
    return ket_qua


def phan_loai_muc_do(danh_sach_anh):
    """Du doan severity truc tiep tu anh full nguoi dung upload."""
    if not danh_sach_anh:
        return []

    mo_hinh = _tai_mo_hinh_muc_do()
    batch = _tien_xu_ly_anh_full(danh_sach_anh)
    ket_qua_du_doan = _du_doan_muc_do(mo_hinh, batch)

    danh_sach_muc_do = []
    for muc_anh, muc_do in zip(danh_sach_anh, ket_qua_du_doan):
        danh_sach_muc_do.append(
            {
                "image_name": muc_anh["name"],
                **muc_do,
            }
        )

    return danh_sach_muc_do


def ghep_chi_tiet_muc_do(danh_sach_muc_do):
    hang = []
    for muc_do in danh_sach_muc_do:
        xac_suat = muc_do.get("probabilities", {})
        hang.append(
            {
                "image_name": muc_do.get("image_name"),
                "severity": muc_do.get("severity"),
                "severity_score": muc_do.get("severity_score"),
                "confidence": f"{muc_do.get('confidence', 0.0):.2%}",
                "minor_prob": f"{xac_suat.get('minor', 0.0):.2%}",
                "moderate_prob": f"{xac_suat.get('moderate', 0.0):.2%}",
                "severe_prob": f"{xac_suat.get('severe', 0.0):.2%}",
            }
        )
    return hang


def tong_hop_muc_do(danh_sach_phat_hien, danh_sach_muc_do):
    tong_hop = {
        "total_damages": len(danh_sach_phat_hien),
        "total_images_classified": len(danh_sach_muc_do),
        "num_dents": 0,
        "num_scratches": 0,
        "num_cracks": 0,
        "max_severity": "none",
        "average_severity_score": 0.0,
    }

    for phat_hien in danh_sach_phat_hien:
        ten_lop = phat_hien.get("class", "")
        if ten_lop == "dent":
            tong_hop["num_dents"] += 1
        if ten_lop == "scratch":
            tong_hop["num_scratches"] += 1
        if ten_lop == "crack":
            tong_hop["num_cracks"] += 1

    if not danh_sach_muc_do:
        return tong_hop

    diem_cao_nhat = 0
    tong_diem = 0
    for muc_do in danh_sach_muc_do:
        diem = muc_do.get("severity_score", 1)
        tong_diem += diem
        if diem > diem_cao_nhat:
            diem_cao_nhat = diem
            tong_hop["max_severity"] = muc_do.get("severity", "minor")

    tong_hop["average_severity_score"] = round(tong_diem / len(danh_sach_muc_do), 2)
    return tong_hop
