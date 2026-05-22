from functools import lru_cache
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

DUONG_DAN_MO_HINH = Path(__file__).resolve().parents[1] / "Weights" / "best_97.pt"
KICH_THUOC_ANH_YOLO = 640
NGUONG_CONF = 0.25
NGUONG_IOU = 0.45
DO_DAY_VIEN = 5
CO_CHU_NHAN = 18
LE_CHU = 4

LOP_HU_HONG = [
    "dent",
    "scratch",
    "crack",
    "lamp_broken",
    "glass_broken",
    "rust",
]

MAU_LOP = {
    "dent": (0, 120, 255),
    "scratch": (255, 140, 0),
    "crack": (255, 60, 60),
    "lamp_broken": (180, 0, 255),
    "glass_broken": (0, 200, 120),
    "rust": (150, 90, 40),
}


def chuan_hoa_ten_lop(ten_lop):
    # """Chuẩn hóa tên lớp về dạng không dấu cách và chữ thường."""
    return ten_lop.strip().lower().replace("-", "_").replace(" ", "_")


@lru_cache(maxsize=1)
def _tai_mo_hinh_yolo():
    return YOLO(str(DUONG_DAN_MO_HINH))


def phat_hien_hu_hong(danh_sach_anh):
    # """Chạy YOLOv8 và trả về danh sách hư hỏng đã phát hiện."""
    if not danh_sach_anh:
        return []

    mo_hinh = _tai_mo_hinh_yolo()
    danh_sach_phat_hien = []
    ma_hu_hong = 1

    for muc_anh in danh_sach_anh:
        anh = muc_anh["image"]
        chieu_rong, chieu_cao = anh.size
        ket_qua = mo_hinh.predict(
            source=np.array(anh),
            imgsz=KICH_THUOC_ANH_YOLO,
            conf=NGUONG_CONF,
            iou=NGUONG_IOU,
            verbose=False,
        )
        if not ket_qua:
            continue

        ket_qua_anh = ket_qua[0]
        if ket_qua_anh.boxes is None or len(ket_qua_anh.boxes) == 0:
            continue

        ten_lop = ket_qua_anh.names
        for hop in ket_qua_anh.boxes:
            chi_so_lop = int(hop.cls.item())
            do_tin_cay = float(hop.conf.item())
            x1, y1, x2, y2 = [int(round(v)) for v in hop.xyxy[0].tolist()]
            x1 = max(0, min(chieu_rong, x1))
            x2 = max(0, min(chieu_rong, x2))
            y1 = max(0, min(chieu_cao, y1))
            y2 = max(0, min(chieu_cao, y2))

            ten_goc = ten_lop.get(chi_so_lop, str(chi_so_lop)) if isinstance(ten_lop, dict) else str(chi_so_lop)
            ten_lop_chuan = chuan_hoa_ten_lop(str(ten_goc))
            ti_le_dien_tich = 0.0
            if chieu_rong > 0 and chieu_cao > 0:
                dien_tich = max(0, (x2 - x1) * (y2 - y1))
                ti_le_dien_tich = round(dien_tich / (chieu_rong * chieu_cao), 4)

            danh_sach_phat_hien.append(
                {
                    "damage_id": ma_hu_hong,
                    "image_name": muc_anh["name"],
                    "class": ten_lop_chuan,
                    "confidence": round(do_tin_cay, 2),
                    "bbox": [x1, y1, x2, y2],
                    "area_ratio": ti_le_dien_tich,
                }
            )
            ma_hu_hong += 1

    return danh_sach_phat_hien


def ve_bbox_anh(danh_sach_anh, danh_sach_phat_hien):
    # """Vẽ bounding box lên ảnh để hiển thị."""
    anh_da_ve = {}
    
    

    for muc_anh in danh_sach_anh:
        anh = muc_anh["image"].convert("RGB")
        anh_bgr = cv2.cvtColor(np.array(anh), cv2.COLOR_RGB2BGR)
        phat_hien_tren_anh = [d for d in danh_sach_phat_hien if d["image_name"] == muc_anh["name"]]

        for phat_hien in phat_hien_tren_anh:
            x1, y1, x2, y2 = phat_hien["bbox"]
            mau = MAU_LOP.get(phat_hien["class"], (255, 0, 0))
            mau_bgr = (mau[2], mau[1], mau[0])
            cv2.rectangle(anh_bgr, (x1, y1), (x2, y2), mau_bgr, DO_DAY_VIEN)
            nhan = f"{phat_hien['class']} {phat_hien['confidence']:.2f}"

            (chieu_rong_chu, chieu_cao_chu), baseline = cv2.getTextSize(nhan,cv2.FONT_HERSHEY_SIMPLEX,0.4,1,)
            x_nhan = max(0, x1)
            y_nhan = max(0, y1 - chieu_cao_chu - baseline - LE_CHU * 2)

            cv2.rectangle(
                anh_bgr,
                (x_nhan, y_nhan),
                (
                    x_nhan + chieu_rong_chu + LE_CHU * 2,
                    y_nhan + chieu_cao_chu + baseline + LE_CHU * 2,
                ),
                (0, 0, 0),
                -1,
            )
            cv2.putText(anh_bgr,nhan,(x_nhan + LE_CHU, y_nhan + chieu_cao_chu + LE_CHU),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255, 255, 255),1,lineType=cv2.LINE_AA,)

        anh_rgb = cv2.cvtColor(anh_bgr, cv2.COLOR_BGR2RGB)
        anh_da_ve[muc_anh["name"]] = Image.fromarray(anh_rgb)

    return anh_da_ve
