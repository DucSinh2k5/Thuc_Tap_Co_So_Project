from __future__ import annotations

import html
import os
import struct
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "slide_bao_cao_du_an.pptx"

SLIDE_W = 12192000
SLIDE_H = 6858000

NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_P = "http://schemas.openxmlformats.org/presentationml/2006/main"


def emu(inches: float) -> int:
    return int(round(inches * 914400))


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def color(hex_color: str) -> str:
    return hex_color.replace("#", "").upper()


def text_run(text: str, size: int = 24, bold: bool = False, col: str = "#111827") -> str:
    b = ' b="1"' if bold else ""
    return (
        f'<a:r><a:rPr lang="vi-VN" sz="{size * 100}"{b}>'
        f'<a:solidFill><a:srgbClr val="{color(col)}"/></a:solidFill>'
        f"</a:rPr><a:t>{esc(text)}</a:t></a:r>"
    )


def paragraph(text: str, size: int = 24, bold: bool = False, col: str = "#111827",
              bullet: bool = False, align: str | None = None) -> str:
    ppr = ""
    if bullet:
        ppr = '<a:pPr marL="342900" indent="-171450"><a:buChar char="•"/></a:pPr>'
    elif align:
        ppr = f'<a:pPr algn="{align}"/>'
    return f"<a:p>{ppr}{text_run(text, size=size, bold=bold, col=col)}</a:p>"


class SlideBuilder:
    def __init__(self, title: str | None = None, subtitle: str | None = None):
        self.elements: list[str] = []
        self.images: list[tuple[Path, str]] = []
        self.shape_id = 2
        if title:
            self.textbox(0.55, 0.35, 12.2, 0.55, title, size=30, bold=True, col="#0F172A")
        if subtitle:
            self.textbox(0.58, 0.92, 11.9, 0.32, subtitle, size=13, col="#64748B")
        self.footer()

    def next_id(self) -> int:
        self.shape_id += 1
        return self.shape_id

    def footer(self) -> None:
        self.rect(0.0, 7.25, 13.333, 0.03, "#E5E7EB", line=None)
        self.textbox(0.55, 7.29, 7.0, 0.22, "Used Car Price AI • XGBoost + YOLOv8s + ConvNeXt-Tiny", size=8, col="#64748B")

    def textbox(self, x: float, y: float, w: float, h: float, text: str,
                size: int = 20, bold: bool = False, col: str = "#111827",
                fill: str | None = None, line: str | None = None, radius: bool = False,
                align: str | None = None) -> None:
        sid = self.next_id()
        fill_xml = '<a:noFill/>' if not fill else f'<a:solidFill><a:srgbClr val="{color(fill)}"/></a:solidFill>'
        line_xml = '<a:ln><a:noFill/></a:ln>' if not line else (
            f'<a:ln w="12700"><a:solidFill><a:srgbClr val="{color(line)}"/></a:solidFill></a:ln>'
        )
        prst = "roundRect" if radius else "rect"
        paras = "".join(paragraph(part, size=size, bold=bold, col=col, align=align) for part in text.split("\n"))
        self.elements.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="TextBox {sid}"/>'
            f'<p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/>'
            f'<a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'
            f'<a:prstGeom prst="{prst}"><a:avLst/></a:prstGeom>{fill_xml}{line_xml}</p:spPr>'
            f'<p:txBody><a:bodyPr wrap="square" lIns="91440" tIns="45720" rIns="91440" bIns="45720"/>'
            f"<a:lstStyle/>{paras}</p:txBody></p:sp>"
        )

    def bullets(self, x: float, y: float, w: float, h: float, items: list[str],
                size: int = 20, col: str = "#111827") -> None:
        sid = self.next_id()
        paras = "".join(paragraph(item, size=size, col=col, bullet=True) for item in items)
        self.elements.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="Bullets {sid}"/>'
            f'<p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/>'
            f'<a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>'
            f'<p:txBody><a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0"/>'
            f"<a:lstStyle/>{paras}</p:txBody></p:sp>"
        )

    def rect(self, x: float, y: float, w: float, h: float, fill: str,
             line: str | None = "#E5E7EB", radius: bool = False) -> None:
        sid = self.next_id()
        line_xml = '<a:ln><a:noFill/></a:ln>' if line is None else (
            f'<a:ln w="12700"><a:solidFill><a:srgbClr val="{color(line)}"/></a:solidFill></a:ln>'
        )
        prst = "roundRect" if radius else "rect"
        self.elements.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="Shape {sid}"/>'
            f'<p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/>'
            f'<a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'
            f'<a:prstGeom prst="{prst}"><a:avLst/></a:prstGeom>'
            f'<a:solidFill><a:srgbClr val="{color(fill)}"/></a:solidFill>{line_xml}</p:spPr></p:sp>'
        )

    def picture(self, path: Path, x: float, y: float, w: float, h: float) -> None:
        if not path.exists():
            return
        sid = self.next_id()
        rid = f"rIdImg{len(self.images) + 1}"
        self.images.append((path, rid))
        self.elements.append(
            f'<p:pic><p:nvPicPr><p:cNvPr id="{sid}" name="{esc(path.name)}"/>'
            f'<p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>'
            f'<p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>'
            f'<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/>'
            f'<a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>'
        )

    def stat_card(self, x: float, y: float, w: float, label: str, value: str, accent: str) -> None:
        self.rect(x, y, w, 1.05, "#F8FAFC", "#CBD5E1", radius=True)
        self.rect(x, y, 0.08, 1.05, accent, line=None)
        self.textbox(x + 0.18, y + 0.12, w - 0.3, 0.25, label, size=10, col="#64748B")
        self.textbox(x + 0.18, y + 0.42, w - 0.3, 0.45, value, size=22, bold=True, col="#0F172A")

    def table(self, x: float, y: float, widths: list[float], row_h: float,
              rows: list[list[str]], header_fill: str = "#F1F5F9") -> None:
        cy = y
        for r, row in enumerate(rows):
            cx = x
            fill = header_fill if r == 0 else "#FFFFFF"
            for c, cell in enumerate(row):
                self.rect(cx, cy, widths[c], row_h, fill, "#E2E8F0")
                self.textbox(cx + 0.04, cy + 0.07, widths[c] - 0.08, row_h - 0.08, cell,
                             size=11 if r == 0 else 10, bold=(r == 0), col="#0F172A")
                cx += widths[c]
            cy += row_h

    def bar(self, x: float, y: float, w: float, h: float, pct: float, fill: str, label: str) -> None:
        self.rect(x, y, w, h, "#F1F5F9", None, radius=True)
        self.rect(x, y, w * pct, h, fill, None, radius=True)
        self.textbox(x + w + 0.12, y - 0.03, 1.8, h + 0.1, label, size=11, bold=True)

    def xml(self) -> str:
        content = "".join(self.elements)
        return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="{NS_A}" xmlns:r="{NS_R}" xmlns:p="{NS_P}">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{SLIDE_W}" cy="{SLIDE_H}"/><a:chOff x="0" y="0"/><a:chExt cx="{SLIDE_W}" cy="{SLIDE_H}"/></a:xfrm></p:grpSpPr>
      {content}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>'''


def make_slides() -> list[SlideBuilder]:
    slides: list[SlideBuilder] = []

    s = SlideBuilder()
    s.rect(0, 0, 13.333, 7.5, "#F8FAFC", None)
    s.rect(0, 0, 0.22, 7.5, "#EF4444", None)
    s.textbox(0.85, 1.05, 11.4, 1.15, "Hệ thống hỗ trợ định giá xe ô tô cũ\ncó xét đến hư hỏng ngoại thất", size=34, bold=True)
    s.textbox(0.9, 2.45, 9.6, 0.45, "Machine Learning + Computer Vision • XGBoost • YOLOv8s • ConvNeXt-Tiny", size=18, col="#475569")
    s.stat_card(0.9, 3.35, 2.65, "Tabular pricing", "XGBoost", "#2563EB")
    s.stat_card(3.85, 3.35, 2.65, "Damage detection", "YOLOv8s", "#EF4444")
    s.stat_card(6.8, 3.35, 2.65, "Severity", "ConvNeXt", "#7C3AED")
    s.stat_card(9.75, 3.35, 2.65, "Demo", "Streamlit", "#16A34A")
    s.textbox(0.9, 6.15, 10.5, 0.35, "Sinh viên: ................................     GVHD: ................................     Môn: Thực tập cơ sở", size=14, col="#334155")
    slides.append(s)

    s = SlideBuilder("Lý do chọn đề tài", "Bài toán định giá xe cũ cần kết hợp cả thông tin kỹ thuật và tình trạng ngoại thất.")
    s.bullets(0.75, 1.45, 6.3, 3.8, [
        "Giá xe cũ không chỉ phụ thuộc vào hãng, đời xe, số km, nhiên liệu hay hộp số.",
        "Tình trạng ngoại thất như xước, móp, nứt, vỡ đèn cũng ảnh hưởng trực tiếp đến giá trị xe.",
        "Nhiều hệ thống định giá chỉ dùng dữ liệu bảng, chưa khai thác tín hiệu từ ảnh xe.",
        "Dự án xây dựng pipeline thử nghiệm kết hợp dữ liệu bảng và ảnh để hỗ trợ định giá sát thực tế hơn.",
    ], size=20)
    s.rect(7.35, 1.5, 5.15, 3.9, "#FEF2F2", "#FECACA", radius=True)
    s.textbox(7.65, 1.85, 4.5, 0.45, "Ý tưởng chính", size=24, bold=True, col="#991B1B")
    s.bullets(7.65, 2.45, 4.45, 2.55, [
        "XGBoost dự đoán giá cơ sở.",
        "YOLO phát hiện vùng hư hỏng.",
        "CNN phân loại mức độ hư hỏng.",
        "Rule-based layer điều chỉnh giá cuối.",
    ], size=18, col="#7F1D1D")
    slides.append(s)

    s = SlideBuilder("Mục tiêu và phạm vi", "Dự án là một demo đa mô hình, không phải hệ thống định giá thương mại hoàn chỉnh.")
    s.stat_card(0.75, 1.35, 2.85, "Mục tiêu 1", "Base price", "#2563EB")
    s.stat_card(3.9, 1.35, 2.85, "Mục tiêu 2", "Damage detection", "#EF4444")
    s.stat_card(7.05, 1.35, 2.85, "Mục tiêu 3", "Severity", "#7C3AED")
    s.stat_card(10.2, 1.35, 2.85, "Mục tiêu 4", "Adjusted price", "#16A34A")
    s.bullets(0.9, 3.0, 5.9, 2.5, [
        "Đầu vào: thông tin xe dạng bảng và ảnh xe tùy chọn.",
        "Đầu ra: giá cơ sở, mức giảm do hư hỏng và giá sau điều chỉnh.",
        "Ứng dụng demo bằng Streamlit.",
    ], size=20)
    s.rect(7.2, 3.0, 5.2, 2.35, "#F8FAFC", "#CBD5E1", radius=True)
    s.textbox(7.55, 3.25, 4.55, 0.35, "Giới hạn quan trọng", size=21, bold=True)
    s.bullets(7.55, 3.75, 4.35, 1.25, [
        "Chưa có ground truth cho final price after damage.",
        "Adjusted price là damage-aware rule-based adjustment.",
    ], size=17)
    slides.append(s)

    s = SlideBuilder("Kiến trúc tổng thể hệ thống", "Pipeline có hai nhánh chính: dữ liệu bảng và dữ liệu ảnh.")
    y = 1.35
    boxes = [
        (0.65, y, 2.2, "User input", "#DBEAFE"),
        (3.25, y, 2.2, "Tabular data", "#DBEAFE"),
        (5.85, y, 2.2, "XGBoost\nbase price", "#DBEAFE"),
        (8.45, y, 2.2, "Rule-based\nadjustment", "#DCFCE7"),
        (11.05, y, 1.75, "Final price", "#DCFCE7"),
    ]
    for x, yy, w, text, fill in boxes:
        s.rect(x, yy, w, 0.9, fill, "#93C5FD" if fill == "#DBEAFE" else "#86EFAC", radius=True)
        s.textbox(x + 0.08, yy + 0.18, w - 0.16, 0.5, text, size=15, bold=True, align="ctr")
    img_boxes = [
        (3.25, 3.05, "Image optional", "#F3E8FF"),
        (5.85, 2.55, "YOLOv8s\nbbox + damage", "#FEE2E2"),
        (5.85, 3.85, "ConvNeXt-Tiny\nseverity", "#F3E8FF"),
    ]
    for x, yy, text, fill in img_boxes:
        s.rect(x, yy, 2.2, 0.9, fill, "#C084FC" if fill == "#F3E8FF" else "#FCA5A5", radius=True)
        s.textbox(x + 0.08, yy + 0.18, 2.04, 0.5, text, size=15, bold=True, align="ctr")
    for x1, y1, w in [(2.85, y + 0.42, 0.4), (5.45, y + 0.42, 0.4), (8.05, y + 0.42, 0.4), (10.65, y + 0.42, 0.4)]:
        s.rect(x1, y1, w, 0.05, "#64748B", None)
    s.rect(2.85, 3.47, 0.4, 0.05, "#64748B", None)
    s.rect(8.05, 3.0, 0.4, 0.05, "#64748B", None)
    s.rect(8.05, 4.3, 0.4, 0.05, "#64748B", None)
    s.bullets(0.8, 5.3, 11.8, 1.0, [
        "Nếu không upload ảnh: hệ thống bỏ qua YOLO/CNN và final price = base price.",
        "Nếu có ảnh: YOLO và CNN chạy song song, sau đó rule-based layer tính mức giảm giá.",
    ], size=18)
    slides.append(s)

    s = SlideBuilder("Dữ liệu sử dụng", "Dự án dùng thiết kế dữ liệu mô-đun thay vì một dataset multimodal đồng bộ hoàn toàn.")
    rows = [
        ["Nhánh", "Nguồn", "Vai trò"],
        ["Tabular", "Kaggle used-cars-price-prediction", "Ground truth giá xe cũ để train XGBoost"],
        ["YOLO", "Kaggle CarDD_COCO + Roboflow Universe", "Detect 6 lớp hư hỏng ngoại thất"],
        ["CNN", "Kaggle car-damage-detection", "Phân loại ảnh full thành minor/moderate/severe"],
    ]
    s.table(0.7, 1.45, [1.8, 4.6, 5.5], 0.62, rows)
    s.bullets(0.85, 4.2, 11.5, 1.6, [
        "YOLO dataset gốc tải về dạng COCO, được upload lên Roboflow để export định dạng YOLOv8.",
        "merge_Data = khoảng 4000 ảnh gốc + 2307 ảnh bổ sung cho crack, scratch, dent.",
        "Dữ liệu ảnh và dữ liệu bảng không đồng bộ theo từng chiếc xe, nên hệ thống kết hợp ở mức pipeline.",
    ], size=18)
    slides.append(s)

    s = SlideBuilder("Phân bố nhãn YOLO", "Số liệu từ Roboflow sau khi loại bỏ các lớp nhiễu a, e, f.")
    rows = [
        ["Nhãn", "Bounding box", "Tỷ lệ"],
        ["scratch", "3.595", "41,1%"],
        ["dent", "2.543", "29,1%"],
        ["crack", "898", "10,3%"],
        ["lamp_broken", "704", "8,1%"],
        ["glass_broken", "681", "7,8%"],
        ["tire_flat", "319", "3,6%"],
    ]
    s.table(0.75, 1.35, [2.7, 2.1, 1.7], 0.47, rows)
    bars = [("scratch", 3595, "#7C3AED"), ("dent", 2543, "#A3E635"), ("crack", 898, "#0EA5E9"),
            ("lamp_broken", 704, "#14B8A6"), ("glass_broken", 681, "#E91E63"), ("tire_flat", 319, "#F97316")]
    max_v = 3595
    for i, (name, val, col) in enumerate(bars):
        yy = 1.45 + i * 0.62
        s.textbox(7.4, yy - 0.03, 1.65, 0.28, name, size=11, bold=True)
        s.bar(9.0, yy, 2.55, 0.22, val / max_v, col, f"{val:,}".replace(",", "."))
    s.textbox(0.8, 5.45, 11.3, 0.7, "Nhận xét: phân bố không đều, trong đó scratch và dent chiếm tỷ trọng lớn nhất. Đây là lý do cần phân tích theo từng lớp thay vì chỉ nhìn metric trung bình.", size=17, fill="#F8FAFC", line="#E2E8F0", radius=True)
    slides.append(s)

    s = SlideBuilder("XGBoost dự đoán giá cơ sở", "Nhánh tabular học quan hệ giữa thuộc tính xe và giá thị trường.")
    s.bullets(0.8, 1.35, 5.8, 2.8, [
        "Input: hãng xe, model, năm sản xuất, số km, nhiên liệu, hộp số, số chủ, dung tích, công suất...",
        "Tiền xử lý: làm sạch dữ liệu, chuẩn hóa feature, encode biến phân loại, tạo feature như tuổi xe và km/năm.",
        "So sánh baseline, Random Forest và XGBoost; chọn XGBoost làm mô hình chính.",
        "Checkpoint tích hợp: Models/model.pkl.",
    ], size=18)
    s.stat_card(7.1, 1.45, 2.4, "Test R²", "≈ 0,9367", "#2563EB")
    s.stat_card(9.85, 1.45, 2.4, "RMSE", "≈ 1,4313", "#2563EB")
    s.stat_card(7.1, 2.85, 2.4, "MAE", "≈ 0,8914", "#2563EB")
    s.stat_card(9.85, 2.85, 2.4, "Mô hình", "XGBoost", "#2563EB")
    s.textbox(7.2, 4.45, 4.8, 0.8, "Giá XGBoost là base price. Khi không upload ảnh, hệ thống dùng base price làm giá cuối cùng.", size=17, fill="#EFF6FF", line="#BFDBFE", radius=True)
    slides.append(s)

    s = SlideBuilder("YOLOv8s phát hiện hư hỏng", "Mô hình detection chính: YOLOv8s train trên merge_Data.")
    s.picture(ROOT / "Quan_sat" / "yolo_car_report" / "val_batch0_pred.jpg", 0.65, 1.25, 5.7, 3.2)
    s.bullets(6.65, 1.35, 5.8, 2.45, [
        "6 lớp: crack, dent, glass_broken, lamp_broken, scratch, tire_flat.",
        "Dataset cuối: merge_Data, tập trung bổ sung crack/scratch/dent.",
        "Cấu hình chính: epochs=100, batch=16, imgsz=640.",
        "Checkpoint tích hợp: Models/best.pt.",
    ], size=18)
    s.stat_card(6.85, 4.35, 2.35, "Precision", "0,801", "#EF4444")
    s.stat_card(9.45, 4.35, 2.35, "Recall", "0,682", "#EF4444")
    s.stat_card(6.85, 5.65, 2.35, "mAP@0.5", "0,726", "#EF4444")
    s.stat_card(9.45, 5.65, 2.35, "mAP@0.5:0.95", "0,568", "#EF4444")
    slides.append(s)

    s = SlideBuilder("Data augmentation khi train YOLO", "Ultralytics áp dụng augmentation online trong model.train().")
    rows = [
        ["Tham số", "Giá trị", "Ý nghĩa"],
        ["hsv_h/s/v", "0.015 / 0.7 / 0.4", "Thay đổi màu, bão hòa, độ sáng"],
        ["translate", "0.1", "Dịch ảnh theo không gian"],
        ["scale", "0.5", "Mô phỏng nhiều khoảng cách chụp"],
        ["fliplr", "0.5", "Lật ngang với xác suất 50%"],
        ["mosaic", "1.0", "Ghép ảnh, tăng ngữ cảnh và vật thể nhỏ"],
        ["close_mosaic", "10", "Tắt mosaic ở 10 epoch cuối"],
    ]
    s.table(0.75, 1.25, [2.0, 2.4, 6.7], 0.55, rows)
    s.textbox(0.85, 5.55, 11.2, 0.65, "Các augmentation như mixup, cutmix, copy_paste đang tắt trong run hiện tại. Như vậy dự án vừa mở rộng dữ liệu thật, vừa dùng augmentation mặc định để tăng khả năng tổng quát hóa.", size=17, fill="#F8FAFC", line="#E2E8F0", radius=True)
    slides.append(s)

    s = SlideBuilder("CNN severity classification", "Mô hình chính hiện tại: ConvNeXt-Tiny, nhận ảnh full người dùng upload.")
    rows = [
        ["Split", "01-minor", "02-moderate", "03-severe", "Tổng"],
        ["training", "1.076", "924", "945", "2.945"],
        ["validation", "82", "157", "151", "390"],
    ]
    s.table(0.75, 1.3, [1.7, 2.05, 2.15, 2.05, 1.7], 0.58, rows)
    s.bullets(0.85, 3.2, 5.8, 2.1, [
        "Đầu vào CNN là ảnh full, không phải crop từ YOLO.",
        "Các lớp thực tế: 01-minor, 02-moderate, 03-severe.",
        "Train theo transfer learning 2 phase: head trước, block cuối + head sau.",
        "Checkpoint tích hợp: Models/ConvNeXt.pkl.",
    ], size=18)
    models = [
        ("ResNet18", 0.6490, "#94A3B8"),
        ("EfficientNet-B0", 0.6817, "#94A3B8"),
        ("EfficientNet-B2", 0.6954, "#94A3B8"),
        ("ResNet50", 0.7043, "#94A3B8"),
        ("ConvNeXt-Tiny", 0.7084, "#7C3AED"),
    ]
    for i, (name, val, col) in enumerate(models):
        yy = 3.0 + i * 0.55
        s.textbox(7.25, yy - 0.03, 1.95, 0.28, name, size=11, bold=name == "ConvNeXt-Tiny")
        s.bar(9.2, yy, 2.4, 0.22, val / 0.75, col, f"F1 {val:.4f}".replace(".", ","))
    slides.append(s)

    s = SlideBuilder("Rule-based adjustment", "Tầng cuối kết hợp giá cơ sở và tín hiệu hư hỏng từ ảnh.")
    s.bullets(0.8, 1.35, 5.8, 2.8, [
        "Input: base price từ XGBoost.",
        "YOLO cung cấp số lượng, loại và diện tích vùng hư hỏng.",
        "CNN cung cấp mức severity tổng quát: minor/moderate/severe.",
        "Rule-based layer tính damage deduction và final adjusted price.",
    ], size=18)
    s.rect(7.0, 1.5, 5.1, 3.15, "#F8FAFC", "#CBD5E1", radius=True)
    s.textbox(7.3, 1.85, 4.45, 0.45, "Công thức diễn giải", size=22, bold=True)
    s.textbox(7.35, 2.55, 4.35, 0.45, "Final Price = Base Price - Damage Deduction", size=18, bold=True, col="#0F172A")
    s.textbox(7.35, 3.3, 4.25, 0.75, "Vì chưa có ground truth giá sau hư hỏng, đây là cơ chế adjustment có giải thích, không phải supervised final-price predictor.", size=15, col="#475569")
    slides.append(s)

    s = SlideBuilder("Giao diện demo Streamlit", "Người dùng nhập thông tin xe và có thể upload ảnh hoặc bỏ qua ảnh.")
    s.bullets(0.8, 1.35, 5.6, 2.4, [
        "Form nhập thông tin: hãng, model, năm, số km, nhiên liệu, hộp số, số chủ...",
        "Upload ảnh là tùy chọn.",
        "Nếu không upload ảnh: kết quả là base price từ XGBoost.",
        "Nếu upload ảnh: chạy thêm YOLO + ConvNeXt-Tiny và hiển thị adjusted price.",
    ], size=18)
    s.rect(7.0, 1.25, 5.25, 4.8, "#F8FAFC", "#CBD5E1", radius=True)
    s.textbox(7.35, 1.55, 4.6, 0.35, "Luồng demo", size=22, bold=True)
    flow = ["Nhập dữ liệu xe", "Upload ảnh optional", "Analyze Car", "Base price / Damage deduction", "Final adjusted price"]
    for i, step in enumerate(flow):
        yy = 2.15 + i * 0.68
        s.rect(7.55, yy, 3.9, 0.42, "#FFFFFF", "#CBD5E1", radius=True)
        s.textbox(7.7, yy + 0.07, 3.6, 0.22, step, size=13, bold=True, align="ctr")
        if i < len(flow) - 1:
            s.rect(9.45, yy + 0.46, 0.04, 0.22, "#94A3B8", None)
    slides.append(s)

    s = SlideBuilder("Kết quả thực nghiệm chính", "Tách rõ ba nhánh: tabular, detection và classification.")
    rows = [
        ["Mô-đun", "Mô hình chính", "Kết quả / vai trò"],
        ["Tabular", "XGBoost", "R² test ≈ 0,9367; dùng làm base price"],
        ["YOLO", "YOLOv8s merge_Data", "mAP@0.5 ≈ 0,726; phát hiện vùng hư hỏng"],
        ["CNN", "ConvNeXt-Tiny", "Test acc ≈ 0,7077; macro F1 ≈ 0,7084"],
        ["Adjustment", "Rule-based", "Tính mức giảm giá có giải thích"],
    ]
    s.table(0.75, 1.35, [2.0, 3.1, 6.7], 0.65, rows)
    s.textbox(0.9, 5.35, 11.4, 0.72, "Nhận xét: pipeline hoạt động theo hướng mô-đun, dễ kiểm tra từng thành phần. Điểm quan trọng là không đánh đồng adjusted price với ground truth giá sau hư hỏng.", size=17, fill="#F8FAFC", line="#E2E8F0", radius=True)
    slides.append(s)

    s = SlideBuilder("Hạn chế hiện tại", "Các hạn chế này nên được trình bày thẳng thắn vì chúng định hướng phần phát triển tiếp theo.")
    s.bullets(0.8, 1.35, 11.5, 4.2, [
        "Chưa có dataset multimodal đồng bộ theo từng xe gồm dữ liệu bảng, ảnh và giá sau hư hỏng.",
        "Chưa có ground truth cho final price after damage, nên tầng cuối vẫn là rule-based.",
        "Dữ liệu ảnh có thể có label noise, đặc biệt với vết xước/móp/nứt mờ.",
        "Phân bố nhãn YOLO không đều; scratch và dent chiếm tỷ trọng lớn.",
        "Severity classification còn khó vì nhãn moderate có tính chủ quan và dễ nhập nhằng.",
    ], size=20)
    slides.append(s)

    s = SlideBuilder("Hướng phát triển", "Ưu tiên cải thiện dữ liệu và đánh giá thực tế trước khi mở rộng mô hình.")
    s.bullets(0.8, 1.35, 11.5, 4.4, [
        "Thu thập dataset đồng bộ hơn: thông tin xe, ảnh hư hỏng và giá giao dịch/giá sau sửa chữa.",
        "Audit lại nhãn ảnh, đặc biệt các lớp crack, scratch, dent và nhãn severity moderate.",
        "Bổ sung ảnh từ nhiều góc chụp, điều kiện ánh sáng, màu xe và mức độ hư hỏng khác nhau.",
        "Thử nghiệm calibration cho confidence của YOLO/CNN trước khi đưa vào rule-based adjustment.",
        "Đóng gói demo ổn định hơn, thêm test script và tài liệu hướng dẫn chạy.",
    ], size=20)
    slides.append(s)

    s = SlideBuilder("Kết luận", "Dự án đã xây dựng được pipeline demo hoàn chỉnh cho bài toán định giá xe cũ có xét đến ảnh hư hỏng.")
    s.rect(0.8, 1.4, 11.7, 3.9, "#F8FAFC", "#CBD5E1", radius=True)
    s.bullets(1.15, 1.75, 10.9, 2.85, [
        "XGBoost dự đoán giá cơ sở từ dữ liệu bảng.",
        "YOLOv8s phát hiện các vùng hư hỏng ngoại thất.",
        "ConvNeXt-Tiny phân loại mức độ hư hỏng từ ảnh full.",
        "Rule-based layer tạo giá sau điều chỉnh và giải thích mức giảm.",
        "Streamlit demo cho phép thử nghiệm luồng end-to-end.",
    ], size=22)
    s.textbox(0.95, 5.75, 11.4, 0.6, "Thông điệp chính: hệ thống chưa thay thế định giá chuyên nghiệp, nhưng chứng minh được hướng kết hợp ML tabular và Computer Vision để hỗ trợ định giá xe cũ.", size=18, bold=True, col="#0F172A", fill="#DCFCE7", line="#86EFAC", radius=True)
    slides.append(s)

    return slides


def content_types(slide_count: int, media_exts: list[str]) -> str:
    defaults = [
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
    ]
    for ext in sorted(set(media_exts)):
        ctype = "image/jpeg" if ext in {"jpg", "jpeg"} else f"image/{ext}"
        defaults.append(f'<Default Extension="{ext}" ContentType="{ctype}"/>')
    overrides = [
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
    ]
    for i in range(1, slide_count + 1):
        overrides.append(f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>')
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
{''.join(defaults)}
{''.join(overrides)}
</Types>'''


def rels_root() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''


def presentation_xml(slide_count: int) -> str:
    sld_ids = "".join(
        f'<p:sldId id="{255 + i}" r:id="rId{i + 1}"/>' for i in range(1, slide_count + 1)
    )
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="{NS_A}" xmlns:r="{NS_R}" xmlns:p="{NS_P}">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst>{sld_ids}</p:sldIdLst>
  <p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>'''


def presentation_rels(slide_count: int) -> str:
    rels = ['<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>']
    for i in range(1, slide_count + 1):
        rels.append(f'<Relationship Id="rId{i + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>')
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
{''.join(rels)}
</Relationships>'''


def slide_rels(slide: SlideBuilder, media_map: dict[Path, str]) -> str:
    rels = ['<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>']
    for path, rid in slide.images:
        target = media_map[path]
        rels.append(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{target}"/>')
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
{''.join(rels)}
</Relationships>'''


def slide_master_xml() -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="{NS_A}" xmlns:r="{NS_R}" xmlns:p="{NS_P}">
  <p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill><a:effectLst/></p:bgPr></p:bg>
    <p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
    <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{SLIDE_W}" cy="{SLIDE_H}"/><a:chOff x="0" y="0"/><a:chExt cx="{SLIDE_W}" cy="{SLIDE_H}"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>'''


def slide_master_rels() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>'''


def slide_layout_xml() -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="{NS_A}" xmlns:r="{NS_R}" xmlns:p="{NS_P}" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
  <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{SLIDE_W}" cy="{SLIDE_H}"/><a:chOff x="0" y="0"/><a:chExt cx="{SLIDE_W}" cy="{SLIDE_H}"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>'''


def slide_layout_rels() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>'''


def theme_xml() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Office Theme">
  <a:themeElements>
    <a:clrScheme name="Codex"><a:dk1><a:srgbClr val="111827"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="1F2937"/></a:dk2><a:lt2><a:srgbClr val="F8FAFC"/></a:lt2><a:accent1><a:srgbClr val="2563EB"/></a:accent1><a:accent2><a:srgbClr val="EF4444"/></a:accent2><a:accent3><a:srgbClr val="16A34A"/></a:accent3><a:accent4><a:srgbClr val="7C3AED"/></a:accent4><a:accent5><a:srgbClr val="F59E0B"/></a:accent5><a:accent6><a:srgbClr val="0EA5E9"/></a:accent6><a:hlink><a:srgbClr val="2563EB"/></a:hlink><a:folHlink><a:srgbClr val="7C3AED"/></a:folHlink></a:clrScheme>
    <a:fontScheme name="Office"><a:majorFont><a:latin typeface="Arial"/></a:majorFont><a:minorFont><a:latin typeface="Arial"/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="Office"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>
  </a:themeElements>
</a:theme>'''


def app_xml(slide_count: int) -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application><PresentationFormat>Widescreen</PresentationFormat><Slides>{slide_count}</Slides><Notes>0</Notes><HiddenSlides>0</HiddenSlides>
</Properties>'''


def core_xml() -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Slide báo cáo dự án định giá xe cũ</dc:title><dc:creator>Codex</dc:creator><cp:lastModifiedBy>Codex</cp:lastModifiedBy><dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>'''


def build_pptx() -> None:
    slides = make_slides()
    image_paths: list[Path] = []
    for s in slides:
        for path, _ in s.images:
            if path not in image_paths:
                image_paths.append(path)
    media_map = {}
    media_exts = []
    for idx, path in enumerate(image_paths, start=1):
        ext = path.suffix.lower().lstrip(".")
        if ext == "jpeg":
            ext = "jpg"
        media_name = f"image{idx}.{ext}"
        media_map[path] = media_name
        media_exts.append(ext)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types(len(slides), media_exts))
        z.writestr("_rels/.rels", rels_root())
        z.writestr("docProps/app.xml", app_xml(len(slides)))
        z.writestr("docProps/core.xml", core_xml())
        z.writestr("ppt/presentation.xml", presentation_xml(len(slides)))
        z.writestr("ppt/_rels/presentation.xml.rels", presentation_rels(len(slides)))
        z.writestr("ppt/slideMasters/slideMaster1.xml", slide_master_xml())
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", slide_master_rels())
        z.writestr("ppt/slideLayouts/slideLayout1.xml", slide_layout_xml())
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", slide_layout_rels())
        z.writestr("ppt/theme/theme1.xml", theme_xml())
        for i, slide in enumerate(slides, start=1):
            z.writestr(f"ppt/slides/slide{i}.xml", slide.xml())
            z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", slide_rels(slide, media_map))
        for path, media_name in media_map.items():
            z.write(path, f"ppt/media/{media_name}")

    print(OUT)


if __name__ == "__main__":
    build_pptx()
