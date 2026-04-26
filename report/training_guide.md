# Training Guide — Used Car Price Prediction Project

## 1. Purpose

Tài liệu này hướng dẫn cách huấn luyện, đánh giá và chọn mô hình cho dự án dự đoán giá xe ô tô cũ với pipeline đa mô hình gồm:

- **YOLOv8** để phát hiện hư hỏng trên ảnh xe.
- **CNN** để phân loại mức độ nghiêm trọng của vùng hư hỏng thành `minor`, `moderate`, `severe`.
- **XGBoost** để dự đoán **base price** từ dữ liệu bảng.
- **Rule-based adjustment layer** để tạo **damage-aware adjusted price** từ base price và thông tin damage trên ảnh.

Mục tiêu của file này là trả lời các câu hỏi sau:

- Muốn train lại hệ thống thì cần chuẩn bị gì.
- Nên theo dõi metric nào cho từng mô hình.
- Nên chọn checkpoint nào làm baseline hiện tại.
- Khi model không cải thiện nhiều thì nên ưu tiên hướng cải thiện nào.

---

## 2. Project Training Pipeline

### 2.1 YOLOv8 — Damage Detection

- **Input:** ảnh xe.
- **Output:** bounding boxes và class hư hỏng.
- **Vai trò:** tạo các feature như số lượng hư hỏng, loại hư hỏng, diện tích vùng hỏng để đưa sang downstream modules.

### 2.2 CNN — Severity Classification

- **Input:** crop vùng damage hoặc ảnh đã được crop sẵn theo vùng hỏng.
- **Output:** `minor`, `moderate`, `severe`.
- **Vai trò:** lượng hóa mức độ nghiêm trọng của hư hỏng để hỗ trợ bước điều chỉnh giá.

### 2.3 XGBoost — Base Price Prediction

- **Input:** dữ liệu bảng của xe cũ.
- **Output:** giá cơ bản của xe.
- **Vai trò:** dự đoán **base market price** trước khi điều chỉnh theo damage nhìn thấy trên ảnh.

### 2.4 Rule-Based Adjustment

- **Input:** base price từ XGBoost, damage features từ YOLO, severity từ CNN.
- **Output:** adjusted price.
- **Lưu ý:** hiện tại chưa có ground-truth chắc chắn cho final price after damage, nên module này được hiểu là **price adjustment estimator**, không phải supervised final-price predictor.

---

## 3. Datasets

### 3.1 YOLO Dataset

Dataset detection chính của dự án hiện tại là bộ dữ liệu theo format YOLO với 6 class:

- `crack`
- `dent`
- `glass shatter`
- `lamp broken`
- `scratch`
- `tire flat`

### Thư mục tham khảo

```text
coco_damage_car_yolov8/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

### Quy tắc dùng split

- `train`: dùng để học tham số.
- `val`: dùng để theo dõi `Precision`, `Recall`, `mAP50`, `mAP50-95` trong quá trình tune.
- `test`: chỉ dùng để đánh giá cuối sau khi đã chốt mô hình và cấu hình.

### Lưu ý quan trọng

Trong các run YOLO của dự án, metric được in sau mỗi epoch là metric trên **tập validation**, không phải test set. Vì vậy:

- log train phản ánh kết quả trên `val`.
- muốn có kết quả cuối trên `test`, cần gọi evaluate riêng bằng `model.val(split="test")`.

### 3.2 CNN Severity Dataset

Dataset severity classification gồm 3 lớp:

- `minor`
- `moderate`
- `severe`

Khuyến nghị:

- crop càng sát vùng damage càng tốt.
- kiểm tra chất lượng crop trước khi train.
- nếu crop lấy từ YOLO thì chất lượng detection sẽ ảnh hưởng trực tiếp đến độ chính xác của CNN.

### 3.3 XGBoost Dataset

Dataset tabular dùng để dự đoán giá xe từ các feature có cấu trúc như:

- `year`
- `num_seats`
- `km_driven`
- `fuel`
- `transmission`
- `brand`
- `model`

Target hiện tại là **base price** chứ không phải final price after damage.

---

## 4. Environment Setup

Khuyến nghị môi trường:

- Python 3.10+
- PyTorch
- Ultralytics
- scikit-learn
- xgboost
- pandas
- numpy
- matplotlib
- opencv-python

### Hardware thực tế đã dùng

Một số run baseline YOLO được train trên **Tesla T4 ~15GB VRAM**. Với môi trường này, cấu hình ổn định nhất hiện tại là:

- `imgsz = 640`
- `batch = 16`

---

## 5. Current YOLO Baseline and Final Model Selection

### 5.1 Các thí nghiệm đã thực hiện

Dự án đã thử các hướng sau cho bài toán damage detection:

1. Train nhiều biến thể mô hình trên dataset ban đầu, gồm:
   - `YOLOv8n`
   - `YOLOv8s`
   - `YOLOv8m`
2. Thử bổ sung thêm dữ liệu ảnh tập trung cho các lớp khó `dent`, `scratch`, `crack` vào tập train.
3. Thực hiện **hai lần** mở rộng dữ liệu theo hướng trên rồi train lại.

### 5.2 Kết luận từ các lần mở rộng dữ liệu

Mặc dù đã bổ sung thêm ảnh cho các lớp khó, các lần train lại sau đó **không cho kết quả tốt hơn**. Kết quả mAP tổng thể thấp hơn so với mô hình tốt nhất trên dataset ban đầu.

Điều này dẫn tới quyết định hiện tại:

- **không dùng checkpoint từ các lần thêm dữ liệu nói trên làm model chính**, vì hiệu năng tổng thể không vượt baseline.
- **giữ checkpoint tốt nhất từ dataset ban đầu** làm mô hình YOLO chính của dự án.

### 5.3 Checkpoint YOLO được chọn hiện tại

Checkpoint YOLO chính hiện tại là `best.pt` thu được từ **dataset ban đầu** với cấu hình:

- `model = YOLOv8s`
- `epochs = 100`
- `batch = 16`
- `imgsz = 640`

Kết quả được ghi nhận để chọn checkpoint này:

- **mAP50 trên test ≈ 0.725 (72.5%)**

### 5.4 Nguyên tắc chọn baseline

`YOLOv8s` trên dataset ban đầu được giữ làm baseline vì:

- cho chất lượng tổng thể tốt hơn các lần mở rộng dữ liệu sau đó.
- cân bằng tốt giữa kích thước model, tốc độ suy luận và accuracy.
- phù hợp nhất để tích hợp vào pipeline hiện tại.

---

## 6. Recommended YOLO Training Procedure

### 6.1 Khi nào train YOLO lại

Chỉ nên train lại YOLO khi có một trong các lý do sau:

- có dữ liệu mới đã được kiểm tra chất lượng.
- đã audit và sửa nhãn cho các lớp khó.
- cần benchmark lại sau khi thay đổi chiến lược annotation hoặc split dữ liệu.

### 6.2 Cấu hình baseline khuyến nghị

```python
from ultralytics import YOLO
import torch

model = YOLO("yolov8s.pt")

results = model.train(
    data="/path/to/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    project="runs/detect",
    name="yolov8s_baseline",
)
```

### 6.3 Đánh giá trên test set sau khi train

```python
from ultralytics import YOLO
import torch

model = YOLO("runs/detect/yolov8s_baseline/weights/best.pt")

metrics = model.val(
    data="/path/to/data.yaml",
    split="test",
    imgsz=640,
    batch=16,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
)

print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
```

### 6.4 Khi nào nên fine-tune `best.pt`

Fine-tune từ `best.pt` chỉ nên làm khi:

- đã có dữ liệu mới chất lượng tốt.
- hoặc đã audit và sửa nhãn.
- hoặc muốn kiểm tra tác động của dữ liệu khó mới thêm vào.

Không nên fine-tune lặp lại quá nhiều trên cùng một bộ dữ liệu cũ nếu metric không cải thiện rõ ràng.

---

## 7. Evaluation Strategy

### 7.1 YOLOv8

Metric chính:

- Precision
- Recall
- mAP50
- mAP50-95

Nguyên tắc:

- dùng `val` để theo dõi trong quá trình tune.
- dùng `test` để xác nhận checkpoint cuối.
- không dùng `test` để chỉnh hyperparameter nhiều lần.

### 7.2 CNN

Metric chính:

- Accuracy
- Macro F1
- Precision / Recall theo lớp
- Confusion Matrix

Không nên chọn mô hình chỉ theo Accuracy vì `minor` và `moderate` có thể chồng lấn.

### 7.3 XGBoost

Metric chính:

- MAE
- RMSE
- MAPE
- R²

Target hiện tại là **base price**, không phải final price after damage.

---

## 8. Data-Centric Improvement Strategy

Khi các biến thể mô hình cho kết quả gần nhau hoặc khi fine-tune không cải thiện đáng kể, nên ưu tiên **cải thiện dữ liệu** hơn là tiếp tục vặn hyperparameter.

### 8.1 Các lớp cần ưu tiên

- `crack`
- `scratch`
- `dent`

### 8.2 Hướng cải thiện dữ liệu khuyến nghị

1. **Audit nhãn** cho `crack`, `scratch`, `dent`.
2. **Thêm dữ liệu khó có chủ đích**, không thêm ngẫu nhiên.
3. **Bổ sung hard negatives**, ví dụ:
   - phản xạ ánh sáng
   - đường viền thân xe
   - vệt bẩn
   - bóng đổ
4. **Kiểm tra duplicate** và ảnh gần giống nhau giữa các split.
5. Chỉ thử tăng `imgsz` sau khi dữ liệu đã được làm sạch hơn.

### 8.3 Khi nào nên tăng dữ liệu

Có thể tăng dữ liệu từ tập train hiện tại lên lớn hơn, nhưng cần ưu tiên:

- ảnh khó cho `crack`, `scratch`, `dent`.
- ảnh mới có giá trị thông tin cao.
- tránh thêm quá nhiều ảnh gần như giống hệt nhau.

Nguyên tắc: **dữ liệu tốt hơn quan trọng hơn dữ liệu nhiều hơn**.

---

## 9. Duplicate Policy

### 9.1 Exact duplicate

Nếu ảnh trùng hoàn toàn và label cũng trùng hoàn toàn:

- có thể loại bớt, nhất là trong train.
- phải loại nếu ảnh trùng xuất hiện ở các split khác nhau.

### 9.2 Duplicate giữa train / val / test

Đây là loại rủi ro nhất vì có thể gây leakage.

Nguyên tắc:

- một ảnh chỉ nên nằm ở **một split duy nhất**.

### 9.3 Ảnh trùng nhưng label khác nhau

Không xóa ngay. Cần review thủ công vì có thể đây là lỗi annotation hoặc lỗi merge dataset.

---

## 10. Troubleshooting

### Nếu YOLO mAP thấp

- kiểm tra label.
- kiểm tra duplicate giữa các split.
- kiểm tra class imbalance.
- kiểm tra xem dữ liệu mới có thực sự tốt hơn hay chỉ nhiều hơn.
- kiểm tra các lớp khó `crack`, `scratch`, `dent` trước tiên.

### Nếu fine-tune không cải thiện

- dừng tăng epochs thêm.
- quay về checkpoint baseline tốt nhất.
- chuyển sang hướng cải thiện dữ liệu.

### Nếu val tốt nhưng test thấp

- kiểm tra leakage.
- kiểm tra phân phối class giữa val và test.
- kiểm tra test set có quá nhỏ hay không.

---

## 11. Experiment Logging Convention

Mỗi lần train nên lưu lại:

- tên run
- model variant
- dataset version
- epochs
- batch size
- image size
- checkpoint được chọn
- metrics trên val
- metrics trên test
- nhận xét ngắn về lý do chọn hoặc loại run đó

Tài liệu ghi chép các lần train cụ thể nên để trong `training_report.md`.

---

## 12. Recommended Next Steps

1. Giữ `YOLOv8s` trên **dataset ban đầu** làm baseline chính.
2. Chỉ train lại khi dữ liệu mới đã được audit rõ ràng.
3. Tập trung vào `crack`, `scratch`, `dent` thay vì chỉ tiếp tục tăng epochs.
4. Dùng checkpoint hiện tại của baseline cho pipeline detection và cho downstream crop-based severity classification.
5. Sau khi dữ liệu tốt hơn, benchmark lại trên cùng một test protocol để quyết định có thay baseline hay không.
