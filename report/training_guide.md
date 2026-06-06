# Training Guide — Used Car Price Prediction Project

## 1. Purpose

Tài liệu này hướng dẫn cách huấn luyện, đánh giá và chọn mô hình cho dự án dự đoán giá xe ô tô cũ với pipeline đa mô hình gồm:

- **YOLOv8** để phát hiện hư hỏng trên ảnh xe.
- **CNN** để phân loại mức độ nghiêm trọng của hư hỏng từ ảnh full thành `minor`, `moderate`, `severe`.
- **XGBoost** để dự đoán **base price** từ dữ liệu bảng.
- **Rule-based adjustment layer** để tạo **damage-aware adjusted price** từ base price và thông tin damage trên ảnh.

Trong ứng dụng Streamlit hiện tại, ảnh xe là đầu vào **không bắt buộc**. Nếu người dùng không upload ảnh, pipeline chỉ chạy nhánh tabular và final price bằng base price từ XGBoost. Nếu có ảnh, hệ thống mới chạy thêm YOLO, CNN ConvNeXt-Tiny và lớp điều chỉnh giá.

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

- **Input:** ảnh full người dùng upload.
- **Output:** `minor`, `moderate`, `severe`.
- **Vai trò:** lượng hóa mức độ nghiêm trọng tổng quát của hư hỏng trong ảnh để hỗ trợ bước điều chỉnh giá.
- **Mô hình chính:** ConvNeXt-Tiny, được chọn sau khi so sánh với ResNet18, ResNet50, EfficientNet-B0 và EfficientNet-B2.
- **Checkpoint đang dùng:** `Models/ConvNeXt.pkl`.

### 2.3 XGBoost — Base Price Prediction

- **Input:** dữ liệu bảng của xe cũ.
- **Output:** giá cơ bản của xe.
- **Vai trò:** dự đoán **base market price** trước khi điều chỉnh theo damage nhìn thấy trên ảnh.
- **Checkpoint đang dùng:** `Models/model.pkl`, gồm preprocessor, danh sách feature được chọn và mô hình XGBoost.

### 2.4 Rule-Based Adjustment

- **Input:** base price từ XGBoost, damage features từ YOLO, severity từ CNN.
- **Output:** adjusted price.
- **Lưu ý:** hiện tại chưa có ground-truth chắc chắn cho final price after damage, nên module này được hiểu là **price adjustment estimator**, không phải supervised final-price predictor. Khi không có damage detection, module trả về `final_price = base_price`.

---

## 3. Datasets

### 3.1 YOLO Dataset

Nguồn dữ liệu gốc cho nhánh YOLO là bộ `car-damage-detection` trên Kaggle: <https://www.kaggle.com/datasets/asfarhossainsitab/car-damage-detection>. Trong dự án, phần detection được lấy từ dữ liệu `CarDD_COCO`. Vì dữ liệu tải về ban đầu ở định dạng COCO, dự án đã upload dữ liệu lên Roboflow để chuyển đổi và export lại theo định dạng YOLOv8, thuận tiện cho quá trình huấn luyện bằng Ultralytics.

Dataset detection chính của dự án hiện tại là bộ dữ liệu theo format YOLO với 6 class:

- `crack`
- `dent`
- `glass shatter`
- `lamp broken`
- `scratch`
- `tire flat`

Phiên bản dữ liệu dùng cho mô hình YOLO cuối là **merge_Data**. Dataset ban đầu có khoảng **4000 ảnh** và được dùng cho các run baseline như `s_89`. Sau đó, dự án bổ sung thêm **2307 ảnh** cho ba lớp yếu nhất là `crack`, `scratch` và `dent`, nâng tổng số ảnh lên khoảng **6307 ảnh**. Dữ liệu bổ sung được lấy từ Roboflow Universe, chủ yếu từ hai nguồn AutoDentify Car Damage Detection (<https://universe.roboflow.com/autodentify/car-damage-detection-ggmju>) và Scratch and Dent (<https://universe.roboflow.com/nibm-7v215/scratch-and-dent-xvjy5>). Việc bổ sung dữ liệu có chủ đích này nhằm cải thiện khả năng phát hiện các hư hỏng nhỏ, mảnh và dễ bị bỏ sót, chứ không phải tăng đều dữ liệu cho tất cả các nhãn.

### Thư mục tham khảo trong môi trường train

```text
coco_damage.yolov8/
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

Trong repository hiện tại, dữ liệu ảnh đầy đủ không được đặt trực tiếp trong `Datasets/`; các notebook huấn luyện, checkpoint và kết quả đánh giá được lưu trong `train/`, `Models/` và `Quan_sat/`.

### Quy tắc dùng split

- `train`: dùng để học tham số.
- `val`: dùng để theo dõi `Precision`, `Recall`, `mAP50`, `mAP50-95` trong quá trình tune.
- `test`: chỉ dùng để đánh giá cuối sau khi đã chốt mô hình và cấu hình.

### Lưu ý quan trọng

Trong các run YOLO của dự án, metric được in sau mỗi epoch là metric trên **tập validation**, không phải test set. Vì vậy:

- log train phản ánh kết quả trên `val`.
- muốn có kết quả cuối trên `test`, cần gọi evaluate riêng bằng `model.val(split="test")`.

### 3.2 CNN Severity Dataset

Nguồn dữ liệu cho nhánh severity classification cũng là bộ `car-damage-detection` trên Kaggle: <https://www.kaggle.com/datasets/asfarhossainsitab/car-damage-detection>. Khác với YOLO, nhánh CNN sử dụng ảnh full và nhãn mức độ hư hỏng để phân loại ảnh thành ba mức độ.

Dataset severity classification gồm 3 lớp:

- `minor`
- `moderate`
- `severe`

Trong hướng hiện tại, CNN severity classification sử dụng **ảnh full** làm đầu vào, không phụ thuộc vào crop bounding box của YOLO. Điều này giúp phần CNN có thể chạy trực tiếp trên ảnh người dùng upload, giống giao diện demo phân loại ảnh full.

Khuyến nghị:

- giữ cùng kiểu đầu vào giữa train và inference: nếu train bằng ảnh full thì khi demo cũng đưa ảnh full vào CNN.
- kiểm tra chất lượng nhãn `minor`, `moderate`, `severe`, vì lớp `moderate` thường dễ nhập nhằng với hai lớp còn lại.
- ưu tiên macro F1 bên cạnh accuracy để tránh chọn mô hình chỉ tốt trên lớp chiếm ưu thế.

### 3.3 XGBoost Dataset

Nguồn dữ liệu bảng cho nhánh dự đoán giá là bộ `used-cars-price-prediction` trên Kaggle: <https://www.kaggle.com/datasets/avikasliwal/used-cars-price-prediction>. Dataset này có ground truth là giá xe cũ, được dùng để huấn luyện mô hình hồi quy dự đoán base price.

Dataset tabular dùng để dự đoán giá xe từ các feature có cấu trúc như:

- `Loai_nhien_lieu`
- `Hop_so`
- `Quyen_so_huu`
- `Muc_tieu_hao(km/l)`
- `Dung_tich(cc)`
- `Cong_suat_toi_da`
- `So_cho_ngoi`
- `Tuoi_xe`
- `Hang_xe`
- `Km_moi_nam`
- `Chay_nhieu`
- `log_Quang_duong_da_di(km)`
- `Top_xe`

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

1. Train nhiều biến thể mô hình trên dataset ban đầu khoảng 4000 ảnh, gồm:
   - `YOLOv8n`
   - `YOLOv8s`
   - `YOLOv8m`
2. Chọn `YOLOv8s` làm baseline phù hợp nhất về cân bằng tốc độ và độ chính xác.
3. Tạo phiên bản dataset **merge_Data** bằng cách thêm 2307 ảnh từ Roboflow Universe cho các lớp khó `dent`, `scratch`, `crack`.
4. Train lại YOLOv8s trên `merge_Data` với cùng cấu hình `epochs=100`, `batch=16`, `imgsz=640`.

### 5.2 Kết luận từ phiên bản merge_Data

Việc mở rộng dữ liệu không được thực hiện đồng đều cho toàn bộ nhãn, mà tập trung vào ba lớp yếu nhất là `crack`, `scratch` và `dent`. Đây là lựa chọn phù hợp với đặc điểm bài toán vì các lớp này thường có vùng hư hỏng nhỏ, mảnh, dễ lẫn với phản sáng, đường gân thân xe hoặc vết bẩn.

Kết quả từ `Quan_sat/yolo_car_report/results.csv` cho thấy mô hình YOLOv8s trên `merge_Data` vẫn giữ hiệu năng ổn định với cùng cấu hình train. So với run `s_89` trên dataset gốc, mô hình mới có lợi thế lớn hơn về dữ liệu cho các lớp khó và phù hợp hơn với mục tiêu phát hiện hư hỏng thực tế.

Khi đọc kết quả mAP, cần lưu ý khả năng xuất hiện **Test Set Label Noise**. Nếu tập test/validation gán nhãn thiếu các vết xước, móp hoặc nứt mờ, mô hình mới có thể phát hiện đúng các tổn thất này nhưng vẫn bị hệ thống chấm điểm tính là **False Positive** do không có ground truth tương ứng. Vì vậy, checkpoint cuối nên được đánh giá kết hợp giữa metrics tĩnh, ảnh dự đoán mẫu và inference trên dữ liệu thực tế.

Điều này dẫn tới quyết định hiện tại:

- `s_89` được giữ làm baseline tham khảo trên dataset gốc.
- YOLOv8s train trên `merge_Data` được chọn làm mô hình YOLO chính của dự án.

### 5.3 Checkpoint YOLO được chọn hiện tại

Checkpoint YOLO chính hiện tại trong ứng dụng là `Models/best.pt`. Mã nguồn `dich_vu/phat_hien_hu_hong.py` tải checkpoint này và chạy suy luận với cấu hình:

- `model = YOLOv8s`
- `dataset = merge_Data`
- `epochs = 100`
- `batch = 16`
- `imgsz = 640` khi inference trong app
- `conf = 0.25`
- `iou = 0.45`

Kết quả được ghi nhận để chọn checkpoint này:

- **best mAP50 trong `Quan_sat/yolo_car_report/results.csv`: epoch 79, Precision ≈ 0.801, Recall ≈ 0.682, mAP50 ≈ 0.726, mAP50-95 ≈ 0.568**
- **best mAP50-95 trong `Quan_sat/yolo_car_report/results.csv`: epoch 75, mAP50-95 ≈ 0.572**

Lưu ý: các chỉ số trong `results.csv` là kết quả validation của run YOLO, không phải test set độc lập. Nếu cần báo cáo số liệu test cuối cùng, cần chạy đánh giá riêng bằng `model.val(split="test")` sau khi đã chốt checkpoint.

### 5.4 Nguyên tắc chọn baseline

`YOLOv8s` trên `merge_Data` được giữ làm mô hình chính vì:

- kế thừa cấu hình ổn định từ baseline `s_89` nhưng có thêm dữ liệu cho các lớp yếu.
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

Quy trình huấn luyện CNN dùng **transfer learning hai phase**. Phase 1 đóng băng backbone và chỉ train classifier/head. Phase 2 mở băng các block cuối cùng với classifier để mô hình học thêm đặc trưng riêng của ảnh hư hỏng xe, nhưng vẫn hạn chế cập nhật quá nhiều trọng số pretrained.

| Notebook                | Backbone                        | Classifier/head thay thế                      | Phase 1 train | Phase 2 fine-tune            |
| ----------------------- | ------------------------------- | --------------------------------------------- | ------------- | ---------------------------- |
| `cnn_train_car.ipynb`   | ResNet18 bản thử nghiệm ban đầu | `fc = Linear(..., 3)`                         | `fc`          | `layer4 + fc`                |
| `resnet_18_new.ipynb`   | ResNet18                        | `fc = Dropout(0.35) + Linear(..., 3)`         | `fc`          | `layer4 + fc`                |
| `EfficientNet_B0.ipynb` | EfficientNet-B0                 | `classifier = Dropout(0.40) + Linear(..., 3)` | `classifier`  | `features[-2:] + classifier` |
| `Efficient_B2.ipynb`    | EfficientNet-B2                 | `classifier = Dropout(0.40) + Linear(..., 3)` | `classifier`  | `features[-2:] + classifier` |
| `ResNet50.ipynb`        | ResNet50                        | `fc = Dropout(0.35) + Linear(..., 3)`         | `fc`          | `layer4 + fc`                |
| `ConvNeXt_Tiny.ipynb`   | ConvNeXt-Tiny                   | `classifier[2] = Linear(..., 3)`              | `classifier`  | `features[-2:] + classifier` |

Tất cả các mô hình đều dự đoán ba lớp severity: `minor`, `moderate`, `severe`. Với ResNet, phase 2 mở `layer4` vì đây là stage residual cuối. Với EfficientNet-B0, EfficientNet-B2 và ConvNeXt-Tiny, phase 2 mở `features[-2:]` vì đây là nhóm block cuối trong backbone.

Các notebook CNN đã thử nghiệm:

| Notebook                | Backbone        | Best validation metric | Test accuracy | Test macro F1 |
| ----------------------- | --------------- | ---------------------: | ------------: | ------------: |
| `resnet_18_new.ipynb`   | ResNet18        |       val_acc = 0.8209 |        0.6564 |        0.6490 |
| `EfficientNet_B0.ipynb` | EfficientNet-B0 |  val_macro_f1 = 0.7945 |        0.6821 |        0.6817 |
| `Efficient_B2.ipynb`    | EfficientNet-B2 |  val_macro_f1 = 0.8022 |        0.6974 |        0.6954 |
| `ResNet50.ipynb`        | ResNet50        |       val_acc = 0.8128 |        0.7026 |        0.7043 |
| `ConvNeXt_Tiny.ipynb`   | ConvNeXt-Tiny   |  val_macro_f1 = 0.8342 |        0.7077 |        0.7084 |

ConvNeXt-Tiny được chọn làm mô hình CNN chính vì có test accuracy và test macro F1 cao nhất trong các mô hình đã thử. Mô hình này cũng đạt validation macro F1 cao nhất, cho thấy khả năng cân bằng giữa ba lớp tốt hơn so với ResNet18 và các biến thể EfficientNet trong thí nghiệm hiện tại.

### 7.3 XGBoost

Metric chính:

- MAE
- RMSE
- MAPE
- R²

Target hiện tại là **base price**, không phải final price after damage.

Kết quả pipeline tabular hiện tại cho thấy XGBoost là mô hình tốt nhất trong các mô hình đã so sánh, với R² test khoảng **0.9367**, RMSE khoảng **1.4313** và MAE khoảng **0.8914** trên thang giá gốc của dataset. Mô hình này được lưu trong `Models/model.pkl` để ứng dụng sử dụng trực tiếp.

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

1. Giữ `YOLOv8s` trên **merge_Data** làm mô hình detection chính.
2. Dùng `s_89` như baseline tham khảo khi cần so sánh với dataset gốc.
3. Tiếp tục ưu tiên audit nhãn và bổ sung dữ liệu cho `crack`, `scratch`, `dent`.
4. Dùng `Models/best.pt` cho pipeline detection và dùng ConvNeXt-Tiny làm mô hình CNN chính để phân loại severity trực tiếp từ ảnh full người dùng upload.
5. Nếu có dữ liệu mới, benchmark lại trên cùng một validation/test protocol để quyết định có thay checkpoint hiện tại hay không.
