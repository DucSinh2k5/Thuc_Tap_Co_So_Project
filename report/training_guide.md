# Training Guide — Used Car Price Prediction Project

## 1. Mục đích

Tài liệu này hướng dẫn cách huấn luyện, đánh giá, chọn mô hình và cải thiện dữ liệu cho dự án dự đoán giá xe ô tô cũ.  
Dự án sử dụng pipeline đa mô hình gồm:

- **YOLOv8** để phát hiện hư hỏng trên ảnh xe
- **CNN** để phân loại mức độ nghiêm trọng của vùng hư hỏng (`minor`, `moderate`, `severe`)
- **XGBoost** để dự đoán **base price** từ dữ liệu bảng
- **Rule-based adjustment layer** để ước lượng **adjusted price** dựa trên hư hỏng nhìn thấy trên ảnh

Tài liệu này tập trung vào câu hỏi: **muốn train lại hoặc cải thiện hệ thống thì phải làm gì, theo thứ tự nào, và đánh giá bằng cách nào**.

---

## 2. Tổng quan pipeline huấn luyện

### 2.1 Mục tiêu của từng mô hình

#### YOLOv8 — Damage Detection

- Input: ảnh xe
- Output: bounding boxes và class của hư hỏng
- Vai trò: tạo damage features cho bước downstream như:
  - `num_dents`
  - `num_scratches`
  - `num_cracks`
  - `total_damage_area`
  - số lượng hư hỏng theo class

#### CNN — Severity Classification

- Input: crop vùng hư hỏng được cắt ra từ YOLO hoặc crop đã gán nhãn sẵn
- Output: `minor`, `moderate`, `severe`
- Vai trò: lượng hóa mức độ hư hỏng để đưa vào lớp điều chỉnh giá

#### XGBoost — Base Price Prediction

- Input: dữ liệu bảng của xe cũ
- Ví dụ: năm sản xuất, số ghế, số km đã đi, nhiên liệu, hộp số, hãng, dòng xe
- Output: **base market price**
- Vai trò: tạo giá cơ sở trước khi điều chỉnh theo damage

#### Rule-Based Adjustment

- Input:
  - base price từ XGBoost
  - damage features từ YOLO
  - severity từ CNN
- Output: **damage-aware adjusted price**
- Lưu ý:
  - vì hiện tại chưa có ground truth rõ ràng cho final price after damage, module này nên được diễn giải là **price adjustment estimator**, không phải supervised final-price regressor

---

## 3. Datasets

### 3.1 Dataset cho YOLOv8

Dataset YOLO chính hiện tại là bộ dữ liệu damage detection theo format YOLOv8 với 6 class:

- `crack`
- `dent`
- `glass shatter`
- `lamp broken`
- `scratch`
- `tire flat`

### Cấu trúc thư mục tham khảo

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

### Split đã dùng trong các run baseline

- **Train:** 2800 images
- **Validation:** 800 images
- **Test:** giữ riêng để đánh giá cuối

### Nguyên tắc dùng split

- `train` dùng để học tham số
- `val` dùng để theo dõi:
  - Precision
  - Recall
  - mAP50
  - mAP50-95
- `test` **không dùng để tune model**
- chỉ đánh giá `test` sau khi đã chọn xong model/config tốt nhất

### Lưu ý quan trọng

Trong các run YOLO trước đây, metric sau mỗi epoch được tính trên **tập `val`**, không phải `test`.  
Vì vậy:

- nếu nhìn kết quả trong log train, đó là **validation metrics**
- nếu muốn có đánh giá cuối, cần chạy `model.val(split="test")` riêng sau khi train

---

### 3.2 Dataset cho CNN Severity Classification

Dataset severity classification gồm 3 class:

- `minor`
- `moderate`
- `severe`

### Mục đích

CNN không dự đoán loại damage, mà chỉ dự đoán **mức độ nghiêm trọng** của vùng hư hỏng.

### Input khuyến nghị

- ảnh crop đúng vùng damage
- crop càng sát vùng hỏng càng tốt
- tránh crop quá rộng làm nhiễu nền

### Ghi chú thực hành

- Nếu crop từ YOLO, cần kiểm tra chất lượng crop
- Nếu bbox YOLO chưa tốt, CNN severity cũng sẽ bị kéo xuống

---

### 3.3 Dataset cho XGBoost

Dataset tabular dùng để dự đoán giá xe cũ từ thông tin có cấu trúc.

### Mục tiêu

Dự đoán **base market price** của xe dựa trên các feature bảng.

### Ví dụ feature

- `year`
- `num_seats`
- `km_driven`
- `fuel`
- `transmission`
- `brand`
- `model`

### Ghi chú

Base price từ XGBoost nên được hiểu là:

- giá thị trường của xe ở trạng thái trung bình
- chưa phản ánh chính xác hoàn toàn từng hư hỏng cụ thể nhìn thấy trên ảnh

---

## 4. Môi trường huấn luyện

### 4.1 Phần mềm

Khuyến nghị:

- Python 3.10+
- PyTorch
- Ultralytics
- scikit-learn
- xgboost
- pandas
- numpy
- matplotlib
- opencv-python

### 4.2 Phần cứng

Có thể train trên:

- Kaggle Notebook
- Google Colab
- máy local có GPU

### Ghi chú từ các run trước

Một số run YOLO baseline đã được train trên **Tesla T4 ~15GB VRAM**.  
Với môi trường này, cấu hình thực tế ổn định nhất là:

- `imgsz=640`
- `batch=16`

---

## 5. Baseline hiện tại và cách chọn mô hình

### 5.1 YOLO baseline hiện tại

Từ các run đã thực hiện trên dataset YOLO 6 lớp:

- **YOLOv8n**: `mAP50 ≈ 0.714`, `mAP50-95 ≈ 0.568`
- **YOLOv8s**: `mAP50 ≈ 0.735`, `mAP50-95 ≈ 0.587`
- **YOLOv8m**: `mAP50 ≈ 0.728`, `mAP50-95 ≈ 0.590`

### Kết luận chọn baseline

**YOLOv8s** được chọn làm baseline chính vì:

- tốt hơn `yolov8n` khá rõ về tổng thể
- gần tương đương `yolov8m` về chất lượng
- nhẹ và nhanh hơn `yolov8m`
- phù hợp nhất để tích hợp vào pipeline hiện tại

### Nhận xét thêm

`yolov8m` chỉ cải thiện rất ít so với `yolov8s`, trong khi chi phí tính toán lớn hơn.  
Khi các model variant cho kết quả gần nhau, nên ưu tiên cải thiện **dữ liệu** trước khi tiếp tục tăng kích thước model.

---

### 5.2 Các lớp mạnh và lớp yếu của YOLO baseline

### Các lớp mạnh

- `glass shatter`
- `lamp broken`
- `tire flat`

Các lớp này có mAP cao và ổn định hơn.

### Các lớp yếu / khó

- `crack`
- `scratch`
- `dent`

Đây là các lớp cần ưu tiên cải thiện vì:

- khó nhìn
- nhỏ, mảnh hoặc biên không rõ
- dễ nhầm với phản sáng, vệt bẩn, đường viền thân xe
- nhãn dễ không nhất quán

### Ưu tiên cải thiện

1. `crack`
2. `scratch`
3. `dent`

---

### 5.3 XGBoost baseline

XGBoost hiện đang là baseline cho **base price prediction**.

### Vai trò

- không thay thế YOLO/CNN
- mà là thành phần dự đoán giá cơ sở trước khi điều chỉnh theo damage

### Nguyên tắc chọn model

Chọn model có:

- MAE thấp
- RMSE thấp
- MAPE thấp
- R² cao
- và feature importance hợp lý

---

### 5.4 CNN baseline

CNN severity classifier hiện là baseline cho bài toán `minor / moderate / severe`.

### Nguyên tắc chọn model

Không chỉ nhìn Accuracy.  
Cần đánh giá thêm:

- Macro F1
- Confusion Matrix
- Precision / Recall theo từng lớp

Vì `minor` và `moderate` thường dễ chồng lấn.

---

## 6. Hướng dẫn train YOLOv8

### 6.1 Khi nào train YOLO?

Train lại YOLO khi:

- có thêm dữ liệu mới
- đã relabel / làm sạch nhãn
- muốn benchmark model variant mới
- muốn xác nhận thay đổi augmentation, image size hoặc learning rate có ích hay không

### 6.2 Cấu hình baseline khuyến nghị

```python
from ultralytics import YOLO
import torch

model = YOLO("yolov8s.pt")

results = model.train(
    data="/path/to/data.yaml",
    epochs=70,
    imgsz=640,
    batch=16,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    project="/path/to/results",
    name="yolov8s_baseline",
)
```

### Cấu hình baseline hiện tại

- model: `yolov8s.pt`
- epochs: `70`
- imgsz: `640`
- batch: `16`
- optimizer: `auto`
- validation: `split=val`

### 6.3 Evaluate riêng sau khi train

Sau khi train xong, luôn đánh giá riêng trên `test` nếu có.

```python
from ultralytics import YOLO
import torch

model = YOLO("/path/to/results/yolov8s_baseline/weights/best.pt")

metrics = model.val(
    data="/path/to/data.yaml",
    split="test",
    imgsz=640,
    batch=16,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
)

print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)

for i, name in model.names.items():
    p, r, ap50, ap = metrics.box.class_result(i)
    print(name, p, r, ap50, ap)
```

### 6.4 Resume và fine-tune

#### Resume đúng run cũ

Dùng khi run bị dừng giữa chừng:

```python
from ultralytics import YOLO

model = YOLO("/path/to/last.pt")
results = model.train(resume=True)
```

#### Fine-tune từ best.pt

Dùng khi muốn tinh chỉnh tiếp từ model tốt nhất:

```python
from ultralytics import YOLO
import torch

model = YOLO("/path/to/best.pt")

results = model.train(
    data="/path/to/data.yaml",
    epochs=30,
    imgsz=640,
    batch=16,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    project="/path/to/results",
    name="yolov8s_ft",
)
```

### Khi nào nên fine-tune?

Nên fine-tune khi:

- đã có `best.pt` tốt
- muốn train thêm vài epoch
- đã cải thiện dữ liệu
- hoặc muốn giảm learning rate để tinh chỉnh tiếp

---

## 7. Hướng dẫn train CNN Severity Classification

### 7.1 Mục tiêu

Dự đoán severity cho từng vùng damage:

- `minor`
- `moderate`
- `severe`

### 7.2 Quy trình khuyến nghị

1. Chuẩn bị dữ liệu crop
2. Chia `train / val / test`
3. Train classifier
4. Theo dõi loss và metric
5. Chọn model tốt nhất theo Macro F1 trên `val`
6. Evaluate cuối trên `test`

### 7.3 Metric cần theo dõi

- Accuracy
- Precision
- Recall
- Macro F1
- Confusion Matrix

### 7.4 Dấu hiệu cần can thiệp

- Accuracy cao nhưng Macro F1 thấp
- `minor` bị nhầm nhiều sang `moderate`
- `severe` có recall thấp
- loss train giảm nhưng val loss tăng

### 7.5 Hướng cải thiện chính

- tăng epoch nếu train quá ít
- augment dữ liệu
- kiểm tra chất lượng crop
- cân bằng lại class nếu cần

---

## 8. Hướng dẫn train XGBoost

### 8.1 Mục tiêu

Dự đoán **base price** từ dữ liệu bảng.

### 8.2 Quy trình khuyến nghị

1. Làm sạch dữ liệu bảng
2. Encode categorical features
3. Chia `train / val / test`
4. Train XGBoost
5. Tune nhẹ nếu cần
6. Evaluate trên `test`

### 8.3 Metric chính

- MAE
- RMSE
- MAPE
- R²

### 8.4 Dấu hiệu model ổn

- MAE và RMSE hợp lý
- MAPE không quá cao
- feature importance khớp trực giác
- prediction không lệch quá mạnh ở xe đắt hoặc xe rẻ

### 8.5 Vai trò trong pipeline

XGBoost không cần học damage từ ảnh.  
Nó chỉ cần làm tốt bài toán **base price prediction**.

---

## 9. Chiến lược đánh giá

### 9.1 Với YOLO

#### Trong lúc phát triển model

Dùng `val` để:

- so sánh `yolov8n / yolov8s / yolov8m`
- so sánh epochs
- so sánh batch size
- theo dõi mAP, precision, recall

#### Sau khi đã chốt config

Dùng `test` để:

- đánh giá cuối
- báo cáo metric sạch hơn
- so sánh công bằng giữa các bản model cuối

#### Quy tắc quan trọng

**Tập nào đã dùng để chọn model thì không được gọi là test.**

### 9.2 Với CNN

- tune trên `val`
- báo cáo cuối trên `test`

### 9.3 Với XGBoost

- tune trên `val` hoặc cross-validation
- báo cáo cuối trên `test`

---

## 10. Các metric cần theo dõi và cách diễn giải

### 10.1 YOLO

#### Metric chính

- Precision
- Recall
- mAP50
- mAP50-95

#### Ưu tiên thực tế của bài toán

Đối với car damage detection trong pipeline pricing:

1. mAP50-95
2. mAP50
3. Recall các lớp khó
4. tốc độ inference

#### Cách đọc metric

- **Precision cao**: ít false positive hơn
- **Recall cao**: ít bỏ sót hơn
- **mAP50**: chất lượng detection tổng quát
- **mAP50-95**: nghiêm ngặt hơn, phản ánh bbox quality tốt hơn

### 10.2 CNN

- Accuracy
- Macro F1
- confusion matrix

### 10.3 XGBoost

- MAE
- RMSE
- MAPE
- R²

---

## 11. Quy tắc chọn model

### 11.1 YOLO

#### Chọn mô hình theo thứ tự

1. chất lượng trên `val`
2. chất lượng trên `test`
3. tốc độ inference
4. kích thước model
5. độ ổn định qua nhiều run

#### Baseline hiện tại

Ưu tiên:

- **YOLOv8s trên bộ dữ liệu cũ 6 lớp**

#### Không nên chọn model chỉ vì:

- train lâu hơn
- model lớn hơn
- dataset nhiều ảnh hơn nhưng nhãn kém hơn
- mAP chỉ nhỉnh rất ít nhưng inference chậm hơn đáng kể

### 11.2 CNN

Chọn model có:

- Macro F1 tốt nhất
- confusion matrix hợp lý
- ít bias về một class

### 11.3 XGBoost

Chọn model có lỗi thấp và ổn định trên test.

---

## 12. Khi nào nên chỉnh hyperparameter?

### 12.1 Khi nào nên chỉnh

Chỉ nên ưu tiên chỉnh hyperparameter khi:

- dữ liệu đã tương đối sạch
- model đang học được nhưng còn dư địa
- metric giữa các run khác nhau rõ rệt

### 12.2 Khi nào không nên quá tập trung vào hyperparameter

Nếu:

- `yolov8n / s / m` cho kết quả gần nhau
- fine-tune `best.pt` chỉ cải thiện rất ít
- tăng epochs hoặc tăng image size không thay đổi đáng kể

thì nhiều khả năng nút thắt nằm ở **dữ liệu**, không phải hyperparameter.

### 12.3 Kinh nghiệm thực tế của dự án

Trong dự án này, khi các variant cho kết quả tương đối gần nhau, nên:

- ưu tiên cải thiện dữ liệu
- thay vì tiếp tục vặn epochs, batch size, learning rate trong thời gian dài

---

## 13. Learning rate, epochs, batch size, image size

### 13.1 Epochs

#### Khuyến nghị hiện tại

- baseline: `70`
- thử mở rộng hợp lý: `100`

#### Khi nào nên tăng

- loss còn giảm
- val metric còn tăng
- chưa thấy dấu hiệu overfit

#### Khi nào không nên tăng thêm

- mAP gần như đứng yên
- model khác nhau chỉ chênh rất ít
- đã thử nhiều lượt mà không đổi đáng kể

### 13.2 Batch size

#### Với Tesla T4 ~15GB

Khuyến nghị thực tế:

- `batch=16` là baseline tốt
- có thể thử `12` khi tăng `imgsz`
- không nên tăng quá mạnh nếu VRAM hạn chế

### 13.3 Image size

#### Khi nào nên thử tăng

- sau khi đã làm sạch dữ liệu
- đặc biệt nếu lớp khó là vật thể nhỏ hoặc mảnh

#### Khuyến nghị

- baseline: `640`
- thử thêm: `800`

### 13.4 Learning rate

#### Điều quan trọng

Nếu dùng `optimizer="auto"`, Ultralytics có thể bỏ qua `lr0` người dùng đặt và tự chọn optimizer/lr.

#### Khuyến nghị

- train baseline: giữ `optimizer=auto`
- fine-tune từ `best.pt`: có thể đặt optimizer rõ ràng và giảm lr
- ví dụ:
  - `AdamW`
  - `lr0=3e-4`
  - hoặc `1e-4` cho fine-tune nhẹ

---

## 14. Data-centric improvement strategy

Đây là phần quan trọng nhất của guide.

### 14.1 Kết luận chung

Khi các model YOLO khác nhau cho kết quả gần nhau, nên ưu tiên:

- **cải thiện dữ liệu**
- hơn là tiếp tục vặn hyperparameter

### 14.2 Ba lớp cần ưu tiên

- `crack`
- `scratch`
- `dent`

### 14.3 Những việc nên làm

#### 1. Audit nhãn

Kiểm tra:

- bbox có quá lỏng hoặc quá chặt không
- cùng một kiểu damage có bị gán class khác nhau không
- `scratch` và `crack` có bị chồng nghĩa không
- `dent` có được annotate nhất quán không

#### 2. Bổ sung dữ liệu có chủ đích

Không thêm ảnh ngẫu nhiên.  
Nên thêm:

- crack nhỏ, mảnh, khó thấy
- scratch phản sáng, mỏng, dài
- dent ở góc xiên, ánh sáng xấu
- ảnh có background dễ gây nhầm

#### 3. Thêm hard negatives

Ví dụ:

- vệt sáng phản chiếu
- mép cửa, đường ghép thân xe
- vệt bụi, bùn, nước
- họa tiết, decal, bóng đổ

#### 4. Kiểm tra duplicate

Nên kiểm tra:

- duplicate trong `train`
- duplicate giữa `train` và `val/test`
- cùng ảnh nhưng label khác nhau

#### 5. Cân lại split nếu cần

- giữ `test` riêng khi có thể
- dùng `val` để tune
- nếu dữ liệu rất ít, có thể xem xét k-fold trên `train + val`

### 14.4 Điều không nên làm đầu tiên

Không nên mặc định:

- tăng model size
- tăng epoch mãi
- tăng threshold trong tool annotate
- đổi sang dataset nhiều ảnh hơn nhưng chưa audit nhãn

---

## 15. Duplicate policy

### 15.1 Các loại duplicate

#### Exact duplicate

- ảnh giống hệt nhau
- label giống hệt

#### Cross-split duplicate

- cùng ảnh xuất hiện ở `train` và `val/test`

#### Same image, different label

- ảnh trùng nhưng annotation khác

### 15.2 Cách xử lý

- Exact duplicate: xóa bớt
- Cross-split duplicate: giữ ở một split duy nhất
- Same image, different label: review thủ công

### 15.3 Near-duplicate

Nếu là nhiều frame gần giống nhau:

- không cần xóa hết
- nhưng không nên để quá nhiều ảnh gần như giống hệt nhau trong cùng split

---

## 16. Troubleshooting

### 16.1 YOLO mAP thấp

Kiểm tra:

- `data.yaml`
- class mapping
- label format
- duplicate
- split leakage
- class imbalance
- chất lượng nhãn

### 16.2 Recall thấp ở lớp khó

Làm theo thứ tự:

1. audit label
2. thêm dữ liệu khó có chủ đích
3. thêm hard negatives
4. tăng image size
5. fine-tune lại

### 16.3 Precision thấp

Kiểm tra:

- background gây nhầm
- conf threshold khi inference
- hard negatives
- nhãn false positive

### 16.4 CNN không học tốt

Kiểm tra:

- crop có đúng vùng damage không
- class imbalance
- epoch có quá ít không
- ảnh có nhiễu nền quá nhiều không

### 16.5 XGBoost lỗi cao

Kiểm tra:

- missing values
- encoding categorical
- outliers
- leakage
- feature engineering

---

## 17. Quy ước lưu kết quả

### 17.1 Thư mục gợi ý

```text
project_root/
├── Datasets/
├── notebooks/
├── models/
├── results/
│   ├── yolo/
│   ├── cnn/
│   └── xgboost/
├── reports/
│   ├── dataset_report.md
│   ├── training_report.md
│   └── training_guide.md
└── src/
```

### 17.2 Quy tắc đặt tên run

Ví dụ:

- `yolov8s_baseline_70e`
- `yolov8s_ft_30e_lr3e4`
- `yolov8s_img800_b12`
- `cnn_resnet18_severity_v1`
- `xgb_baseprice_v1`

### 17.3 Những file nên lưu

- `best.pt`
- `last.pt`
- `results.csv`
- confusion matrix / plots quan trọng
- script train
- note ngắn cho từng run

---

## 18. Kế hoạch huấn luyện khuyến nghị tiếp theo

### 18.1 YOLO

#### Giai đoạn 1 — giữ baseline

- giữ `yolov8s` bộ cũ làm baseline chính

#### Giai đoạn 2 — cải thiện dữ liệu

- audit `crack / scratch / dent`
- loại duplicate
- thêm hard negatives
- tăng dữ liệu từ khoảng 2800 lên 4000 theo hướng có chủ đích

#### Giai đoạn 3 — train lại

Thử các run sau:

1. `yolov8s`, `epochs=100`, `imgsz=640`, `batch=16`
2. `best.pt` fine-tune thêm `20–30 epochs`
3. `yolov8s`, `imgsz=800`, `batch=12`

#### Giai đoạn 4 — đánh giá cuối

- evaluate trên `test`
- so với baseline cũ
- chỉ thay baseline nếu model mới thắng rõ ràng

### 18.2 CNN

- train lại lâu hơn nếu epoch hiện tại còn ít
- kiểm tra macro F1
- cải thiện dữ liệu severity nếu confusion giữa `minor` và `moderate` còn cao

### 18.3 XGBoost

- giữ vai trò base price predictor
- tune nhẹ nếu cần
- có thể thử feature fusion sau khi YOLO/CNN ổn định hơn

---

## 19. Tiêu chí thành công thực tế cho dự án

### YOLO

- có baseline ổn định
- test metric không tụt quá mạnh so với val
- lớp khó được cải thiện có ý nghĩa

### CNN

- phân biệt severity đủ tốt để dùng trong pricing adjustment

### XGBoost

- base price đủ ổn định để làm nền cho bước điều chỉnh

### Toàn pipeline

- có thể tạo ra **damage-aware adjusted price**
- giải thích được từng bước
- nhất quán trong báo cáo kỹ thuật

---

## 20. Tóm tắt nhanh

| Câu hỏi                             | Câu trả lời ngắn                                       |
| ----------------------------------- | ------------------------------------------------------ |
| Model YOLO nên dùng hiện tại?       | **YOLOv8s trên bộ dữ liệu cũ 6 lớp**                   |
| Metric train log lấy từ đâu?        | Từ **validation set**                                  |
| Test dùng để làm gì?                | Đánh giá cuối sau khi chốt model                       |
| Nên ưu tiên gì khi model gần nhau?  | **Cải thiện dữ liệu trước**                            |
| Class nào cần tập trung nhất?       | `crack`, `scratch`, `dent`                             |
| Nên tăng dữ liệu kiểu nào?          | Có chủ đích, không ngẫu nhiên                          |
| Có nên kiểm tra duplicate không?    | **Có, rất nên**                                        |
| Có nên fine-tune `best.pt`?         | Có, nhưng chỉ hiệu quả khi đi kèm data/setting tốt hơn |
| Có nên tăng learning rate?          | Không phải ưu tiên đầu tiên                            |
| Hướng cải thiện mạnh nhất hiện tại? | **Data-centric improvement**                           |

---

## 21. Kết luận

Trong dự án này, **training không chỉ là chạy thêm epoch**.  
Phần quan trọng nhất là:

- chọn đúng baseline
- đánh giá đúng bằng `train / val / test`
- hiểu khi nào nên chỉnh model
- và đặc biệt là ưu tiên **cải thiện dữ liệu** khi các model cho kết quả gần nhau

Baseline hiện tại đủ tốt để tiếp tục xây dựng pipeline.  
Các bước cải thiện tiếp theo nên đi theo hướng:

- relabel
- thêm dữ liệu khó có chủ đích
- hard negatives
- kiểm tra duplicate
- rồi mới train lại YOLO và cập nhật pipeline pricing
