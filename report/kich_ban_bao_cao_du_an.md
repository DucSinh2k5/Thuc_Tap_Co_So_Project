# Kịch Bản Báo Cáo Dự Án

## 0. Cách Dùng File Này

File này là ghi chú để em luyện nói và dùng khi báo cáo với thầy. Có thể dùng theo 3 mức:

- **Nói ngắn 7-10 phút:** đọc phần `Kịch bản nói 7-10 phút`.
- **Trình bày đầy đủ:** đi theo từng mục từ trên xuống.
- **Bị hỏi sâu:** xem phần `Nếu thầy hỏi`.

Thông điệp chính nên nhấn mạnh:

> Dự án của em là một hệ thống ước lượng giá xe cũ có xét đến tình trạng hư hỏng ngoại thất. Hệ thống không chỉ dự đoán giá từ thông tin dạng bảng, mà còn nhận ảnh xe, phát hiện hư hỏng, đánh giá mức độ hư hỏng và điều chỉnh giá cuối cùng.

---

## 1. Mở Đầu Báo Cáo

### Cần nói

Kính thưa thầy, trong đồ án này em xây dựng một hệ thống demo dự đoán giá xe cũ có xét đến tình trạng ngoại thất của xe.

Bài toán của em gồm hai nhóm dữ liệu chính:

- **Dữ liệu bảng:** thông tin xe như hãng xe, dòng xe, năm sản xuất, số km đã đi, hộp số, loại nhiên liệu, dung tích động cơ, công suất, số chỗ ngồi.
- **Dữ liệu ảnh:** ảnh xe dùng để phát hiện các hư hỏng như vết xước, vết móp, vết nứt, đèn vỡ, kính vỡ, lốp xẹp.

Đầu ra của hệ thống gồm:

- **Base price:** giá cơ sở được dự đoán từ thông tin xe bằng mô hình XGBoost.
- **Damage detection:** danh sách vùng hư hỏng trên ảnh bằng YOLO.
- **Severity classification:** mức độ hư hỏng `minor`, `moderate`, `severe` bằng CNN ConvNeXt-Tiny.
- **Final adjusted price:** giá cuối cùng sau khi trừ theo mức hư hỏng bằng tầng rule-based adjustment.

### Điểm cần nhấn mạnh

Dự án này không phải là một mô hình multimodal end-to-end học trực tiếp giá xe sau hư hỏng, vì hiện tại chưa có ground truth cho **final price after damage**.

Vì vậy em thiết kế theo hướng **hybrid pipeline**:

1. XGBoost học giá thị trường cơ sở từ dữ liệu bảng.
2. YOLO và CNN trích xuất tín hiệu hư hỏng từ ảnh.
3. Một tầng luật kết hợp các tín hiệu này để điều chỉnh giá.

---

## 2. Tổng Quan Kiến Trúc Hệ Thống

### Cần nói

Hệ thống của em có 3 nhánh chính:

| Nhánh | Nhiệm vụ | Mô hình/file chính | Đầu ra |
| --- | --- | --- | --- |
| Tabular pricing | Dự đoán giá cơ sở của xe | XGBoost, `Models/model.pkl` | `base_price` |
| Damage detection | Phát hiện vùng hư hỏng trên ảnh | YOLOv8s, `Models/best.pt` | class, confidence, bbox, area_ratio |
| Severity classification | Phân loại mức độ hư hỏng | ConvNeXt-Tiny, `Models/ConvNeXt.pkl` | minor/moderate/severe |
| Rule adjustment | Tính giá sau khi trừ hư hỏng | `dich_vu/dinh_gia.py` | `final_price` |

Luồng chạy trong app:

```text
Thông tin xe + ảnh upload
        |
        |-- app.py chuẩn hóa thông tin xe
        |
        |-- XGBoost dự đoán base price
        |
        |-- nếu có ảnh:
        |       |-- YOLO phát hiện hư hỏng
        |       |-- ConvNeXt phân loại mức độ
        |       |-- rule-based adjustment tính tiền trừ
        |
        |-- Streamlit hiển thị base price, damage deduction, final price
```

### File/hàm cần chỉ

- `app.py`
  - `chay_ung_dung()`: khởi tạo giao diện Streamlit.
  - `chay_pipeline(thong_tin_xe, danh_sach_anh)`: nối tất cả nhánh mô hình lại.
  - `chuan_hoa_thong_tin_xe(thong_tin_xe)`: biến input form thành schema đúng với model tabular.
- `dich_vu/dinh_gia.py`
  - `du_doan_gia_co_ban(thong_tin_xe)`: gọi XGBoost.
  - `tinh_dieu_chinh_gia(...)`: tính tiền trừ theo hư hỏng.
- `dich_vu/phat_hien_hu_hong.py`
  - `phat_hien_hu_hong(danh_sach_anh)`: chạy YOLO.
  - `ve_bbox_anh(...)`: vẽ bounding box để hiển thị.
- `dich_vu/muc_do_hu_hong.py`
  - `phan_loai_muc_do(danh_sach_anh)`: chạy ConvNeXt-Tiny.
  - `tong_hop_muc_do(...)`: tổng hợp số damage và mức độ.

---

## 3. Dữ Liệu Sử Dụng

### 3.1. Dữ liệu bảng cho bài toán giá xe

### Cần nói

Dữ liệu bảng được lấy từ bộ **used-cars-price-prediction** trên Kaggle. Các file trong dự án:

- `Datasets/train-dataset.csv`: dữ liệu train, khoảng 6019 dòng dữ liệu.
- `Datasets/test-dataset.csv`: dữ liệu test, khoảng 1234 dòng dữ liệu.
- `Datasets/train_cleaned.csv`: bản dữ liệu sau tiền xử lý.

Cột mục tiêu là:

- `Price`, sau khi rename thành `Gia_theo_lakh`.

Một số cột đầu vào:

- `Name` -> `Ten_xe`
- `Year` -> `Nam_san_xuat`
- `Kilometers_Driven` -> `Quang_duong_da_di(km)`
- `Fuel_Type` -> `Loai_nhien_lieu`
- `Transmission` -> `Hop_so`
- `Owner_Type` -> `Quyen_so_huu`
- `Mileage` -> `Muc_tieu_hao(km/l)`
- `Engine` -> `Dung_tich(cc)`
- `Power` -> `Cong_suat_toi_da`
- `Seats` -> `So_cho_ngoi`

### Khó khăn với dữ liệu bảng

- `New_Price` thiếu rất nhiều, khoảng 86% trong train.
- Các cột `Mileage`, `Engine`, `Power` có đơn vị dạng text như `18.2 kmpl`, `1199 CC`, `88.7 bhp`, cần tách số.
- Có giá trị bất thường như số km rất lớn, số ghế bằng 0, công suất `null bhp`.
- Cột tên xe có rất nhiều giá trị unique, nên cần gộp top xe/hãng xe để tránh quá nhiều category hiếm.

### 3.2. Dữ liệu ảnh cho YOLO

### Cần nói

Nhánh detection dùng dữ liệu từ bộ **car-damage-detection**, phần `CarDD_COCO`. Dữ liệu gốc ở định dạng COCO, sau đó được chuyển sang định dạng YOLOv8 thông qua Roboflow.

Ban đầu có khoảng **4000 ảnh**. Sau khi benchmark, em thấy các lớp khó là:

- `crack`
- `scratch`
- `dent`

Nên em bổ sung thêm **2307 ảnh** từ Roboflow Universe cho các lớp này. Dataset sau merge có khoảng **6307 ảnh**.

### 3.3. Dữ liệu ảnh cho severity classification

### Cần nói

Nhánh severity classification dùng ảnh full và nhãn:

- `minor`
- `moderate`
- `severe`

Điểm đáng chú ý là trong pipeline hiện tại, CNN nhận **ảnh full người dùng upload**, không nhận crop bbox từ YOLO. YOLO và CNN xử lý song song:

- YOLO cho biết có hư hỏng gì, ở đâu, diện tích bao nhiêu.
- CNN cho biết mức độ tổng quát của ảnh là nhẹ, vừa hay nặng.

---

## 4. Quy Trình Làm Việc Với Dữ Liệu Bảng

### Cần nói

Với nhánh tabular, em làm theo các bước:

1. **EDA trước xử lý**
   - Đọc dữ liệu, kiểm tra missing, kiểu dữ liệu, thống kê cơ bản.
   - File tham khảo: `tests/test_tabular/test-dataset-tabular.ipynb`, `src/EDA_before.py`, `Quan_sat/eda_before.txt`.

2. **Chuẩn hóa cột và làm sạch**
   - Rename cột sang tên rõ nghĩa bằng tiếng Việt.
   - Lọc các dòng không hợp lệ.
   - Tách số từ các cột có đơn vị.
   - Map category như hộp số, nhiên liệu, số lần sở hữu.
   - File: `src/load_data_and_cleaning.py`.

3. **Feature engineering**
   - Tạo `Tuoi_xe` từ năm sản xuất.
   - Tạo `Hang_xe` từ tên xe.
   - Tạo `Km_moi_nam`.
   - Tạo `Chay_nhieu`.
   - Tạo `log_Quang_duong_da_di(km)`.
   - Tạo `Top_xe` để gộp các dòng xe phổ biến, xe ít gặp đưa về `Other`.
   - File: `src/feature_engineering.py`.

4. **Xử lý missing và outlier**
   - Numeric dùng median.
   - Categorical dùng mode/Unknown.
   - Outlier được clip theo IQR.
   - Quan trọng: các tham số median, imputer, top category, outlier bounds được fit trên train rồi dùng lại cho val/test để tránh data leakage.

5. **Tiền xử lý cho model**
   - Numeric: impute median.
   - Categorical: `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`.
   - File: `src/preprocessing.py`.

6. **Train và đánh giá**
   - Chia train/validation 80/20.
   - So sánh baseline, Random Forest và XGBoost.
   - Chọn XGBoost để triển khai.
   - File: `src/train_and_evaluate.py`, `src/main.py`.

### Các hàm quan trọng

| File | Hàm | Vai trò |
| --- | --- | --- |
| `src/load_data_and_cleaning.py` | `doi_ten_cot(df)` | Rename cột về schema thống nhất |
| `src/load_data_and_cleaning.py` | `loai_bo_hang_ban(df)` | Lọc dòng không hợp lệ |
| `src/load_data_and_cleaning.py` | `chuyen_cot_sang_so(df)` | Tách số từ cột có đơn vị |
| `src/load_data_and_cleaning.py` | `chuyen_cot_sang_category(df)` | Mã hóa nhiên liệu, hộp số, sở hữu |
| `src/feature_engineering.py` | `tao_moi_feature(df, km_median=None)` | Tạo tuổi xe, hãng xe, km mỗi năm, log km |
| `src/feature_engineering.py` | `xu_ly_gia_tri_thieu(df, imputers=None)` | Fit/transform missing value |
| `src/feature_engineering.py` | `gioi_han_xe(...)`, `gioi_han_hang_xe(...)` | Gộp category hiếm |
| `src/feature_engineering.py` | `xu_ly_outlier(df, bounds=None)` | Clip outlier theo IQR |
| `src/preprocessing.py` | `tien_xu_ly(num, cat)` | Tạo ColumnTransformer |
| `src/train_and_evaluate.py` | `compare_models(...)` | So sánh RF và XGBoost |
| `src/train_and_evaluate.py` | `train_model(...)` | Train XGBoost final |
| `src/train_and_evaluate.py` | `feature_importance_report(...)` | Xuất feature importance |
| `src/train_and_evaluate.py` | `save(...)` | Lưu model deploy vào `Models/model.pkl` |

---

## 5. Kết Quả Nhánh Tabular

### Cần nói

Kết quả baseline:

| Baseline | RMSE | MAE | R2 |
| --- | ---: | ---: | ---: |
| Mean baseline | 5.7028 | 4.6119 | -0.0044 |
| Median baseline | 5.9453 | 4.0818 | -0.0916 |

Kết quả so sánh model:

| Model | RMSE | MAE | R2 validation |
| --- | ---: | ---: | ---: |
| Random Forest | 1.6074 | 0.9962 | 0.9202 |
| XGBoost | 1.4144 | 0.8856 | 0.9382 |

Em chọn **XGBoost** vì:

- R2 validation cao hơn Random Forest.
- RMSE và MAE thấp hơn.
- Phù hợp với dữ liệu bảng có cả numeric và categorical đã encode.
- Dễ lưu và tích hợp vào app bằng `joblib`.

### Feature quan trọng

Theo `feature_importance_xgb.csv`, các feature quan trọng nhất:

| Feature | Importance |
| --- | ---: |
| `Cong_suat_toi_da` | 0.3664 |
| `Hop_so` | 0.1694 |
| `Dung_tich(cc)` | 0.1624 |
| `Tuoi_xe` | 0.1076 |
| `So_cho_ngoi` | 0.0557 |

Theo permutation importance:

| Feature | Importance mean |
| --- | ---: |
| `Cong_suat_toi_da` | 0.5682 |
| `Tuoi_xe` | 0.2559 |
| `Dung_tich(cc)` | 0.1068 |
| `Hang_xe` | 0.0635 |
| `Hop_so` | 0.0361 |

### Câu nói gợi ý

> Kết quả này cho thấy mô hình học được các yếu tố hợp lý về mặt thực tế: công suất, tuổi xe, dung tích, hãng xe và hộp số đều là các yếu tố ảnh hưởng mạnh đến giá xe cũ.

---

## 6. Quy Trình Làm Việc Với YOLO Damage Detection

### Cần nói

Với nhánh detection, mục tiêu của em là phát hiện các vùng hư hỏng trên ảnh xe. Em dùng YOLO vì đây là mô hình object detection nhanh, phù hợp với ứng dụng demo cần inference trực tiếp.

Em đã thử các hướng:

- YOLOv8n: nhẹ, nhanh, dùng làm baseline.
- YOLOv8s: cân bằng tốt hơn giữa tốc độ và độ chính xác.
- YOLOv8m: lớn hơn nhưng chi phí cao, không phù hợp bằng YOLOv8s cho demo.
- Faster R-CNN ResNet50-FPN: dùng để so sánh phụ.

Sau benchmark, em chọn **YOLOv8s** và cải thiện bằng cách mở rộng dataset, tập trung vào các lớp khó `crack`, `scratch`, `dent`.

### Cấu hình YOLO chính

| Thông số | Giá trị |
| --- | --- |
| Model | YOLOv8s |
| Dataset | `merge_Data` |
| Ảnh gốc | khoảng 4000 ảnh |
| Ảnh bổ sung | 2307 ảnh |
| Tổng ảnh | khoảng 6307 ảnh |
| Epochs | 100 |
| Batch size | 16 |
| Image size | 640 |
| Checkpoint app | `Models/best.pt` |
| Inference conf | 0.25 |
| Inference IoU | 0.45 |

### Kết quả YOLO

Kết quả từ `Quan_sat/yolo_car_report/results.csv`:

| Mốc | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| Epoch có mAP50 cao nhất | ~0.801 | ~0.682 | ~0.726 | ~0.568 |
| Epoch có mAP50-95 cao nhất | - | - | - | ~0.572 |

So sánh phụ với Faster R-CNN:

| Model | mAP@0.5 | mAP@0.5:0.95 | mAP@0.75 |
| --- | ---: | ---: | ---: |
| YOLOv8s merge_Data | 0.726 | 0.568 | - |
| Faster R-CNN ResNet50-FPN | 0.306 | 0.128 | 0.084 |

### Câu nói gợi ý

> Em không chỉ chọn model theo một lần train, mà có benchmark nhiều biến thể. YOLOv8s được chọn vì cân bằng giữa độ chính xác, tốc độ inference và kích thước mô hình. Việc bổ sung dữ liệu cho `crack`, `scratch`, `dent` là do ba lớp này thường nhỏ, mảnh và dễ bị bỏ sót hơn.

### File/hàm cần chỉ

- `dich_vu/phat_hien_hu_hong.py`
  - `DUONG_DAN_MO_HINH = Models/best.pt`
  - `KICH_THUOC_ANH_YOLO = 640`
  - `NGUONG_CONF = 0.25`
  - `NGUONG_IOU = 0.45`
  - `phat_hien_hu_hong(danh_sach_anh)`
  - `ve_bbox_anh(danh_sach_anh, danh_sach_phat_hien)`
- `Quan_sat/yolo_car_report/`
  - `results.csv`
  - `confusion_matrix.png`
  - `BoxPR_curve.png`
  - `results.png`

---

## 7. Quy Trình Làm Việc Với CNN Severity Classification

### Cần nói

Nhánh CNN có nhiệm vụ phân loại mức độ hư hỏng của ảnh thành 3 mức:

- `minor`
- `moderate`
- `severe`

Em thử nhiều backbone theo hướng transfer learning. Quy trình train gồm:

1. Phase 1: đóng băng backbone, train classifier/head.
2. Phase 2: fine-tune các block cuối của backbone và classifier để mô hình thích nghi với dữ liệu hư hỏng xe.

### Kết quả các backbone CNN

| Notebook | Backbone | Best validation metric | Test accuracy | Test macro F1 |
| --- | --- | ---: | ---: | ---: |
| `resnet_18_new.ipynb` | ResNet18 | val_acc = 0.8209 | 0.6564 | 0.6490 |
| `EfficientNet_B0.ipynb` | EfficientNet-B0 | val_macro_f1 = 0.7945 | 0.6821 | 0.6817 |
| `Efficient_B2.ipynb` | EfficientNet-B2 | val_macro_f1 = 0.8022 | 0.6974 | 0.6954 |
| `ResNet50.ipynb` | ResNet50 | val_acc = 0.8128 | 0.7026 | 0.7043 |
| `ConvNeXt_Tiny.ipynb` | ConvNeXt-Tiny | val_macro_f1 = 0.8342 | 0.7077 | 0.7084 |

Em chọn **ConvNeXt-Tiny** vì:

- Validation macro F1 cao nhất: 0.8342.
- Test accuracy cao nhất: 0.7077.
- Test macro F1 cao nhất: 0.7084.
- Tổng quát hóa tốt hơn ResNet18 trong bài toán có nhiều chi tiết nhỏ và ranh giới nhãn mờ.

### Demo định tính

Trong báo cáo có một ví dụ cùng một ảnh test:

| Model | Dự đoán | Xác suất |
| --- | ---: | ---: |
| ResNet18 | SEVERE | 60.61% |
| EfficientNet-B0 | MINOR | 42.60% |
| EfficientNet-B2 | SEVERE | 42.31% |
| ResNet50 | SEVERE | 40.88% |
| ConvNeXt-Tiny | MODERATE | 44.35% |

Nhãn đúng của ảnh đó là `moderate`, nên ConvNeXt-Tiny là mô hình dự đoán đúng trong ví dụ này.

### File/hàm cần chỉ

- `dich_vu/muc_do_hu_hong.py`
  - `CAC_MUC_DO = ["minor", "moderate", "severe"]`
  - `DUONG_DAN_MO_HINH = Models/ConvNeXt.pkl`
  - `BIEN_DOI_DANH_GIA = ConvNeXt_Tiny_Weights.DEFAULT.transforms()`
  - `_tao_convnext_tiny()`
  - `_tai_mo_hinh_muc_do()`
  - `phan_loai_muc_do(danh_sach_anh)`
  - `tong_hop_muc_do(danh_sach_phat_hien, danh_sach_muc_do)`
- `train/ConvNeXt_Tiny.ipynb`: notebook train model chính.

### Giới hạn cần nói rõ

> Accuracy của CNN khoảng 70.77%, nên em không xem severity là nhãn tuyệt đối chắc chắn. Em xem nó như một tín hiệu hỗ trợ cho tầng điều chỉnh giá. Lớp `moderate` khó nhất vì nằm giữa `minor` và `severe`.

---

## 8. Tầng Điều Chỉnh Giá Theo Hư Hỏng

### Cần nói

Sau khi có:

- base price từ XGBoost,
- class/area/confidence từ YOLO,
- severity từ ConvNeXt,

em dùng một tầng rule-based để tính tỉ lệ trừ giá. Lý do dùng rule-based là vì hiện tại chưa có nhãn ground truth cho giá xe sau hư hỏng.

Công thức tổng quát trong code:

```text
diem_hu_hong = min(MUC_GIAM_TOI_DA, tong_diem * HE_SO_DIEM_SANG_TI_LE)
tien_tru = gia_co_ban * diem_hu_hong
gia_sau = gia_co_ban - tien_tru
```

Thông số chính:

| Thông số | Giá trị | Ý nghĩa |
| --- | ---: | --- |
| `MUC_GIAM_TOI_DA` | 0.3 | Trừ tối đa 30% giá trị xe |
| `HE_SO_DIEM_SANG_TI_LE` | 0.012 | Đổi damage score sang deduction rate |
| `LAKH_INR_SANG_VND` | 30,000,000 | Đổi giá dự đoán từ lakh sang VND |

Trọng số theo lớp hư hỏng:

| Lớp | Trọng số |
| --- | ---: |
| `scratch` | 0.75 |
| `dent` | 1.0 |
| `crack` | 1.15 |
| `tire_flat` | 1.1 |
| `glass_broken` | 1.25 |
| `lamp_broken` | 1.35 |

Ngoài ra, điểm trừ còn phụ thuộc vào:

- `severity_score`: minor = 1, moderate = 2, severe = 3.
- `area_ratio`: bbox chiếm bao nhiêu diện tích ảnh.
- `confidence`: độ tin cậy của YOLO.
- `so_lan_da_gap`: nếu cùng một lớp lặp lại nhiều lần thì có hệ số giảm dần để tránh trừ quá mạnh.

### Câu nói gợi ý

> Tầng rule-based này giúp em kết hợp được đầu ra của các mô hình riêng lẻ thành một kết quả có ý nghĩa với bài toán: giá cuối cùng. Nó cũng minh bạch hơn, vì em có thể giải thích tại sao giá bị trừ: do có bao nhiêu damage, lớp nào, mức độ nào và diện tích ảnh hưởng bao nhiêu.

### File/hàm cần chỉ

- `dich_vu/dinh_gia.py`
  - `TRONG_SO_LOP`
  - `HE_SO_DIEM_SANG_TI_LE`
  - `MUC_GIAM_TOI_DA`
  - `du_doan_gia_co_ban(thong_tin_xe)`
  - `tinh_dieu_chinh_gia(gia_co_ban, danh_sach_phat_hien, danh_sach_muc_do)`

---

## 9. Tích Hợp Ứng Dụng Streamlit

### Cần nói

Sau khi có các mô hình riêng lẻ, em tích hợp thành một app Streamlit để demo.

Người dùng có thể:

1. Chọn hãng xe và dòng xe.
2. Nhập thông tin xe: năm sản xuất, số km, hộp số, nhiên liệu, số ghế, dung tích động cơ, công suất.
3. Upload ảnh xe nếu muốn tính giá có xét damage.
4. Bấm `Analyze Car`.
5. Hệ thống hiển thị:
   - ảnh gốc và ảnh có bounding box,
   - bảng detection,
   - bảng severity,
   - base price,
   - damage deduction,
   - final adjusted price.

### File giao diện

| File | Vai trò |
| --- | --- |
| `app.py` | Entry point, nối pipeline |
| `giao_dien/bo_cuc.py` | CSS và layout đầu trang |
| `giao_dien/thanh_phan.py` | Form input, upload ảnh, hiển thị kết quả |
| `tien_ich/du_lieu_mau.py` | Giá trị mặc định, danh sách fuel/transmission |
| `tien_ich/dinh_dang.py` | Format VND, phần trăm |
| `tien_ich/trang_thai.py` | Session state Streamlit |

### Demo nói trực tiếp

Khi demo, nên nói theo thứ tự:

1. Đây là form thông tin xe. Brand/model được lấy từ dataset để giảm sai lệch category với model.
2. Khi không upload ảnh, hệ thống chỉ dự đoán giá cơ sở bằng XGBoost.
3. Khi upload ảnh, app sẽ chạy thêm YOLO và ConvNeXt.
4. YOLO vẽ bbox và trả bảng class, confidence, area ratio.
5. ConvNeXt trả severity của ảnh.
6. Cuối cùng, rule-based adjustment tính tiền trừ và giá sau điều chỉnh.

Lệnh chạy:

```powershell
streamlit run app.py
```

---

## 10. Những Gì Đã Đạt Được

### Cần nói

Qua dự án, em đã đạt được các kết quả sau:

1. **Xây dựng pipeline tabular hoàn chỉnh**
   - EDA, cleaning, feature engineering, preprocessing, train/evaluate, save model.
   - XGBoost đạt R2 validation **0.9382**.

2. **Xây dựng nhánh damage detection**
   - Benchmark YOLOv8n, YOLOv8s, YOLOv8m.
   - Mở rộng dataset có mục tiêu cho `crack`, `scratch`, `dent`.
   - YOLOv8s trên `merge_Data` đạt mAP50 khoảng **0.726**.

3. **Xây dựng nhánh severity classification**
   - Thử nhiều backbone CNN: ResNet18, EfficientNet-B0, EfficientNet-B2, ResNet50, ConvNeXt-Tiny.
   - Chọn ConvNeXt-Tiny với test accuracy **0.7077**, test macro F1 **0.7084**.

4. **Tích hợp thành ứng dụng demo**
   - Streamlit app nhận thông tin xe và ảnh.
   - Hiển thị detection, severity, base price và final price.

5. **Có báo cáo và artifact**
   - `report/dataset_report.md`
   - `report/training_report.md`
   - `Quan_sat/model_comparison_report.txt`
   - `Quan_sat/yolo_car_report/`
   - `Models/model.pkl`, `Models/best.pt`, `Models/ConvNeXt.pkl`

---

## 11. Khó Khăn Trong Quá Trình Làm

### Bảng khó khăn và cách xử lý

| Khó khăn | Mô tả | Cách em xử lý |
| --- | --- | --- |
| Dữ liệu bảng có nhiều đơn vị text | `Mileage`, `Engine`, `Power` không phải số thuần | Viết hàm tách số trong `chuyen_cot_sang_so` |
| Missing value | `New_Price` thiếu rất nhiều, một số cột như `Power`, `Engine`, `Seats` thiếu | Loại cột quá thiếu, impute median/mode |
| Category quá nhiều | Tên xe có nhiều giá trị unique | Tạo `Top_xe`, `Hang_xe`, gộp giá trị hiếm về `Other` |
| Outlier | Số km, giá, công suất có giá trị cực đoan | Clip theo IQR trong `xu_ly_outlier` |
| Tránh data leakage | Dễ bị fit imputer/top category trên cả test | Fit trên train, reuse cho val/test |
| Damage nhỏ, khó detect | `scratch`, `dent`, `crack` nhỏ và dễ bị ảnh hưởng bởi ánh sáng | Bổ sung 2307 ảnh có chủ đích cho các lớp này |
| Annotation noise | Test set có thể thiếu nhãn, làm mô hình phát hiện đúng bị tính thành false positive | Không chỉ nhìn mAP, có xem inference định tính |
| Severity khó | `moderate` nằm giữa minor/severe, dễ nhầm | Benchmark nhiều backbone, chọn ConvNeXt-Tiny |
| Không có nhãn final price sau damage | Không thể train supervised final adjusted price | Dùng rule-based adjustment minh bạch |
| Tích hợp model | Cần đồng bộ schema input app với schema train model | Viết `chuan_hoa_thong_tin_xe` trong `app.py` |

### Câu nói gợi ý

> Khó khăn lớn nhất của em không chỉ là train model, mà là làm sao kết hợp ba bài toán khác nhau thành một hệ thống có thể demo được. Dữ liệu giá xe và dữ liệu damage không đồng bộ theo từng chiếc xe, nên em phải thiết kế theo hướng modular pipeline thay vì end-to-end.

---

## 12. Hạn Chế Hiện Tại

### Cần nói

Dự án hiện tại vẫn có một số hạn chế:

1. **Chưa có dataset multimodal đồng bộ**
   - Ảnh damage và dữ liệu giá xe đến từ các nguồn khác nhau.
   - Chưa có mối quan hệ trực tiếp theo từng chiếc xe giữa damage và giá bán.

2. **Chưa có ground truth cho final adjusted price**
   - Giá cuối cùng sau khi xét damage hiện được tính bằng rule-based.
   - Chưa phải mô hình supervised học từ nhãn thật.

3. **CNN severity còn giới hạn**
   - Test accuracy khoảng 70.77%.
   - Lớp `moderate` còn nhập nhằng.

4. **Severity hiện chạy trên ảnh full**
   - Nếu ảnh có nhiều vùng hư hỏng, một severity cho cả ảnh có thể chưa chi tiết.
   - Hướng cải tiến là crop từng bbox từ YOLO rồi phân loại severity riêng cho từng damage.

5. **App mới ở mức MVP**
   - Vẫn cần đóng gói path/model tốt hơn nếu chuyển sang máy khác.
   - Cần thêm unit test/automation test thật sự.

---

## 13. Hướng Phát Triển

### Cần nói

Nếu tiếp tục phát triển, em sẽ làm các hướng:

1. **Cải thiện dữ liệu**
   - Thu thập dữ liệu xe có ảnh hư hỏng và giá thực tế sau khi định giá.
   - Bổ sung hard negatives và kiểm tra duplicate/label noise.

2. **Cải thiện severity**
   - Cắt crop bbox từ YOLO rồi cho CNN phân loại từng vùng damage.
   - Hoặc dùng multi-task model vừa detect vừa đánh severity.

3. **Cải thiện price adjustment**
   - Nếu có nhãn giá sau hư hỏng, có thể train model học trực tiếp tỉ lệ trừ giá.
   - Hiện tại rule-based minh bạch, nhưng cần dữ liệu thật để hiệu chỉnh trọng số.

4. **Cải thiện deploy**
   - Bỏ hard-code path.
   - Đóng gói config.
   - Thêm test tự động cho preprocessing và prediction.
   - Cache model và tối ưu inference.

---

## 14. Nếu Thầy Hỏi

### Vì sao chọn XGBoost thay vì Random Forest?

Em có so sánh trên validation. Random Forest đạt R2 0.9202, còn XGBoost đạt R2 0.9382, RMSE và MAE cũng thấp hơn. Ngoài ra XGBoost phù hợp với dữ liệu bảng đã xử lý và dễ triển khai trong pipeline.

### Vì sao dùng OrdinalEncoder cho categorical?

Vì model tree-based như XGBoost/Random Forest có thể làm việc với số nguyên encode từ category. Em dùng `handle_unknown="use_encoded_value", unknown_value=-1` để khi app gặp category mới không bị lỗi.

### Em có tránh data leakage không?

Có. Trong `src/main.py`, em chia train/validation trước. Các thông số như median km, imputer, top category, outlier bounds được fit trên train rồi mới reuse cho validation/test.

### Vì sao không dùng một mô hình end-to-end cho cả ảnh và bảng?

Vì hiện tại dữ liệu ảnh damage và dữ liệu giá xe không đồng bộ theo từng chiếc xe, và không có nhãn giá sau hư hỏng. Nếu train end-to-end sẽ không có target đúng. Vì vậy em chọn thiết kế modular: XGBoost dự đoán giá cơ sở, YOLO/CNN trích xuất damage, rule-based tính điều chỉnh.

### Vì sao chọn YOLOv8s?

YOLOv8n nhanh nhưng khả năng học damage nhỏ hạn chế. YOLOv8m lớn hơn nhưng chi phí cao, không phù hợp bằng cho demo. YOLOv8s cân bằng giữa độ chính xác và tốc độ, đạt mAP50 khoảng 0.726 trên run `merge_Data`.

### Vì sao bổ sung dữ liệu cho `crack`, `scratch`, `dent`?

Vì đây là các lớp khó: vùng hư hỏng nhỏ, mảnh, dễ bị ảnh hưởng bởi ánh sáng và góc chụp. Bổ sung có mục tiêu giúp mô hình có thêm mẫu học cho các trường hợp khó thay vì tăng dữ liệu đại trà.

### Vì sao chọn ConvNeXt-Tiny?

Em benchmark 5 backbone. ConvNeXt-Tiny có validation macro F1 0.8342, test accuracy 0.7077 và test macro F1 0.7084, tốt nhất trong các model đã thử.

### Tại sao CNN phân loại ảnh full mà không phân loại từng bbox?

Trong bản hiện tại, để phù hợp với dữ liệu severity và app demo, em dùng ảnh full. Tuy nhiên em nhận thức đây là hạn chế. Hướng phát triển tốt hơn là crop từng bbox từ YOLO rồi cho CNN phân loại severity từng damage.

### Rule-based adjustment có chủ quan không?

Có một phần chủ quan, vì chưa có ground truth giá sau hư hỏng. Tuy nhiên cách này minh bạch và giải thích được: mỗi lớp damage có trọng số, severity có điểm, bbox có diện tích, confidence có hệ số. Khi có dữ liệu thật, các hệ số này có thể được học hoặc hiệu chỉnh lại.

---

## 15. Kịch Bản Nói 7-10 Phút

### 1 phút - Giới thiệu bài toán

Kính thưa thầy, dự án của em là hệ thống demo dự đoán giá xe cũ có xét đến tình trạng hư hỏng ngoại thất. Đầu vào gồm thông tin bảng của xe và ảnh xe. Đầu ra là giá cơ sở, các hư hỏng phát hiện trên ảnh, mức độ hư hỏng và giá sau điều chỉnh.

### 1 phút - Kiến trúc

Hệ thống gồm ba nhánh: XGBoost cho giá cơ sở, YOLOv8s cho phát hiện hư hỏng, ConvNeXt-Tiny cho phân loại mức độ. Cuối cùng em dùng rule-based adjustment để tính giá cuối cùng. Em chọn hướng modular vì dữ liệu giá và dữ liệu ảnh không đồng bộ theo từng xe và chưa có nhãn giá sau hư hỏng.

### 2 phút - Tabular pipeline

Với dữ liệu bảng, em làm EDA, rename cột, tách số từ các cột có đơn vị, xử lý missing, tạo feature mới như tuổi xe, hãng xe, km mỗi năm, log km và Top_xe. Sau đó em chia train/validation, preprocess numeric/categorical và train model. Kết quả XGBoost đạt R2 validation 0.9382, tốt hơn Random Forest 0.9202, nên em chọn XGBoost làm model deploy.

### 2 phút - Image pipeline

Với damage detection, em benchmark YOLOv8n, YOLOv8s, YOLOv8m. YOLOv8s cân bằng tốt nhất nên được chọn. Sau đó em mở rộng dataset từ khoảng 4000 ảnh lên khoảng 6307 ảnh bằng cách bổ sung 2307 ảnh cho các lớp khó `crack`, `scratch`, `dent`. Kết quả YOLOv8s trên `merge_Data` đạt mAP50 khoảng 0.726.

Với severity classification, em thử ResNet18, EfficientNet-B0, EfficientNet-B2, ResNet50 và ConvNeXt-Tiny. ConvNeXt-Tiny tốt nhất với test accuracy 0.7077 và macro F1 0.7084, nên em dùng làm model chính.

### 1 phút - Tích hợp app

Em tích hợp các thành phần vào Streamlit. Người dùng nhập thông tin xe và upload ảnh. App chuẩn hóa input, dự đoán base price bằng XGBoost, chạy YOLO để vẽ bbox, chạy ConvNeXt để lấy severity, rồi tính damage deduction và final adjusted price.

### 1 phút - Khó khăn

Khó khăn lớn nhất là dữ liệu không đồng bộ và không có nhãn giá sau damage. Vì vậy em không thể train một model end-to-end, mà phải thiết kế hybrid pipeline. Ngoài ra dữ liệu bảng có missing/outlier/đơn vị text, còn dữ liệu ảnh có class imbalance, damage nhỏ và label noise.

### 1 phút - Kết luận

Kết quả cuối cùng là em xây dựng được một MVP hoàn chỉnh: có pipeline train tabular, có mô hình detection, có mô hình severity, có rule adjustment và có app demo. Hướng phát triển tiếp theo là thu thập dữ liệu multimodal đồng bộ hơn, phân loại severity theo crop bbox và học trực tiếp tỉ lệ trừ giá khi có nhãn thực tế.

---

## 16. Các File Nên Mở Khi Báo Cáo

Nếu cần demo code, nên mở theo thứ tự:

1. `app.py`
   - Chỉ hàm `chay_pipeline`.
   - Nói: đây là nơi kết nối 3 nhánh mô hình.

2. `src/main.py`
   - Chỉ split train/validation và flow train tabular.
   - Nói: đây là pipeline train XGBoost.

3. `src/feature_engineering.py`
   - Chỉ các feature `Tuoi_xe`, `Hang_xe`, `Km_moi_nam`, `Chay_nhieu`.

4. `dich_vu/phat_hien_hu_hong.py`
   - Chỉ `phat_hien_hu_hong` và thông số YOLO.

5. `dich_vu/muc_do_hu_hong.py`
   - Chỉ ConvNeXt-Tiny và 3 class severity.

6. `dich_vu/dinh_gia.py`
   - Chỉ `TRONG_SO_LOP`, `MUC_GIAM_TOI_DA`, `tinh_dieu_chinh_gia`.

7. `Quan_sat/model_comparison_report.txt`
   - Chỉ bảng XGBoost vs Random Forest.

8. `Quan_sat/yolo_car_report/results.png` hoặc `results.csv`
   - Chỉ kết quả train YOLO.

9. `report/training_report.md`
   - Chỉ bảng CNN và kết quả chọn ConvNeXt-Tiny.

---

## 17. Câu Kết

Em có thể kết thúc bằng câu này:

> Tổng kết lại, dự án của em đã xây dựng được một pipeline hybrid cho bài toán định giá xe cũ có xét đến hư hỏng ngoại thất. Phần giá xe được học bằng XGBoost, phần hình ảnh được xử lý bằng YOLOv8s và ConvNeXt-Tiny, sau đó kết hợp bằng một tầng điều chỉnh giá minh bạch. Dự án vẫn còn hạn chế về dữ liệu multimodal và nhãn giá sau hư hỏng, nhưng đã đạt được mục tiêu xây dựng một hệ thống demo hoàn chỉnh, có kết quả định lượng và có khả năng mở rộng tiếp.

