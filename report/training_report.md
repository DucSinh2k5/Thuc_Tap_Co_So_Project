# Training Report

## 1. Mục tiêu báo cáo

Tài liệu này ghi lại các lần huấn luyện mô hình ảnh trong pipeline dự đoán giá xe cũ, gồm **YOLOv8** cho bài toán **car damage detection** và **CNN** cho bài toán **severity classification**. Nhánh ảnh có nhiệm vụ trích xuất các tín hiệu thị giác như:

- số lượng hư hỏng theo lớp,
- diện tích vùng hư hỏng,
- mức độ hư hỏng từ mô hình **severity classification**,
- tín hiệu cho bước **price adjustment** trong tầng hybrid phía sau.

Báo cáo này tập trung vào các thực nghiệm của phần **object detection** và **CNN classification**, bao gồm:

- benchmark nhiều biến thể YOLOv8 trên dataset ban đầu,
- thử nghiệm mở rộng dataset theo hướng **merge_Data**, bổ sung 2307 ảnh cho các lớp khó `crack`, `scratch`, `dent`,
- benchmark các backbone CNN cho severity classification gồm ResNet18, ResNet50, EfficientNet-B0, EfficientNet-B2 và ConvNeXt-Tiny,
- thử nghiệm Faster R-CNN ResNet50-FPN ở mức so sánh phụ,
- quyết định chọn checkpoint YOLO và CNN chính thức để dùng cho pipeline hiện tại.

---

## 2. Bối cảnh thực nghiệm

### 2.1 Dataset ban đầu

Dataset detection ban đầu có khoảng **4000 ảnh** và được tổ chức theo định dạng YOLO với 6 lớp:

- `crack`
- `dent`
- `glass shatter`
- `lamp broken`
- `scratch`
- `tire flat`

Split dùng trong các run baseline ban đầu:

- `train`: 2800 ảnh
- `val`: 800 ảnh
- `test`: phần còn lại, giữ riêng để đánh giá cuối

Run `s_89` được xem là một trong các baseline chính trên phiên bản dataset gốc này.

### 2.2 Dataset merge_Data

Sau khi đánh giá baseline, dự án tạo thêm phiên bản dataset **merge_Data** bằng cách bổ sung **2307 ảnh** vào dữ liệu ban đầu. Dữ liệu bổ sung không được thêm đều cho tất cả các nhãn, mà tập trung vào ba lớp yếu nhất:

- `crack`
- `scratch`
- `dent`

Ba lớp này được ưu tiên vì thường có vùng hư hỏng nhỏ, mảnh, dễ bị ảnh hưởng bởi ánh sáng, màu sơn, góc chụp và chất lượng annotation. Sau khi mở rộng, tổng số ảnh của dataset tăng từ khoảng **4000 ảnh** lên khoảng **6307 ảnh**.

### 2.3 Môi trường huấn luyện

Các thực nghiệm YOLO được train trên môi trường dùng GPU **Tesla T4 (~15GB VRAM)** với:

- Python `3.12.x`
- PyTorch `2.10.0+cu128`
- Ultralytics `8.4.x`

### 2.4 Cấu hình baseline chung

Các cấu hình phổ biến đã dùng trong giai đoạn benchmark gồm:

- `imgsz = 640`
- `batch = 16`
- `pretrained = True`
- `optimizer = auto` hoặc `AdamW` do Ultralytics tự chọn
- `device = 0`

---

## 3. Benchmark trên dataset ban đầu

### 3.1 Các mô hình đã thử

Trên dataset ban đầu, dự án đã benchmark ba biến thể:

- `YOLOv8n`
- `YOLOv8s`
- `YOLOv8m`

### 3.2 Kết quả tổng hợp từ các run nền tảng

| Run / Model | Dataset | Cấu hình chính | Vai trò trong dự án |
| --- | --- | --- | --- |
| YOLOv8n | Dataset gốc khoảng 4000 ảnh | `imgsz=640`, `batch=16` | Baseline nhẹ, tốc độ tốt nhưng năng lực học đặc trưng hư hỏng còn hạn chế. |
| YOLOv8s (`s_89`) | Dataset gốc khoảng 4000 ảnh | `epochs=100`, `imgsz=640`, `batch=16` | Baseline YOLOv8s chính trước khi merge dữ liệu; mAP@0.5 ghi nhận khoảng 0.716 trong báo cáo thử nghiệm cũ. |
| YOLOv8m | Dataset gốc khoảng 4000 ảnh | `imgsz=640`, `batch=16` | Mô hình lớn hơn, chi phí tính toán cao hơn nên không phù hợp bằng YOLOv8s cho demo Streamlit. |

### 3.3 Nhận xét từ benchmark ban đầu

- `YOLOv8n` là mô hình nhẹ nhất, phù hợp làm baseline nhẹ nhưng chưa phải lựa chọn tốt nhất về tổng thể.
- `YOLOv8s` cho cân bằng tốt nhất giữa độ chính xác, tốc độ suy luận và kích thước mô hình.
- `YOLOv8m` chỉ cải thiện rất ít so với `YOLOv8s`, nhưng chi phí tính toán cao hơn rõ rệt.
- Vì vậy, dự án giữ kiến trúc YOLOv8s và tiếp tục cải thiện theo hướng dữ liệu, thay vì chuyển sang mô hình lớn hơn.

### 3.4 Kết luận giai đoạn benchmark ban đầu

Sau benchmark trên dataset ban đầu, `YOLOv8s` được chọn làm ứng viên mạnh nhất để tiếp tục phát triển.

---

## 4. Thử nghiệm mở rộng dữ liệu merge_Data

### 4.1 Lý do mở rộng dữ liệu

Các lớp khó của dataset ban đầu là:

- `crack`
- `scratch`
- `dent`

Đây là các lớp có xu hướng khó detect hơn do:

- vùng hỏng nhỏ hoặc mảnh,
- dễ bị ảnh hưởng bởi phản sáng, bóng đổ và góc chụp,
- nhãn dễ thiếu nhất quán hơn các lớp dễ như `glass shatter` hoặc `lamp broken`.

Vì vậy, sau giai đoạn benchmark ban đầu, dự án đã thử **bổ sung thêm dữ liệu ảnh cho ba lớp này** thay vì tăng dữ liệu đồng đều cho tất cả các nhãn. Mục tiêu của bước này là cải thiện khả năng nhận diện các dạng hư hỏng nhỏ và khó quan sát trong điều kiện ảnh thực tế.

### 4.2 Cấu hình train trên merge_Data

Mô hình mới vẫn giữ nguyên cấu hình của baseline YOLOv8s:

- **model:** `YOLOv8s`
- **dataset:** `merge_Data`
- **số ảnh gốc:** khoảng 4000 ảnh
- **số ảnh bổ sung:** 2307 ảnh cho `crack`, `scratch`, `dent`
- **epochs:** `100`
- **batch size:** `16`
- **imgsz:** `640`
- **pretrained:** `True`

Việc giữ nguyên kiến trúc và hyperparameter giúp so sánh rõ hơn tác động của việc thay đổi dataset. Nói cách khác, sự khác biệt chính giữa baseline `s_89` và mô hình hiện tại nằm ở dữ liệu huấn luyện, không phải do thay đổi cấu hình train.

### 4.3 Kết quả của run merge_Data

Kết quả trong `Quan_sat/yolo_car_report/results.csv` cho thấy:

- epoch có **mAP@0.5 cao nhất** là epoch 79, với Precision ≈ 0.801, Recall ≈ 0.682, mAP@0.5 ≈ 0.726 và mAP@0.5:0.95 ≈ 0.568.
- epoch có **mAP@0.5:0.95 cao nhất** là epoch 75, với mAP@0.5:0.95 ≈ 0.572.
- các epoch cuối vẫn duy trì mAP@0.5 quanh 0.715-0.720 và mAP@0.5:0.95 quanh 0.562-0.565, cho thấy quá trình train tương đối ổn định.

### 4.4 Diễn giải

Việc bổ sung 2307 ảnh cho `dent`, `scratch`, `crack` giúp mô hình có thêm ví dụ học cho các dạng hư hỏng khó. Đây là hướng cải thiện hợp lý vì ba lớp này thường khó hơn các lớp có dấu hiệu rõ như `glass shatter`, `lamp broken` hoặc `tire flat`.

Tuy nhiên, do dữ liệu mới chỉ tập trung vào ba lớp yếu, phân phối lớp của dataset sau khi merge không còn tăng đều giữa các nhãn. Vì vậy, khi phân tích kết quả cần nhìn cả metric tổng thể lẫn mục tiêu thực tế của dự án: giảm bỏ sót các vết nứt, vết móp và vết xước trong ảnh xe.

---

## 5. Checkpoint YOLO được chọn hiện tại

Sau tất cả các thực nghiệm đã thử, checkpoint YOLO chính thức được chọn để dùng trong pipeline hiện tại là:

- **mô hình:** `YOLOv8s`
- **nguồn dữ liệu:** dataset `merge_Data`
- **epochs:** `100`
- **batch size:** `16`
- **imgsz khi chạy trong app:** `640`
- **checkpoint tích hợp:** `Models/best.pt`

### Kết quả dùng để chốt mô hình

- **best mAP50 trong `Quan_sat/yolo_car_report/results.csv`: epoch 79, Precision ≈ 0.801, Recall ≈ 0.682, mAP50 ≈ 0.726, mAP50-95 ≈ 0.568**
- **best mAP50-95 trong `Quan_sat/yolo_car_report/results.csv`: epoch 75, mAP50-95 ≈ 0.572**

### Lý do chọn

Checkpoint này được giữ làm mô hình chính vì:

- kế thừa cấu hình ổn định của baseline `s_89`,
- dùng dataset lớn hơn và tập trung hơn cho các lớp khó `crack`, `scratch`, `dent`,
- đủ phù hợp để dùng trong pipeline hiện tại gồm detection, severity classification và price adjustment.

### Phân tích sự lệch pha metrics do nhiễu nhãn

Trong quá trình so sánh mô hình, cần phân biệt giữa **metrics tĩnh** trên tập validation/test và năng lực phát hiện thực tế khi inference trên ảnh hoặc video mới. Điểm mAP hiển thị tĩnh của mô hình `merge_Data` có thể thấp hơn một số cấu hình cũ trên tập test do hiện tượng **Test Set Label Noise**. Cụ thể, tập test cũ có khả năng gán nhãn thiếu các vết xước, vết móp hoặc vết nứt mờ. Khi mô hình mới nhạy hơn và phát hiện đúng các tổn thất nhỏ này, hệ thống đánh giá tự động lại không tìm thấy ground truth tương ứng và tính chúng thành **False Positive**, từ đó kéo Precision và mAP xuống một cách cơ học.

Vì vậy, việc chọn checkpoint không chỉ dựa trên một con số mAP duy nhất. Thực nghiệm inference trên video và ảnh thực tế cho thấy mô hình YOLOv8s trên `merge_Data` bám vết ổn định hơn, phát hiện tốt hơn các dạng hư hỏng nhỏ như `scratch`, `dent` và `crack`. Điều này cho thấy mô hình có khả năng tổng quát hóa thực tế tốt hơn, phù hợp với mục tiêu của hệ thống là phát hiện hư hỏng ngoại thất để phục vụ định giá.

Lưu ý: `Quan_sat/yolo_car_report/results.csv` là kết quả validation trong quá trình train. Nếu cần báo cáo test set cuối cùng một cách tách biệt, cần chạy thêm `model.val(split="test")` trên checkpoint đã chốt. Trong ứng dụng Streamlit, file đang được gọi trực tiếp là `Models/best.pt` thông qua module `dich_vu/phat_hien_hu_hong.py`.

---

## 6. Thử nghiệm CNN cho severity classification

### 6.1 Mục tiêu của nhánh CNN

Nhánh CNN có nhiệm vụ phân loại mức độ hư hỏng của ảnh xe thành ba lớp:

- `minor`
- `moderate`
- `severe`

Khác với mô tả cũ dùng crop damage từ YOLO, hướng hiện tại của dự án sử dụng **ảnh full người dùng upload** làm đầu vào cho CNN. Cách này phù hợp với giao diện demo hiện tại: người dùng đưa vào một ảnh xe, mô hình CNN phân loại trực tiếp mức độ hư hỏng tổng quát của ảnh, sau đó kết quả được dùng cùng các feature detection từ YOLO trong tầng rule-based adjustment.

### 6.2 Các mô hình CNN đã thử nghiệm

Dự án đã thử nghiệm 5 backbone CNN trong thư mục `train/`:

| Notebook | Backbone | Best validation metric | Test accuracy | Test macro F1 |
| --- | --- | ---: | ---: | ---: |
| `resnet_18_new.ipynb` | ResNet18 | val_acc = 0.8209 | 0.6564 | 0.6490 |
| `EfficientNet_B0.ipynb` | EfficientNet-B0 | val_macro_f1 = 0.7945 | 0.6821 | 0.6817 |
| `Efficient_B2.ipynb` | EfficientNet-B2 | val_macro_f1 = 0.8022 | 0.6974 | 0.6954 |
| `ResNet50.ipynb` | ResNet50 | val_acc = 0.8128 | 0.7026 | 0.7043 |
| `ConvNeXt_Tiny.ipynb` | ConvNeXt-Tiny | val_macro_f1 = 0.8342 | 0.7077 | 0.7084 |

Các chỉ số trên cho thấy ResNet18 đạt validation accuracy tương đối cao nhưng giảm mạnh trên test set. Điều này phản ánh khả năng tổng quát hóa chưa ổn định, đặc biệt với lớp trung gian `moderate`. Các backbone mới hơn như EfficientNet-B2, ResNet50 và ConvNeXt-Tiny cải thiện dần kết quả test.

### 6.3 Lý do chọn ConvNeXt-Tiny

ConvNeXt-Tiny được chọn làm mô hình CNN chính vì đạt kết quả tốt nhất trong nhóm mô hình đã thử:

- validation macro F1 cao nhất: **0.8342**,
- test accuracy cao nhất: **0.7077**,
- test macro F1 cao nhất: **0.7084**.

So với ResNet18, ConvNeXt-Tiny có kiến trúc hiện đại hơn và khả năng học đặc trưng thị giác tốt hơn trong các bài toán classification ảnh tự nhiên. Với bài toán hư hỏng xe, sự khác biệt giữa `minor`, `moderate` và `severe` thường phụ thuộc vào các chi tiết nhỏ, phản xạ ánh sáng, vùng móp/xước mờ và bố cục ảnh. ConvNeXt-Tiny cho kết quả cân bằng hơn trên test set, nên phù hợp hơn để dùng làm checkpoint CNN chính của dự án.

### 6.4 Demo định tính trên cùng một ảnh test

Ngoài các metric định lượng, dự án có thử chạy cùng một ảnh xe bị hư hỏng qua 5 mô hình CNN để quan sát kết quả inference:

| Mô hình | Kết quả dự đoán | Xác suất |
| --- | ---: | ---: |
| ResNet18 | SEVERE | 60.61% |
| EfficientNet-B0 | MINOR | 42.60% |
| EfficientNet-B2 | SEVERE | 42.31% |
| ResNet50 | SEVERE | 40.88% |
| ConvNeXt-Tiny | MODERATE | 44.35% |

Với ảnh test này, nhãn đúng là `moderate`, do đó ConvNeXt-Tiny là mô hình duy nhất dự đoán đúng trong ví dụ định tính. ResNet18, EfficientNet-B2 và ResNet50 đều nghiêng về `severe`, trong khi EfficientNet-B0 dự đoán `minor`. Dù confidence của ConvNeXt-Tiny chưa quá cao, kết quả này phù hợp với bảng đánh giá định lượng, nơi ConvNeXt-Tiny đạt test accuracy và macro F1 tốt nhất. Khi đưa vào báo cáo Word, có thể chèn hình minh họa với caption: **So sánh kết quả dự đoán severity của 5 mô hình CNN trên cùng một ảnh test. ConvNeXt-Tiny là mô hình duy nhất dự đoán đúng nhãn moderate trong ví dụ này.**

### 6.5 Hạn chế của nhánh CNN

Dù ConvNeXt-Tiny là mô hình tốt nhất trong các thử nghiệm hiện tại, test accuracy khoảng 70.77% cho thấy bài toán severity classification vẫn còn khó. Lớp `moderate` thường là lớp nhập nhằng nhất vì nằm giữa hai mức `minor` và `severe`. Vì vậy, khi đưa vào tầng điều chỉnh giá, nên xem kết quả CNN là một tín hiệu hỗ trợ thay vì nhãn tuyệt đối hoàn toàn chắc chắn.

---

## 7. Thử nghiệm Faster R-CNN để so sánh phụ

Ngoài YOLOv8, dự án có thử nghiệm thêm **Faster R-CNN ResNet50-FPN** trong notebook `train/train_eval_faster_rcnn_30e_640 (1).ipynb`. Kết quả được lưu tại `Quan_sat/R-CNN_report/` với cấu hình:

- **epochs:** `30`
- **batch size:** `16`
- **imgsz:** `640`

Kết quả tổng hợp:

| Model | mAP@0.5 | mAP@0.5:0.95 | mAP@0.75 |
| --- | ---: | ---: | ---: |
| YOLOv8s merge_Data | 0.726 | 0.568 | - |
| Faster R-CNN ResNet50-FPN | 0.306 | 0.128 | 0.084 |

Kết quả này thấp hơn rõ rệt so với YOLOv8s trong cùng bài toán damage detection. Vì vậy, Faster R-CNN chỉ được dùng như một thí nghiệm đối chứng nhỏ trong báo cáo, còn YOLOv8s vẫn là mô hình detection chính của hệ thống.

---

## 8. Kết luận tổng thể

### 8.1 Kết luận về lựa chọn mô hình

Mô hình YOLO hiện tại nên dùng cho dự án là:

- **`YOLOv8s` trained on dataset `merge_Data`**
- **`epochs = 100`, `batch = 16`, `imgsz = 640` khi tích hợp trong app**
- **`Models/best.pt` với dataset `merge_Data`, validation mAP50 tốt nhất khoảng 0.726**

Mô hình CNN hiện tại nên dùng cho severity classification là:

- **ConvNeXt-Tiny từ `train/ConvNeXt_Tiny.ipynb`**
- **checkpoint tích hợp:** `Models/ConvNeXt.pkl`
- **input là ảnh full người dùng upload**
- **test accuracy khoảng 0.7077 và test macro F1 khoảng 0.7084**

### 8.2 Kết luận về chiến lược cải thiện

Từ các lần train đã thực hiện, có thể rút ra rằng:

- benchmark nhiều model variant là hữu ích để tìm baseline tốt,
- việc giữ nguyên cấu hình và chỉ thay đổi dataset giúp đánh giá rõ tác động của dữ liệu,
- bổ sung dữ liệu nên có mục tiêu rõ ràng, trong dự án này là ba lớp `crack`, `scratch`, `dent`,
- cải thiện tiếp theo vẫn nên ưu tiên theo hướng **data-centric**:
  - audit nhãn,
  - kiểm tra duplicate,
  - bổ sung dữ liệu khó có chủ đích,
  - thêm hard negatives,
  - chỉ train lại sau khi chất lượng dữ liệu tốt hơn.

### 8.3 Trạng thái hiện tại

- Baseline tham khảo: **`s_89` trên dataset ban đầu khoảng 4000 ảnh**.
- Mô hình YOLO chính: **YOLOv8s trên dataset `merge_Data` khoảng 6307 ảnh**.
- Mô hình CNN chính: **ConvNeXt-Tiny phân loại severity từ ảnh full**.
- Checkpoint production-like hiện tại: **`Models/best.pt` từ run `epochs=100`, `batch=16`, `imgsz=640`**.
- Checkpoint CNN cần đồng bộ khi triển khai: **ConvNeXt-Tiny từ `train/ConvNeXt_Tiny.ipynb`**, lưu dưới `Models/ConvNeXt.pkl` nếu dùng trực tiếp trong ứng dụng.
- Dữ liệu bổ sung: **2307 ảnh cho `dent/scratch/crack`, không thêm đều cho tất cả nhãn**.
- Faster R-CNN ResNet50-FPN: **chỉ dùng làm so sánh phụ, không chọn làm mô hình chính**.
