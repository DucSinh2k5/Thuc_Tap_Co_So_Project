# Dataset Report

## 1. Mục đích của dataset trong dự án

Bộ dữ liệu trong dự án này được xây dựng để phục vụ bài toán **ước lượng giá xe ô tô cũ có xét đến tình trạng hư hỏng ngoại thất**. Thay vì chỉ dự đoán giá từ các thuộc tính dạng bảng, dự án kết hợp ba nguồn dữ liệu khác nhau để tạo ra một pipeline đa mô-đun:

- **Dữ liệu tabular** dùng để dự đoán **base price** của xe bằng XGBoost.
- **Dữ liệu ảnh damage detection** dùng để phát hiện các vùng hư hỏng trên xe bằng YOLO.
- **Dữ liệu ảnh severity classification** dùng để phân loại mức độ nghiêm trọng của hư hỏng từ ảnh xe full bằng CNN.

Sau khi mô hình tabular dự đoán giá cơ sở, các thông tin trích xuất từ hai mô hình ảnh như số lượng vùng hư hỏng, loại hư hỏng, diện tích vùng hỏng từ YOLO và mức độ nghiêm trọng từ CNN trên ảnh full sẽ được đưa vào lớp **hybrid rule-based adjustment** để ước lượng mức giảm giá và tạo ra **adjusted price**.

Trong ứng dụng hiện tại, phần ảnh là **tùy chọn**. Nếu người dùng không upload ảnh, hệ thống bỏ qua nhánh YOLO/CNN và giá cuối cùng chính là giá cơ sở được dự đoán từ mô hình tabular XGBoost.

Do hiện tại chưa có ground truth cho **final car price after damage**, phần cuối của hệ thống được hiểu là một cơ chế **damage-aware price adjustment** chứ không phải một bộ dự đoán giá cuối được học trực tiếp từ nhãn.

### 1.1. Sơ đồ pipeline hệ thống

Sơ đồ dưới đây tóm tắt luồng xử lý tổng thể của hệ thống, từ đầu vào người dùng đến kết quả định giá cuối cùng:

![Sơ đồ pipeline kiến trúc hệ thống](system_pipeline_diagram.png)

Trong pipeline này, nhánh dữ liệu bảng luôn được xử lý để tạo **base price** bằng XGBoost. Nhánh ảnh chỉ được kích hoạt khi người dùng upload ảnh xe; khi đó YOLOv8s phát hiện các vùng hư hỏng, đồng thời CNN ConvNeXt-Tiny phân loại mức độ nghiêm trọng trực tiếp từ ảnh full người dùng đưa vào. Tầng rule-based sau đó kết hợp thông tin detection và severity để điều chỉnh giá.

---

## 2. Tổng quan các nguồn dữ liệu

Dự án sử dụng ba nhóm dữ liệu chính:

### 2.1. Severity classification dataset

- **Mục đích:** huấn luyện mô hình CNN để phân loại mức độ hư hỏng thành `minor`, `moderate`, `severe`.
- **Nguồn:** bộ dữ liệu `car-damage-detection` trên Kaggle: <https://www.kaggle.com/datasets/asfarhossainsitab/car-damage-detection>.
- **Cách sử dụng trong dự án:** dùng ảnh full trong severity dataset để huấn luyện và đánh giá bài toán image classification.

### 2.2. Damage detection dataset (YOLO)

- **Mục đích:** huấn luyện YOLO để phát hiện các vùng hư hỏng trên xe.
- **Nguồn gốc dữ liệu:** lấy từ phần `CarDD_COCO` trong bộ dữ liệu `car-damage-detection` trên Kaggle: <https://www.kaggle.com/datasets/asfarhossainsitab/car-damage-detection>.
- **Nguồn dữ liệu bổ sung:** sau giai đoạn baseline, dự án bổ sung thêm **2307 ảnh** từ Roboflow Universe, lấy chủ yếu từ hai nguồn:
  - AutoDentify Car Damage Detection: <https://universe.roboflow.com/autodentify/car-damage-detection-ggmju>
  - Scratch and Dent: <https://universe.roboflow.com/nibm-7v215/scratch-and-dent-xvjy5>
- **Tiền xử lý / chuyển đổi:** dữ liệu gốc từ Kaggle được tải về ở định dạng COCO, sau đó được upload lên Roboflow để chuyển đổi và xuất lại theo định dạng YOLOv8. Vì vậy, cấu trúc dữ liệu hiện tại dùng file cấu hình `data.yaml` và các thư mục `train`, `val`, `test` thay cho cách tổ chức annotation kiểu COCO ban đầu.
- **Phiên bản dữ liệu cuối:** ban đầu dự án dùng khoảng **4000 ảnh** từ Kaggle cho các run nền tảng như `s_89`. Sau đó, dự án tạo phiên bản **merge_Data** bằng cách bổ sung thêm **2307 ảnh** có chủ đích từ Roboflow Universe cho ba lớp yếu nhất là `crack`, `scratch` và `dent`. Việc bổ sung này không áp dụng đều cho toàn bộ nhãn mà tập trung vào các lớp khó nhằm cải thiện khả năng phát hiện vết nứt, vết xước và vết móp.

### 2.3. Tabular price dataset

- **Mục đích:** huấn luyện mô hình XGBoost để dự đoán giá cơ sở của xe từ các đặc trưng có cấu trúc.
- **Nguồn:** bộ dữ liệu `used-cars-price-prediction` trên Kaggle: <https://www.kaggle.com/datasets/avikasliwal/used-cars-price-prediction>.
- **Cách sử dụng trong dự án:** dữ liệu được giữ nguyên ở dạng bảng với các file `train-dataset.csv` và `test-dataset.csv`.

---

## 3. Cấu trúc thư mục dữ liệu

Cấu trúc thư mục hiện tại được tổ chức như sau:

```text
Datasets/
├── Used_Car_Dataset.csv
├── train_cleaned.csv
├── train-dataset.csv
├── test.csv
└── test-dataset.csv

Models/
├── model.pkl
├── best.pt
└── ConvNeXt.pkl

Quan_sat/
├── yolo_car_report/        # report của YOLOv8s trên merge_Data
├── R-CNN_report/
├── baseline_report.txt
├── feature_importance_xgb.csv
├── feature_importance_rf.csv
├── permutation_importance_xgb.csv
└── permutation_importance_rf.csv

train/
├── s_100_800.ipynb
├── s_89.ipynb              # run YOLOv8s trên dataset gốc, dùng làm baseline
├── test.ipynb
├── resnet_18_new.ipynb       # baseline CNN ResNet18
├── EfficientNet_B0.ipynb
├── Efficient_B2.ipynb
├── ResNet50.ipynb
├── ConvNeXt_Tiny.ipynb       # mô hình CNN chính hiện tại
└── train_eval_faster_rcnn_30e_640 (1).ipynb
```

### Giải thích cấu trúc

#### `train-dataset.csv` và `test-dataset.csv`

Hai file dữ liệu bảng dùng cho bài toán **price regression**.

- `train-dataset.csv`: dùng để huấn luyện mô hình XGBoost.
- `test-dataset.csv`: dùng để đánh giá mô hình trên dữ liệu chưa thấy.

Các thuộc tính tabular đại diện cho thông tin mô tả xe như năm sản xuất, số chỗ ngồi, số km đã đi và các đặc trưng cấu trúc khác liên quan đến giá xe.

#### Dữ liệu ảnh YOLO và CNN

Dữ liệu ảnh phục vụ YOLO/CNN được sử dụng trong các notebook huấn luyện trong thư mục `train/`. Trong repository hiện tại, các mô hình đã huấn luyện và kết quả quan sát được lưu lại dưới dạng checkpoint và báo cáo:

- `Models/best.pt`: checkpoint YOLO đang được ứng dụng Streamlit sử dụng để phát hiện hư hỏng.
- `Models/ConvNeXt.pkl`: checkpoint CNN chính dùng cho severity classification; hướng hiện tại chọn ConvNeXt-Tiny thay cho ResNet18.
- `Quan_sat/yolo_car_report/`: ảnh minh họa batch train/validation, confusion matrix, PR/F1 curve và `results.csv` của run YOLOv8s trên dataset `merge_Data`.
- `Quan_sat/R-CNN_report/`: kết quả thử nghiệm Faster R-CNN để tham khảo/so sánh.

---

## 4. Vai trò của từng dataset trong pipeline mô hình

Bộ dữ liệu không được dùng như một tập duy nhất cho một mô hình end-to-end, mà được chia thành ba nhánh tương ứng với ba nhiệm vụ khác nhau trong cùng một hệ thống.

### 4.1. Nhánh tabular pricing

Dữ liệu từ `train-dataset.csv` và `test-dataset.csv` được dùng để huấn luyện mô hình **XGBoost Regressor** nhằm dự đoán **base market price** của xe dựa trên các thuộc tính có cấu trúc.

Đầu ra của nhánh này là mức giá tham chiếu ban đầu trước khi xét đến hư hỏng ngoại thất phát hiện từ ảnh.

### 4.2. Nhánh damage detection

Dữ liệu damage detection được sử dụng trong các notebook huấn luyện YOLO nhằm phát hiện các vùng hư hỏng trên thân xe. Dataset ảnh gốc được lấy từ `CarDD_COCO` trên Kaggle ở định dạng COCO, sau đó upload lên Roboflow để chuyển đổi sang định dạng YOLOv8. Repository hiện tại lưu lại notebook, checkpoint và các kết quả quan sát trong `Quan_sat/yolo_car_report/`. Phiên bản dữ liệu cuối được dùng để chọn mô hình là `merge_Data`, được mở rộng từ dataset gốc khoảng 4000 ảnh bằng cách thêm 2307 ảnh từ Roboflow Universe cho các lớp `crack`, `scratch` và `dent`. Mô hình này cung cấp các thông tin như:

- số lượng vùng hư hỏng,
- loại hư hỏng,
- vị trí tương đối,
- diện tích hoặc tỷ lệ vùng ảnh bị ảnh hưởng.

Các kết quả này đóng vai trò là nguồn feature thị giác cho bước điều chỉnh giá.

#### Lý do chọn YOLOv8s làm mô hình detection chính

Trong các mô hình YOLO đã thử nghiệm, dự án chọn **YOLOv8s train trên dataset `merge_Data`** làm mô hình phát hiện hư hỏng chính. Checkpoint đang được ứng dụng sử dụng nằm tại `Models/best.pt`; trong mã nguồn `dich_vu/phat_hien_hu_hong.py`, mô hình được gọi với `imgsz = 640`, `conf = 0.25` và `iou = 0.45`. Run `s_89` trên dataset gốc 4000 ảnh được xem là baseline so sánh, không còn là mô hình chính cuối cùng.

Lý do đầu tiên là **YOLOv8s có mức cân bằng tốt giữa độ chính xác và tốc độ suy luận**. So với YOLOv8n, mô hình YOLOv8s có số tham số lớn hơn nên khả năng học đặc trưng hư hỏng tốt hơn, đặc biệt với các vùng khó như vết xước, vết nứt hoặc vết móp nhỏ. Trong khi đó, so với YOLOv8m, YOLOv8s nhẹ hơn đáng kể, tốc độ suy luận nhanh hơn và phù hợp hơn với hệ thống demo chạy trên máy cá nhân hoặc môi trường Streamlit.

Lý do thứ hai là cấu hình **100 epoch** giúp mô hình có đủ thời gian học các đặc trưng của bộ dữ liệu damage detection mà không cần tăng số epoch quá cao. Với bài toán phát hiện hư hỏng xe, các đặc trưng thị giác có thể khá nhỏ, mảnh và dễ nhầm với phản sáng, đường gân thân xe hoặc vết bẩn. Vì vậy, việc huấn luyện đủ lâu giúp mô hình ổn định hơn so với các run ngắn, đồng thời vẫn tránh làm quá nặng quá trình huấn luyện.

Lý do thứ ba là cấu hình **imgsz = 640** phù hợp với môi trường huấn luyện và triển khai hiện tại. Kích thước này đủ lớn để giữ lại nhiều chi tiết của các vùng hư hỏng, đồng thời vẫn giúp thời gian suy luận không quá nặng khi tích hợp vào ứng dụng. Trong `Quan_sat/yolo_car_report/results.csv`, run YOLOv8s trên `merge_Data` đạt mAP@0.5 tốt nhất ở khoảng epoch 79 với **Precision = 0.801**, **Recall = 0.682**, **mAP@0.5 = 0.726** và **mAP@0.5:0.95 = 0.568**. Nếu xét riêng mAP@0.5:0.95, epoch 75 đạt giá trị cao nhất khoảng **0.572**.

Ngoài ra, việc mở rộng dữ liệu chỉ tập trung vào `dent`, `scratch`, `crack` vì đây là ba lớp có đặc trưng nhỏ, mảnh và dễ bị bỏ sót nhất. Cách mở rộng này phù hợp với mục tiêu thực tế của dự án: tăng khả năng nhận diện các hư hỏng khó, thay vì chỉ tăng dữ liệu một cách đồng đều cho mọi nhãn. Vì vậy, YOLOv8s trên `merge_Data` được giữ làm mô hình detection chính vì cân bằng giữa độ chính xác, tốc độ, khả năng tổng quát hóa và mức độ dễ tích hợp vào pipeline hiện tại gồm: phát hiện hư hỏng bằng YOLO, phân loại mức độ bằng CNN ConvNeXt-Tiny và điều chỉnh giá xe bằng rule-based adjustment.

### 4.3. Nhánh severity classification

Dữ liệu severity classification được dùng trong các notebook CNN trong thư mục `train/` để huấn luyện mô hình phân loại mức độ nghiêm trọng của hư hỏng từ **ảnh full** thành ba mức:

- `minor`
- `moderate`
- `severe`

Trong pipeline mới, CNN không còn được mô tả như một mô hình nhận crop damage từ YOLO. Mô hình severity classification nhận trực tiếp ảnh full người dùng upload và dự đoán một trong ba mức: `minor`, `moderate`, `severe`. YOLO và CNN vì vậy là hai nhánh xử lý ảnh song song: YOLO cung cấp loại/vị trí/diện tích hư hỏng, còn CNN cung cấp nhãn mức độ tổng quát của ảnh.

Các mô hình CNN đã được thử nghiệm gồm:

| Notebook                | Backbone        | Best validation metric | Test accuracy | Test macro F1 |
| ----------------------- | --------------- | ---------------------: | ------------: | ------------: |
| `resnet_18_new.ipynb`   | ResNet18        |       val_acc = 0.8209 |        0.6564 |        0.6490 |
| `EfficientNet_B0.ipynb` | EfficientNet-B0 |  val_macro_f1 = 0.7945 |        0.6821 |        0.6817 |
| `Efficient_B2.ipynb`    | EfficientNet-B2 |  val_macro_f1 = 0.8022 |        0.6974 |        0.6954 |
| `ResNet50.ipynb`        | ResNet50        |       val_acc = 0.8128 |        0.7026 |        0.7043 |
| `ConvNeXt_Tiny.ipynb`   | ConvNeXt-Tiny   |  val_macro_f1 = 0.8342 |        0.7077 |        0.7084 |

Các mô hình CNN được train theo quy trình transfer learning hai phase. Phase 1 chỉ train classifier/head trên backbone pretrained. Phase 2 fine-tune thêm các block cuối của backbone cùng classifier:

| Notebook | Backbone | Classifier/head thay thế | Phase 1 train | Phase 2 fine-tune |
| --- | --- | --- | --- | --- |
| `cnn_train_car.ipynb` | ResNet18 bản thử nghiệm ban đầu | `fc = Linear(..., 3)` | `fc` | `layer4 + fc` |
| `resnet_18_new.ipynb` | ResNet18 | `fc = Dropout(0.35) + Linear(..., 3)` | `fc` | `layer4 + fc` |
| `EfficientNet_B0.ipynb` | EfficientNet-B0 | `classifier = Dropout(0.40) + Linear(..., 3)` | `classifier` | `features[-2:] + classifier` |
| `Efficient_B2.ipynb` | EfficientNet-B2 | `classifier = Dropout(0.40) + Linear(..., 3)` | `classifier` | `features[-2:] + classifier` |
| `ResNet50.ipynb` | ResNet50 | `fc = Dropout(0.35) + Linear(..., 3)` | `fc` | `layer4 + fc` |
| `ConvNeXt_Tiny.ipynb` | ConvNeXt-Tiny | `classifier[2] = Linear(..., 3)` | `classifier` | `features[-2:] + classifier` |

Các classifier/head đều đưa ra ba lớp severity: `minor`, `moderate`, `severe`. Với ResNet, block cuối được fine-tune là `layer4`; với EfficientNet và ConvNeXt-Tiny, các block cuối được fine-tune là `features[-2:]`.

ConvNeXt-Tiny được chọn làm mô hình CNN chính vì đạt kết quả tốt nhất trong nhóm mô hình đã thử trên cả validation macro F1 và test macro F1. Ngoài ra, ConvNeXt-Tiny có kiến trúc hiện đại hơn ResNet18, phù hợp hơn với bài toán phân loại ảnh hư hỏng có nhiều chi tiết nhỏ, phản xạ ánh sáng và khác biệt mờ giữa `minor`, `moderate`, `severe`.

### 4.4. Lớp kết hợp cuối cùng

Sau khi có:

- **base price** từ XGBoost,
- **damage features** từ YOLO,
- **severity labels** từ CNN,

hệ thống áp dụng một lớp **rule-based adjustment** để ước lượng mức giảm giá. Ví dụ, hệ thống có thể sử dụng các đặc trưng như:

- `num_dents`
- `num_scratch`
- `num_lamp_broken`
- `total_damage_area`
- `max_severity`
- `severity_score`

để xây dựng một hàm giảm trừ giá phù hợp.

---

## 5. Định dạng và đặc điểm của từng nhóm dữ liệu

### 5.1. Dữ liệu severity classification

- **Kiểu dữ liệu:** ảnh.
- **Bài toán:** multi-class image classification.
- **Nhãn:** `minor`, `moderate`, `severe`.
- **Mức sử dụng trong dự án:** phân loại mức độ nghiêm trọng của hư hỏng từ ảnh full người dùng upload.

### 5.2. Dữ liệu YOLO detection

- **Kiểu dữ liệu:** ảnh + annotation theo định dạng YOLO.
- **Bài toán:** object detection.
- **Đặc điểm:** dữ liệu gốc từ Kaggle ở định dạng COCO được chuyển sang YOLOv8 thông qua Roboflow. Phiên bản `merge_Data` sau đó bổ sung thêm ảnh từ Roboflow Universe để tăng dữ liệu cho các lớp khó `crack`, `scratch` và `dent`.

### 5.3. Dữ liệu tabular pricing

- **Kiểu dữ liệu:** bảng `. csv`.
- **Bài toán:** regression.
- **Target:** giá xe.
- **Đặc trưng đầu vào:** các thuộc tính cấu trúc của xe.

---

## 6. Nguồn gốc dữ liệu và tính nhất quán

Ba nhóm dữ liệu trong dự án đến từ các nguồn khác nhau và phục vụ các mục tiêu khác nhau, do đó đây là một **modular dataset design** thay vì một tập dữ liệu đồng bộ hoàn toàn theo từng chiếc xe.

Cụ thể, dữ liệu ảnh cho nhánh CNN severity classification xuất phát từ bộ `car-damage-detection` trên Kaggle: <https://www.kaggle.com/datasets/asfarhossainsitab/car-damage-detection>. Với nhánh YOLO damage detection, dữ liệu baseline cũng lấy từ phần `CarDD_COCO` của bộ Kaggle trên, sau đó được chuyển từ COCO sang YOLOv8 bằng Roboflow. Phiên bản YOLO cuối `merge_Data` có bổ sung thêm 2307 ảnh từ Roboflow Universe, gồm AutoDentify Car Damage Detection (<https://universe.roboflow.com/autodentify/car-damage-detection-ggmju>) và Scratch and Dent (<https://universe.roboflow.com/nibm-7v215/scratch-and-dent-xvjy5>). Dữ liệu bảng cho nhánh XGBoost price regression sử dụng bộ `used-cars-price-prediction` trên Kaggle: <https://www.kaggle.com/datasets/avikasliwal/used-cars-price-prediction>.

Điều này có nghĩa là:

- dữ liệu ảnh damage và severity được dùng để học đặc trưng thị giác liên quan đến hư hỏng,
- dữ liệu tabular được dùng để học quan hệ giữa thuộc tính xe và giá thị trường,
- bước hợp nhất cuối cùng được thực hiện ở mức pipeline thay vì dựa trên một dataset multimodal đồng bộ hoàn chỉnh.

Cách tổ chức này phù hợp với mục tiêu xây dựng một hệ thống thử nghiệm đa mô-đun, nhưng cũng kéo theo một hạn chế quan trọng: **chưa có ground truth trực tiếp cho final price after damage**.

---

## 7. Hạn chế hiện tại của dataset

### 7.1. Chưa có nhãn cho giá cuối sau damage

Dự án hiện có nhãn giá cho bài toán tabular regression, nhưng chưa có nhãn trực tiếp cho mức giá cuối cùng sau khi xét đến damage từ ảnh. Vì vậy, phần cuối của pipeline hiện mang tính **damage-aware adjustment** thay vì supervised final-price prediction.

### 7.2. Dữ liệu không đồng bộ hoàn toàn theo từng xe

Ba nhánh dữ liệu được huấn luyện tương đối độc lập. Điều này giúp dễ triển khai và đánh giá từng mô hình riêng, nhưng chưa phản ánh đầy đủ một bài toán multimodal end-to-end trên cùng một đối tượng xe.

### 7.3. Chất lượng đầu vào phụ thuộc vào quá trình chuyển đổi định dạng

Riêng với nhánh detection, dữ liệu gốc COCO đã được đưa qua Roboflow rồi xuất lại theo định dạng YOLOv8. Ngoài ra, phiên bản `merge_Data` còn kết hợp thêm dữ liệu từ Roboflow Universe. Việc này thuận tiện cho huấn luyện và mở rộng tập ảnh, nhưng cần đảm bảo annotation được giữ nhất quán sau khi chuyển đổi và sau khi merge nhiều nguồn dữ liệu.

---

## 8. Kết luận

Bộ dữ liệu của dự án được xây dựng theo hướng thực dụng và mô-đun, gồm ba thành phần phục vụ ba nhiệm vụ khác nhau: **price regression**, **damage detection** và **severity classification**. Cách tổ chức này phù hợp với mục tiêu phát triển một hệ thống ước lượng giá xe cũ có xét đến tình trạng ngoại thất, đồng thời cho phép đánh giá từng thành phần mô hình một cách độc lập.

Dù chưa có dataset multimodal đồng bộ hoàn chỉnh và chưa có ground truth cho final price after damage, cấu trúc dữ liệu hiện tại vẫn đủ để xây dựng một pipeline thử nghiệm hợp lý, gồm:

1. dự đoán giá cơ sở từ dữ liệu bảng,
2. phát hiện hư hỏng từ ảnh,
3. phân loại mức độ nghiêm trọng,
4. điều chỉnh giá theo mức hư hỏng quan sát được.

Nếu không có ảnh đầu vào, hệ thống vẫn hoạt động như một bài toán tabular regression thuần túy: dự đoán base price bằng XGBoost và trả kết quả này làm final price.

Trong các bước phát triển tiếp theo, dự án có thể được cải thiện bằng cách bổ sung dữ liệu đồng bộ hơn giữa tabular và image data, đồng thời xây dựng nhãn hoặc quy tắc định lượng tốt hơn cho phần price adjustment.
