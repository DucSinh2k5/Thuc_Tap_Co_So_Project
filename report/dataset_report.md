# Dataset Report

## 1. Mục đích của dataset trong dự án

Bộ dữ liệu trong dự án này được xây dựng để phục vụ bài toán **ước lượng giá xe ô tô cũ có xét đến tình trạng hư hỏng ngoại thất**. Thay vì chỉ dự đoán giá từ các thuộc tính dạng bảng, dự án kết hợp ba nguồn dữ liệu khác nhau để tạo ra một pipeline đa mô-đun:

- **Dữ liệu tabular** dùng để dự đoán **base price** của xe bằng XGBoost.
- **Dữ liệu ảnh damage detection** dùng để phát hiện các vùng hư hỏng trên xe bằng YOLO.
- **Dữ liệu ảnh severity classification** dùng để phân loại mức độ nghiêm trọng của từng hư hỏng bằng CNN.

Sau khi mô hình tabular dự đoán giá cơ sở, các thông tin trích xuất từ hai mô hình ảnh như số lượng vùng hư hỏng, loại hư hỏng, diện tích vùng hỏng và mức độ nghiêm trọng sẽ được đưa vào lớp **hybrid rule-based adjustment** để ước lượng mức giảm giá và tạo ra **adjusted price**.

Do hiện tại chưa có ground truth cho **final car price after damage**, phần cuối của hệ thống được hiểu là một cơ chế **damage-aware price adjustment** chứ không phải một bộ dự đoán giá cuối được học trực tiếp từ nhãn.

---

## 2. Tổng quan các nguồn dữ liệu

Dự án sử dụng ba nhóm dữ liệu chính:

### 2.1. Severity classification dataset

- **Mục đích:** huấn luyện mô hình CNN để phân loại mức độ hư hỏng thành `minor`, `moderate`, `severe`.
- **Nguồn:** bộ dữ liệu `car-damage-detection` trên Kaggle.
- **Cách sử dụng trong dự án:** giữ nguyên phần severity dataset để phục vụ bài toán classification.

### 2.2. Damage detection dataset (YOLO)

- **Mục đích:** huấn luyện YOLO để phát hiện các vùng hư hỏng trên xe.
- **Nguồn gốc dữ liệu:** lấy từ phần `CarDD_COCO` trong bộ dữ liệu `car-damage-detection` trên Kaggle.
- **Tiền xử lý / chuyển đổi:** dữ liệu được đưa lên Roboflow và xuất lại theo định dạng YOLO để thuận tiện cho quá trình huấn luyện. Vì vậy, cấu trúc dữ liệu hiện tại dùng file cấu hình `data.yaml` và các thư mục `train`, `val`, `test` thay cho cách tổ chức annotation kiểu COCO ban đầu.

### 2.3. Tabular price dataset

- **Mục đích:** huấn luyện mô hình XGBoost để dự đoán giá cơ sở của xe từ các đặc trưng có cấu trúc.
- **Nguồn:** bộ dữ liệu `used-cars-price-prediction` trên Kaggle.
- **Cách sử dụng trong dự án:** dữ liệu được giữ nguyên ở dạng bảng với các file `train-dataset.csv` và `test-dataset.csv`.

---

## 3. Cấu trúc thư mục dữ liệu

Cấu trúc thư mục hiện tại được tổ chức như sau:

```text
Datasets/
├── car_damage_severity/
│   ├── training/
│   └── validation/
│
├── coco_damage_car_yolov8/
│   ├── train/
│   ├── val/
│   ├── test/
│   └── data.yaml
│
├── train-dataset.csv
└── test-dataset.csv
```

### Giải thích cấu trúc

#### `car_damage_severity/`

Thư mục chứa dữ liệu cho bài toán **severity classification**. Dữ liệu được chia thành các tập phục vụ huấn luyện và đánh giá mô hình CNN.

- `training/`: dữ liệu dùng để train mô hình.
- `validation/`: dữ liệu dùng để theo dõi hiệu năng trong quá trình huấn luyện.

Trong mỗi thư mục con, dữ liệu được tổ chức theo nhãn mức độ hư hỏng tương ứng với ba lớp:

- `minor`
- `moderate`
- `severe`

#### `coco_damage_car_yolov8/`

Thư mục chứa dữ liệu cho bài toán **damage detection** bằng YOLO.

- `train/`: tập huấn luyện.
- `val/`: tập validation.
- `test/`: tập test.
- `data.yaml`: file cấu hình dataset cho YOLO, khai báo đường dẫn dữ liệu và danh sách lớp.

Do dữ liệu đã được chuyển đổi sang định dạng YOLO, annotation không còn được tổ chức theo kiểu COCO JSON như nguồn gốc ban đầu, mà đi theo chuẩn YOLO để tương thích trực tiếp với quá trình train.

#### `train-dataset.csv` và `test-dataset.csv`

Hai file dữ liệu bảng dùng cho bài toán **price regression**.

- `train-dataset.csv`: dùng để huấn luyện mô hình XGBoost.
- `test-dataset.csv`: dùng để đánh giá mô hình trên dữ liệu chưa thấy.

Các thuộc tính tabular đại diện cho thông tin mô tả xe như năm sản xuất, số chỗ ngồi, số km đã đi và các đặc trưng cấu trúc khác liên quan đến giá xe.

---

## 4. Vai trò của từng dataset trong pipeline mô hình

Bộ dữ liệu không được dùng như một tập duy nhất cho một mô hình end-to-end, mà được chia thành ba nhánh tương ứng với ba nhiệm vụ khác nhau trong cùng một hệ thống.

### 4.1. Nhánh tabular pricing

Dữ liệu từ `train-dataset.csv` và `test-dataset.csv` được dùng để huấn luyện mô hình **XGBoost Regressor** nhằm dự đoán **base market price** của xe dựa trên các thuộc tính có cấu trúc.

Đầu ra của nhánh này là mức giá tham chiếu ban đầu trước khi xét đến hư hỏng ngoại thất phát hiện từ ảnh.

### 4.2. Nhánh damage detection

Dữ liệu trong `coco_damage_car_yolov8/` được dùng để huấn luyện mô hình **YOLOv8** nhằm phát hiện các vùng hư hỏng trên thân xe. Mô hình này cung cấp các thông tin như:

- số lượng vùng hư hỏng,
- loại hư hỏng,
- vị trí tương đối,
- diện tích hoặc tỷ lệ vùng ảnh bị ảnh hưởng.

Các kết quả này đóng vai trò là nguồn feature thị giác cho bước điều chỉnh giá.

#### Lý do chọn mô hình `yolov8_s100_800`

Trong các mô hình YOLO đã thử nghiệm, dự án lựa chọn checkpoint `yolov8_s100_800` làm mô hình phát hiện hư hỏng chính. Tên checkpoint này thể hiện cấu hình huấn luyện chính: sử dụng biến thể **YOLOv8s**, huấn luyện trong **100 epoch**, với kích thước ảnh đầu vào **800 px**. Đây là cấu hình được chọn vì phù hợp nhất với mục tiêu của dự án: phát hiện tương đối chính xác các vùng hư hỏng ngoại thất nhưng vẫn đủ nhẹ để tích hợp vào ứng dụng demo.

Lý do đầu tiên là **YOLOv8s có mức cân bằng tốt giữa độ chính xác và tốc độ suy luận**. So với YOLOv8n, mô hình YOLOv8s có số tham số lớn hơn nên khả năng học đặc trưng hư hỏng tốt hơn, đặc biệt với các vùng khó như vết xước, vết nứt hoặc vết móp nhỏ. Trong khi đó, so với YOLOv8m, YOLOv8s nhẹ hơn đáng kể, tốc độ suy luận nhanh hơn và phù hợp hơn với hệ thống demo chạy trên máy cá nhân hoặc môi trường Streamlit.

Lý do thứ hai là cấu hình **100 epoch** giúp mô hình có đủ thời gian học các đặc trưng của bộ dữ liệu damage detection mà không cần tăng số epoch quá cao. Với bài toán phát hiện hư hỏng xe, các đặc trưng thị giác có thể khá nhỏ, mảnh và dễ nhầm với phản sáng, đường gân thân xe hoặc vết bẩn. Vì vậy, việc huấn luyện đủ lâu giúp mô hình ổn định hơn so với các run ngắn, đồng thời vẫn tránh làm quá nặng quá trình huấn luyện.

Lý do thứ ba là kích thước ảnh **800 px** được chọn để giữ lại nhiều chi tiết hơn trên ảnh xe. Các hư hỏng như `scratch`, `crack` hoặc `dent` thường có diện tích nhỏ, nếu ảnh đầu vào quá thấp thì các vùng này dễ bị mất chi tiết sau khi resize. Việc tăng kích thước ảnh lên 800 px giúp mô hình quan sát rõ hơn các vùng hỏng nhỏ, từ đó hỗ trợ tốt hơn cho bước phát hiện bounding box và bước phân loại mức độ hư hỏng phía sau.

Ngoài ra, trong quá trình thử nghiệm, dự án cũng đã đánh giá việc bổ sung thêm dữ liệu cho các lớp khó như `dent`, `scratch`, `crack`. Tuy nhiên, các lần mở rộng dữ liệu này chưa tạo ra cải thiện ổn định so với baseline tốt nhất. Vì vậy, checkpoint `yolov8_s100_800` được giữ lại làm mô hình chính vì cho kết quả tổng thể ổn định hơn, dễ tích hợp hơn và phù hợp với pipeline hiện tại gồm: phát hiện hư hỏng bằng YOLO, phân loại mức độ bằng CNN và điều chỉnh giá xe bằng rule-based adjustment.

### 4.3. Nhánh severity classification

Dữ liệu trong `car_damage_severity/` được dùng để huấn luyện mô hình **CNN** nhằm phân loại mức độ nghiêm trọng của vùng hư hỏng thành ba mức:

- `minor`
- `moderate`
- `severe`

Trong pipeline, CNN có thể được dùng sau bước detection để phân loại severity cho từng vùng damage đã phát hiện.

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
- **Mức sử dụng trong dự án:** phân loại mức độ nghiêm trọng của hư hỏng.

### 5.2. Dữ liệu YOLO detection

- **Kiểu dữ liệu:** ảnh + annotation theo định dạng YOLO.
- **Bài toán:** object detection.
- **Đặc điểm:** dữ liệu được chuyển đổi từ nguồn COCO sang định dạng YOLO để phù hợp với quá trình huấn luyện bằng YOLOv8.

### 5.3. Dữ liệu tabular pricing

- **Kiểu dữ liệu:** bảng `. csv`.
- **Bài toán:** regression.
- **Target:** giá xe.
- **Đặc trưng đầu vào:** các thuộc tính cấu trúc của xe.

---

## 6. Nguồn gốc dữ liệu và tính nhất quán

Ba nhóm dữ liệu trong dự án đến từ các nguồn khác nhau và phục vụ các mục tiêu khác nhau, do đó đây là một **modular dataset design** thay vì một tập dữ liệu đồng bộ hoàn toàn theo từng chiếc xe.

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

Riêng với nhánh detection, dữ liệu gốc COCO đã được đưa qua Roboflow rồi xuất lại theo định dạng YOLO. Việc này thuận tiện cho huấn luyện nhưng cần đảm bảo annotation được giữ nhất quán sau khi chuyển đổi.

---

## 8. Kết luận

Bộ dữ liệu của dự án được xây dựng theo hướng thực dụng và mô-đun, gồm ba thành phần phục vụ ba nhiệm vụ khác nhau: **price regression**, **damage detection** và **severity classification**. Cách tổ chức này phù hợp với mục tiêu phát triển một hệ thống ước lượng giá xe cũ có xét đến tình trạng ngoại thất, đồng thời cho phép đánh giá từng thành phần mô hình một cách độc lập.

Dù chưa có dataset multimodal đồng bộ hoàn chỉnh và chưa có ground truth cho final price after damage, cấu trúc dữ liệu hiện tại vẫn đủ để xây dựng một pipeline thử nghiệm hợp lý, gồm:

1. dự đoán giá cơ sở từ dữ liệu bảng,
2. phát hiện hư hỏng từ ảnh,
3. phân loại mức độ nghiêm trọng,
4. điều chỉnh giá theo mức hư hỏng quan sát được.

Trong các bước phát triển tiếp theo, dự án có thể được cải thiện bằng cách bổ sung dữ liệu đồng bộ hơn giữa tabular và image data, đồng thời xây dựng nhãn hoặc quy tắc định lượng tốt hơn cho phần price adjustment.
