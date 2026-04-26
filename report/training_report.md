# Training Report

## 1. Mục tiêu báo cáo

Tài liệu này ghi lại các lần huấn luyện mô hình **YOLOv8** cho bài toán **car damage detection** trong pipeline dự đoán giá xe cũ. Module này có nhiệm vụ phát hiện các hư hỏng ngoại quan trên ảnh xe để tạo ra các feature downstream như:

- số lượng hư hỏng theo lớp,
- diện tích vùng hư hỏng,
- thông tin đầu vào cho bước **severity classification**,
- tín hiệu cho bước **price adjustment** trong tầng hybrid phía sau.

Báo cáo này tập trung vào các thực nghiệm của phần **object detection**, bao gồm:

- benchmark nhiều biến thể YOLOv8 trên dataset ban đầu,
- thử nghiệm mở rộng tập train bằng dữ liệu mới cho các lớp khó,
- quyết định chọn checkpoint YOLO chính thức để dùng cho pipeline hiện tại.

---

## 2. Bối cảnh thực nghiệm

### 2.1 Dataset ban đầu

Dataset detection ban đầu được tổ chức theo định dạng YOLO với 6 lớp:

- `crack`
- `dent`
- `glass shatter`
- `lamp broken`
- `scratch`
- `tire flat`

Split dùng trong các run baseline ban đầu:

- `train`: 2800 ảnh
- `val`: 800 ảnh
- `test`: giữ riêng để đánh giá cuối

### 2.2 Môi trường huấn luyện

Các thực nghiệm YOLO được train trên môi trường dùng GPU **Tesla T4 (~15GB VRAM)** với:

- Python `3.12.x`
- PyTorch `2.10.0+cu128`
- Ultralytics `8.4.x`

### 2.3 Cấu hình baseline chung

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

### 3.2 Kết quả validation tổng hợp

| Model   | Params | GFLOPs | Precision |    Recall |   mAP@0.5 | mAP@0.5:0.95 | Inference (ms/img) |
| ------- | -----: | -----: | --------: | --------: | --------: | -----------: | -----------------: |
| YOLOv8n |  3.01M |    8.2 |     0.748 |     0.678 |     0.714 |        0.568 |                2.1 |
| YOLOv8s | 11.14M |   28.7 | **0.798** |     0.685 | **0.735** |        0.587 |                3.6 |
| YOLOv8m | 25.86M |   79.1 |     0.765 | **0.703** |     0.728 |    **0.590** |                7.7 |

### 3.3 Nhận xét từ benchmark ban đầu

- `YOLOv8n` là mô hình nhẹ nhất, phù hợp làm baseline nhẹ nhưng chưa phải lựa chọn tốt nhất về tổng thể.
- `YOLOv8s` cho cân bằng tốt nhất giữa độ chính xác, tốc độ suy luận và kích thước mô hình.
- `YOLOv8m` chỉ cải thiện rất ít so với `YOLOv8s`, nhưng chi phí tính toán cao hơn rõ rệt.

### 3.4 Kết luận giai đoạn benchmark ban đầu

Sau benchmark trên dataset ban đầu, `YOLOv8s` được chọn làm ứng viên mạnh nhất để tiếp tục phát triển.

---

## 4. Thử nghiệm mở rộng dữ liệu cho các lớp khó

### 4.1 Lý do mở rộng dữ liệu

Các lớp khó của dataset ban đầu là:

- `crack`
- `scratch`
- `dent`

Đây là các lớp có xu hướng khó detect hơn do:

- vùng hỏng nhỏ hoặc mảnh,
- dễ bị ảnh hưởng bởi phản sáng, bóng đổ và góc chụp,
- nhãn dễ thiếu nhất quán hơn các lớp dễ như `glass shatter` hoặc `lamp broken`.

Vì vậy, sau giai đoạn benchmark ban đầu, dự án đã thử **bổ sung thêm dữ liệu ảnh cho ba lớp này vào tập train**.

### 4.2 Các lần thử mở rộng dữ liệu

Đã thực hiện:

- **lần thêm dữ liệu thứ nhất**: bổ sung ảnh cho các lớp `dent`, `scratch`, `crack` rồi train lại.
- **lần thêm dữ liệu thứ hai**: tiếp tục bổ sung dữ liệu theo cùng hướng rồi train lại thêm một lần nữa.

Mục tiêu của hai lần thử này là kiểm tra xem việc tăng dữ liệu có chủ đích cho các lớp khó có thể cải thiện mAP tổng thể hay không.

### 4.3 Kết quả của các lần mở rộng dữ liệu

Kết quả thực tế cho thấy:

- các run sau khi thêm dữ liệu **không vượt được baseline tốt nhất từ dataset ban đầu**,
- mAP tổng thể **thấp hơn** so với mô hình tốt nhất của bộ dữ liệu gốc,
- checkpoint sinh ra từ các lần mở rộng dữ liệu **không được chọn** làm mô hình chính thức.

### 4.4 Diễn giải

Điều này cho thấy việc bổ sung dữ liệu mới cho `dent`, `scratch`, `crack` trong hai lần thử nghiệm vừa qua **chưa đủ để tạo cải thiện thực sự**. Các nguyên nhân có thể gồm:

- dữ liệu mới chưa đủ sạch hoặc chưa đủ đại diện,
- dữ liệu mới có thể làm phân phối train lệch đi theo hướng không có lợi,
- cần audit nhãn kỹ hơn trước khi tiếp tục gộp dữ liệu,
- số lượng tăng thêm không quan trọng bằng chất lượng và độ khó đúng mục tiêu.

---

## 5. Checkpoint YOLO được chọn hiện tại

Sau tất cả các thực nghiệm đã thử, checkpoint YOLO chính thức được chọn để dùng trong pipeline hiện tại là:

- **mô hình:** `YOLOv8s`
- **nguồn dữ liệu:** dataset ban đầu
- **epochs:** `100`
- **batch size:** `16`
- **imgsz:** `640`

### Kết quả dùng để chốt mô hình

- **mAP50 trên tập test ≈ 0.725 (72.5%)**

### Lý do chọn

Checkpoint này được giữ làm mô hình chính vì:

- ổn định hơn các run mở rộng dữ liệu sau đó,
- cho hiệu năng tổng thể tốt hơn,
- đủ phù hợp để dùng trong pipeline hiện tại gồm detection, severity classification và price adjustment.

---

## 6. Kết luận tổng thể

### 6.1 Kết luận về lựa chọn mô hình

Mô hình YOLO hiện tại nên dùng cho dự án là:

- **`YOLOv8s` trained on the original dataset**
- **`epochs = 100`, `batch = 16`, `imgsz = 640`**
- **`best.pt` với test mAP50 khoảng 0.725**

### 6.2 Kết luận về chiến lược cải thiện

Từ các lần train đã thực hiện, có thể rút ra rằng:

- benchmark nhiều model variant là hữu ích để tìm baseline tốt,
- nhưng việc thêm dữ liệu mới không tự động giúp mô hình tốt hơn,
- khi các run mới cho mAP thấp hơn baseline cũ, cần giữ lại baseline cũ làm checkpoint chính,
- cải thiện tiếp theo nên ưu tiên theo hướng **data-centric**:
  - audit nhãn,
  - kiểm tra duplicate,
  - bổ sung dữ liệu khó có chủ đích,
  - thêm hard negatives,
  - chỉ train lại sau khi chất lượng dữ liệu tốt hơn.

### 6.3 Trạng thái hiện tại

- Baseline chính: **YOLOv8s trên dataset ban đầu**.
- Checkpoint production-like hiện tại: **`best.pt` từ run `epochs=100`, `batch=16`, `imgsz=640`**.
- Các run dùng dữ liệu mở rộng cho `dent/scratch/crack`: **chưa đủ tốt để thay thế baseline**.
