# Training Report

## 1. Mục tiêu báo cáo

Tài liệu này ghi lại các lần huấn luyện mô hình **YOLOv8** cho bài toán **car damage detection** trong pipeline dự đoán giá xe cũ. Vai trò của module này là phát hiện các hư hỏng ngoại quan trên ảnh xe, từ đó sinh ra các feature như số lượng hư hỏng, loại hư hỏng, diện tích vùng hư hỏng và các thông tin đầu vào cho bước **severity classification** cũng như **price adjustment** ở tầng hybrid phía sau.

Phạm vi của báo cáo hiện tại chỉ tập trung vào **thực nghiệm object detection** với ba biến thể:

- `YOLOv8n`
- `YOLOv8s`
- `YOLOv8m`

Các mô hình **CNN severity classification** và **XGBoost pricing** sẽ được ghi lại ở các báo cáo hoặc phiên bản cập nhật tiếp theo.

---

## 2. Bối cảnh thực nghiệm

### 2.1 Mục tiêu thực nghiệm

Mục tiêu của giai đoạn này là:

1. So sánh ba biến thể YOLOv8 theo hướng **nhẹ - vừa - mạnh**.
2. Đánh giá sự đánh đổi giữa:
   - độ chính xác detection,
   - tốc độ suy luận,
   - kích thước mô hình,
   - chi phí tính toán.
3. Chọn một mô hình phù hợp nhất để tích hợp vào pipeline tổng thể của dự án.

### 2.2 Dataset dùng để train detection

Dataset detection được tổ chức theo định dạng YOLO, gồm:

- `train`: 2800 ảnh
- `val`: 800 ảnh
- số lớp: 6

Các lớp hư hỏng được detect gồm:

- `crack`
- `dent`
- `glass shatter`
- `lamp broken`
- `scratch`
- `tire flat`

### 2.3 Môi trường huấn luyện

Các lần train được thực hiện trên môi trường sử dụng GPU **Tesla T4 (14GB VRAM)** với:

- Python `3.12.13`
- PyTorch `2.10.0+cu128`
- Ultralytics `8.4.33` hoặc `8.4.37`

Lưu ý: log của lần train `YOLOv8n` cho thấy dữ liệu được đọc từ Google Drive, trong khi `YOLOv8s` và `YOLOv8m` được train từ đường dẫn local trong `/content/data`. Điều này có thể ảnh hưởng tới tổng thời gian huấn luyện và khiến việc so sánh thời gian train giữa ba mô hình không hoàn toàn tuyệt đối.

---

## 3. Cấu hình huấn luyện chung

Ba thực nghiệm được giữ tương đối nhất quán về cấu hình để việc so sánh công bằng hơn.

### 3.1 Tham số chính

- `epochs = 70`
- `imgsz = 640`
- `batch = 16`
- `device = 0`
- `pretrained = True`
- `optimizer = auto`
- `patience = 100`
- `deterministic = True`
- `amp = True`

### 3.2 Optimizer thực tế được chọn

Mặc dù trong lệnh train có truyền các giá trị mặc định như `lr0=0.01` và `momentum=0.937`, Ultralytics đã tự động chọn lại optimizer thành:

- `AdamW`
- `lr = 0.001`
- `momentum = 0.9`

### 3.3 Data augmentation và thiết lập liên quan

Một số augmentation/thiết lập xuất hiện nhất quán trong log:

- `fliplr = 0.5`
- `hsv_h = 0.015`
- `hsv_s = 0.7`
- `hsv_v = 0.4`
- `translate = 0.1`
- `scale = 0.5`
- `mosaic = 1.0`
- `close_mosaic = 10`
- Albumentations gồm: `Blur`, `MedianBlur`, `ToGray`, `CLAHE`

---

## 4. Kết quả từng thực nghiệm

## 4.1 YOLOv8n

### Cấu hình mô hình

- Weights khởi tạo: `yolov8n.pt`
- Số tham số: `3,012,018`
- GFLOPs: `8.2`
- Kích thước file `best.pt`: `6.2 MB`
- Thời gian train toàn bộ: `2.237 giờ`

### Kết quả validation tốt nhất

- Precision: `0.748`
- Recall: `0.678`
- mAP@0.5: `0.714`
- mAP@0.5:0.95: `0.568`

### Tốc độ suy luận

- Preprocess: `0.2 ms/image`
- Inference: `2.1 ms/image`
- Postprocess: `3.3 ms/image`

### Nhận xét

YOLOv8n là mô hình nhỏ nhất trong ba thực nghiệm, có ưu điểm rõ ràng ở kích thước gọn nhẹ và tốc độ inference nhanh. Tuy nhiên, chất lượng detection tổng thể thấp hơn hai mô hình còn lại, đặc biệt ở các lớp khó như `crack` và `scratch`. Mô hình vẫn đủ tốt để làm baseline hoặc phương án lightweight, nhưng chưa phải lựa chọn tối ưu nếu muốn cân bằng tốt giữa accuracy và deployment.

---

## 4.2 YOLOv8s

### Cấu hình mô hình

- Weights khởi tạo: `yolov8s.pt`
- Số tham số: `11,137,922`
- GFLOPs: `28.7`
- Kích thước file `best.pt`: `22.5 MB`
- Thời gian train toàn bộ: `1.907 giờ`

### Kết quả validation tốt nhất

- Precision: `0.798`
- Recall: `0.685`
- mAP@0.5: `0.735`
- mAP@0.5:0.95: `0.587`

### Tốc độ suy luận

- Preprocess: `0.2 ms/image`
- Inference: `3.6 ms/image`
- Postprocess: `2.5 ms/image`

### Nhận xét

YOLOv8s cho kết quả cân bằng nhất trong ba mô hình. Đây là mô hình có:

- **Precision cao nhất**,
- **mAP@0.5 cao nhất**,
- kích thước vẫn đủ gọn để triển khai,
- tốc độ inference vẫn nhanh.

So với `YOLOv8n`, phiên bản `s` cải thiện rõ rệt về độ chính xác nhưng chưa tăng chi phí tính toán quá mạnh. So với `YOLOv8m`, phiên bản `s` chỉ thấp hơn rất ít về `Recall` và `mAP@0.5:0.95`, trong khi nhẹ hơn đáng kể và chạy nhanh hơn rõ rệt. Đây là ứng viên tốt nhất để tích hợp vào pipeline hiện tại.

---

## 4.3 YOLOv8m

### Cấu hình mô hình

- Weights khởi tạo: `yolov8m.pt`
- Số tham số: `25,859,794`
- GFLOPs: `79.1`
- Kích thước file `best.pt`: `52.0 MB`
- Thời gian train toàn bộ: `2.377 giờ`

### Kết quả validation tốt nhất

- Precision: `0.765`
- Recall: `0.703`
- mAP@0.5: `0.728`
- mAP@0.5:0.95: `0.590`

### Tốc độ suy luận

- Preprocess: `0.3 ms/image`
- Inference: `7.7 ms/image`
- Postprocess: `3.0 ms/image`

### Nhận xét

YOLOv8m là mô hình mạnh nhất về dung lượng và chi phí tính toán. Trong ba mô hình, phiên bản `m` đạt:

- **Recall cao nhất**,
- **mAP@0.5:0.95 cao nhất**.

Điều này cho thấy mô hình có khả năng học tốt hơn ở tiêu chuẩn đánh giá nghiêm ngặt hơn và nhận diện được nhiều đối tượng hơn. Tuy nhiên, đổi lại:

- tốc độ inference chậm hơn đáng kể,
- mô hình nặng hơn nhiều,
- kích thước `best.pt` vượt ngưỡng 50MB.

Vì vậy, YOLOv8m phù hợp khi ưu tiên tối đa hóa chất lượng detection, nhưng chưa phải lựa chọn hiệu quả nhất cho pipeline hiện tại nếu cần cân bằng giữa hiệu năng và khả năng triển khai.

---

## 5. So sánh tổng hợp

| Model   | Params | GFLOPs | Train time (h) | Best weight | Precision |    Recall |   mAP@0.5 | mAP@0.5:0.95 | Inference (ms/img) |
| ------- | -----: | -----: | -------------: | ----------: | --------: | --------: | --------: | -----------: | -----------------: |
| YOLOv8n |  3.01M |    8.2 |          2.237 |      6.2 MB |     0.748 |     0.678 |     0.714 |        0.568 |                2.1 |
| YOLOv8s | 11.14M |   28.7 |          1.907 |     22.5 MB | **0.798** |     0.685 | **0.735** |        0.587 |                3.6 |
| YOLOv8m | 25.86M |   79.1 |          2.377 |     52.0 MB |     0.765 | **0.703** |     0.728 |    **0.590** |                7.7 |

### Kết luận từ bảng so sánh

- **YOLOv8n**: nhỏ nhất, nhanh nhất, phù hợp làm baseline hoặc lựa chọn lightweight.
- **YOLOv8s**: cân bằng tốt nhất giữa accuracy, tốc độ và kích thước mô hình.
- **YOLOv8m**: mạnh nhất về recall và mAP@0.5:0.95, nhưng chi phí tính toán cao hơn rõ rệt.

---

## 6. Diễn biến học theo epoch

### 6.1 YOLOv8n

Với `YOLOv8n`, metric validation tăng khá đều theo epoch. Ở giai đoạn đầu mô hình cải thiện nhanh, sau đó đi vào vùng tăng trưởng chậm hơn từ khoảng epoch 40 trở đi. Việc `best.pt` đạt `mAP@0.5 = 0.714` và `mAP@0.5:0.95 = 0.568` cho thấy mô hình vẫn hội tụ ổn định, nhưng biên cải thiện về cuối không còn lớn.

### 6.2 YOLOv8s

`YOLOv8s` cho quá trình học ổn định và là thực nghiệm có đường tăng trưởng tốt nhất về `mAP@0.5`. Sau khoảng epoch 50, metric đã tiệm cận mức tốt và những epoch cuối chủ yếu giúp tinh chỉnh thêm. Việc mô hình đạt `mAP@0.5 = 0.735` và `mAP@0.5:0.95 = 0.587` cho thấy 70 epoch là đủ để có một checkpoint mạnh cho giai đoạn hiện tại.

### 6.3 YOLOv8m

`YOLOv8m` tiếp tục cải thiện tới rất gần cuối quá trình train. Ở các epoch cuối, `mAP@0.5:0.95` đạt đỉnh `0.590`, cho thấy mô hình lớn hơn vẫn còn khả năng học tốt thêm ở cuối training. Tuy nhiên, lợi ích bổ sung so với `YOLOv8s` không quá lớn nếu so với phần chi phí inference tăng thêm.

### 6.4 Nhận xét chung

Nhìn chung, cả ba mô hình đều chưa cho thấy dấu hiệu overfit mạnh trong log đã cung cấp. Metric validation vẫn tăng hoặc giữ ổn định tới vùng epoch cuối. Điều này gợi ý rằng các lần train sau có thể tiếp tục thử:

- tăng số epoch,
- dùng early stopping chặt hơn,
- hoặc tinh chỉnh learning rate / augmentation,

để kiểm tra xem còn có thể cải thiện thêm hay không.

---

## 7. Phân tích theo lớp hư hỏng

Dựa trên kết quả validation của các checkpoint tốt nhất, có thể chia các lớp thành hai nhóm:

### 7.1 Nhóm detect tốt

Các lớp có kết quả mạnh và khá ổn định qua cả ba mô hình:

- `glass shatter`
- `lamp broken`
- `tire flat`

Đây là các lớp có đặc trưng hình ảnh nổi bật hơn, dễ tách biệt khỏi nền và ít mơ hồ hơn trong annotation.

### 7.2 Nhóm còn khó

Các lớp còn khó và cần ưu tiên cải thiện:

- `crack`
- `scratch`
- `dent`

Trong đó:

- `crack` là lớp khó nhất, mAP còn thấp ở cả ba mô hình.
- `scratch` cũng chưa cao, có thể do đặc trưng mảnh, nhỏ, dễ bị ảnh hưởng bởi ánh sáng và góc chụp.
- `dent` ở mức trung bình, có khả năng bị ảnh hưởng bởi phản chiếu bề mặt xe và chất lượng ảnh.

Điều này rất quan trọng cho pipeline tổng thể, vì `scratch` và `dent` lại là hai loại hư hỏng có khả năng xuất hiện nhiều trong bài toán định giá xe cũ. Nếu detection chưa đủ mạnh ở các lớp này, tầng pricing phía sau cũng sẽ bị ảnh hưởng.

---

## 8. Quyết định chọn mô hình hiện tại

### Mô hình được ưu tiên tích hợp: `YOLOv8s`

Lý do chọn `YOLOv8s` ở giai đoạn hiện tại:

1. đạt **mAP@0.5 cao nhất**;
2. đạt **precision cao nhất**;
3. tốc độ suy luận vẫn nhanh;
4. kích thước mô hình còn gọn, dễ tích hợp vào pipeline;
5. cân bằng tốt hơn giữa chất lượng và chi phí triển khai so với `YOLOv8m`.

### Vai trò của hai mô hình còn lại

- **YOLOv8n**: giữ lại như baseline nhẹ, hữu ích nếu cần so sánh hoặc build phiên bản tối ưu tốc độ.
- **YOLOv8m**: giữ lại như upper-bound reference cho chất lượng detection, đặc biệt nếu về sau có nhu cầu ưu tiên recall hơn tốc độ.

---

## 9. Hạn chế của giai đoạn train hiện tại

1. Mới chỉ benchmark trên ba biến thể mặc định `n/s/m`, chưa có tuning sâu về hyperparameter.
2. Chưa kiểm soát hoàn toàn yếu tố I/O giữa các lần train, nên thời gian train không phản ánh hoàn toàn năng lực mô hình.
3. Chưa có thêm các thực nghiệm như:
   - thay đổi `imgsz`,
   - thay đổi `batch size`,
   - freeze/unfreeze backbone,
   - train lâu hơn 70 epoch,
   - class reweighting hoặc sampling cho các lớp khó.
4. Chưa đánh giá trên tập ảnh thực tế từ pipeline hoàn chỉnh sau khi nối với bước severity classification.

---

## 10. Hướng cải thiện tiếp theo

### 10.1 Về training

- Train lại cả ba mô hình trong cùng một điều kiện I/O để so sánh công bằng hơn.
- Thử tăng epoch lên `100` hoặc `120` và dùng early stopping phù hợp.
- Theo dõi thêm `results.csv`, loss curves, PR curve và confusion matrix để kết luận chắc hơn.
- Thử tinh chỉnh augmentation cho các lớp khó như `crack` và `scratch`.

### 10.2 Về dữ liệu

- Rà soát lại chất lượng annotation ở các lớp khó.
- Tăng dữ liệu cho `crack`, `scratch`, `dent`.
- Bổ sung ảnh có điều kiện ánh sáng và góc chụp đa dạng hơn.

### 10.3 Về tích hợp hệ thống

- Dùng output của `YOLOv8s` để crop hoặc trích xuất vùng hư hỏng phục vụ CNN severity classification.
- Từ detection output, xây dựng các feature như:
  - số lượng hư hỏng theo loại,
  - diện tích vùng hư hỏng,
  - vùng hư hỏng lớn nhất,
  - tỷ lệ vùng hư hỏng trên ảnh,
  - đặc trưng tổ hợp giữa loại hư hỏng và severity.
- Sau đó nối sang bước rule-based adjustment để tạo ra damage-aware price estimate.

---

## 11. Kết luận chung

Giai đoạn benchmark YOLOv8 cho thấy cả ba mô hình đều học được bài toán car damage detection ở mức tốt. Trong đó:

- `YOLOv8n` phù hợp cho nhu cầu gọn nhẹ và tốc độ cao.
- `YOLOv8m` đạt recall và mAP@0.5:0.95 tốt nhất nhưng nặng hơn đáng kể.
- `YOLOv8s` là lựa chọn hợp lý nhất ở thời điểm hiện tại vì cho **sự cân bằng tốt nhất giữa độ chính xác, tốc độ và khả năng triển khai**.

Với kết quả này, `YOLOv8s` nên được dùng làm detector chính trong pipeline của dự án dự đoán giá xe cũ, trong khi `YOLOv8m` có thể giữ làm mốc tham chiếu khi cần tối ưu chất lượng ở các phiên bản sau.
