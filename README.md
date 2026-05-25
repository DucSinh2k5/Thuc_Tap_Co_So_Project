# Thuc_Tap_Co_So_Project

## Demo MVP Streamlit

Kho này bao gồm một MVP Streamlit để demo dự đoán giá xe cũ. Giao diện thu thập thông tin xe,
nhận ảnh, chạy pipeline (định giá theo bảng, phát hiện hư hỏng bằng YOLO, phân loại mức độ,
điều chỉnh theo luật), và hiển thị kết quả trực quan.

## Bắt đầu nhanh

1. Tạo môi trường ảo và cài phụ thuộc:
   pip install -r requirements.txt
2. Chạy ứng dụng:
   streamlit run app.py

## Tích hợp mô hình

- Phát hiện hư hỏng dùng Models/best_100_800sz.pt (YOLOv8) trong dich_vu/phat_hien_hu_hong.py.
- Phân loại mức độ hư hỏng dùng Models/cnn_car.pkl trong dich_vu/muc_do_hu_hong.py.
- Định giá hiện còn theo luật; hãy thay thế du_doan_gia_co_ban và tinh_dieu_chinh_gia trong
  dich_vu/dinh_gia.py.

## Cấu trúc dự án

- app.py
- giao_dien/ (bố cục và thành phần Streamlit)
- dich_vu/ (dịch vụ dự đoán)
- tien_ich/ (định dạng, dữ liệu mẫu, trợ giúp trạng thái phiên)
- assets/

## Tài nguyên

- Bộ dữ liệu nguồn: https://drive.google.com/drive/folders/1l8PDbF6fKk7dPHlFM9dzpRb4kd3I8ZU2?usp=sharing
- Báo cáo huấn luyện: https://drive.google.com/drive/folders/1VIUuRqvohVyurhAZm-QcN7PRIgavPlwz?usp=drive_link
