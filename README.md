# Thuc_Tap_Co_So_Project

## Streamlit MVP Demo

This repository includes a Streamlit MVP for a used car price prediction demo. The UI collects car info,
accepts images, runs a pipeline (tabular pricing, YOLO damage detection, severity grading, rule-based
adjustment), and shows visual results.

## Quick Start

1. Create a virtual environment and install dependencies:
   pip install -r requirements.txt
2. Run the app:
   streamlit run app.py

## Model Integration

- Damage detection uses Models/best_100_800sz.pt (YOLOv8) in dich_vu/phat_hien_hu_hong.py.
- Severity classification uses Models/cnn_car.pkl in dich_vu/muc_do_hu_hong.py.
- Pricing remains heuristic; replace du_doan_gia_co_ban and tinh_dieu_chinh_gia in
  dich_vu/dinh_gia.py.

## Project Structure

- app.py
- giao_dien/ (Streamlit layout and components)
- dich_vu/ (prediction services)
- tien_ich/ (formatters, sample data, session state helpers)
- assets/

## Resources

- Source Dataset: https://drive.google.com/drive/folders/1l8PDbF6fKk7dPHlFM9dzpRb4kd3I8ZU2?usp=sharing
- Train Report: https://drive.google.com/drive/folders/1VIUuRqvohVyurhAZm-QcN7PRIgavPlwz?usp=drive_link
