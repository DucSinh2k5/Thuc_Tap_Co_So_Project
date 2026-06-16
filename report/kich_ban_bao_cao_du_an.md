# Kich Ban Bao Cao Du An

## 0. Cach Dung File Nay

File nay la ghi chu noi khi bao cao. Em co the dung theo 3 muc:

- **Noi ngan 7-10 phut:** doc cac muc "Can noi".
- **Bi hoi sau:** xem muc "Neu thay hoi".
- **Can mo code/minh chung:** xem muc "File/hang can chi".

Chu de bao cao nen nhan manh:

> Du an cua em la mot he thong uoc luong gia xe cu co xet den tinh trang hu hong ngoai that. He thong khong chi du doan gia tu thong tin bang, ma con nhan anh xe, phat hien hu hong, danh gia muc do hu hong va dieu chinh gia cuoi cung.

---

## 1. Mo Dau Bao Cao

### Can noi

Kinh thua thay, trong do an nay em xay dung mot he thong demo du doan gia xe cu co xet den tinh trang ngoai that cua xe.

Bai toan cua em gom hai loai du lieu:

- **Du lieu bang**: cac thong tin xe nhu hang xe, dong xe, nam san xuat, so km da di, hop so, loai nhien lieu, dung tich dong co, cong suat...
- **Du lieu anh**: anh xe de phat hien cac hu hong nhu vet xuoc, vet mop, vet nut, den vo, kinh vo, lop xep...

Dau ra cua he thong gom:

- **Base price**: gia co so du doan tu thong tin xe bang mo hinh XGBoost.
- **Damage detection**: danh sach cac vung hu hong tren anh bang YOLO.
- **Severity classification**: muc do hu hong `minor`, `moderate`, `severe` bang CNN ConvNeXt-Tiny.
- **Final adjusted price**: gia cuoi cung sau khi tru theo muc hu hong bang tang rule-based adjustment.

### Diem can nhan manh

Day khong phai la mot mo hinh multimodal end-to-end hoc truc tiep gia sau hu hong, vi hien tai khong co ground truth cho **final price after damage**. Vi vay em thiet ke theo huong **hybrid pipeline**:

1. XGBoost hoc gia thi truong co so tu du lieu bang.
2. YOLO va CNN trich xuat tin hieu hu hong tu anh.
3. Mot tang luat ket hop cac tin hieu nay de dieu chinh gia.

---

## 2. Tong Quan Kien Truc He Thong

### Can noi

He thong cua em co 3 nhanh chinh:

| Nhanh | Nhiem vu | Mo hinh/file chinh | Dau ra |
| --- | --- | --- | --- |
| Tabular pricing | Du doan gia co so cua xe | XGBoost, `Models/model.pkl` | `base_price` |
| Damage detection | Phat hien vung hu hong tren anh | YOLOv8s, `Models/best.pt` | class, confidence, bbox, area_ratio |
| Severity classification | Phan loai muc do hu hong | ConvNeXt-Tiny, `Models/ConvNeXt.pkl` | minor/moderate/severe |
| Rule adjustment | Tinh gia sau khi tru hu hong | `dich_vu/dinh_gia.py` | `final_price` |

Luong chay trong app:

```text
Thong tin xe + anh upload
        |
        |-- app.py chuan hoa thong tin xe
        |
        |-- XGBoost du doan base price
        |
        |-- neu co anh:
        |       |-- YOLO phat hien hu hong
        |       |-- ConvNeXt phan loai muc do
        |       |-- rule-based adjustment tinh tien tru
        |
        |-- Streamlit hien thi base price, damage deduction, final price
```

### File/hang can chi

- `app.py`
  - `chay_ung_dung()`: khoi tao giao dien Streamlit.
  - `chay_pipeline(thong_tin_xe, danh_sach_anh)`: noi tat ca nhanh mo hinh lai.
  - `chuan_hoa_thong_tin_xe(thong_tin_xe)`: bien input form thanh schema dung voi model tabular.
- `dich_vu/dinh_gia.py`
  - `du_doan_gia_co_ban(thong_tin_xe)`: goi XGBoost.
  - `tinh_dieu_chinh_gia(...)`: tinh tien tru theo hu hong.
- `dich_vu/phat_hien_hu_hong.py`
  - `phat_hien_hu_hong(danh_sach_anh)`: chay YOLO.
  - `ve_bbox_anh(...)`: ve bounding box de hien thi.
- `dich_vu/muc_do_hu_hong.py`
  - `phan_loai_muc_do(danh_sach_anh)`: chay ConvNeXt-Tiny.
  - `tong_hop_muc_do(...)`: tong hop so damage va muc do.

---

## 3. Du Lieu Su Dung

### 3.1. Du lieu bang cho bai toan gia xe

### Can noi

Du lieu bang duoc lay tu bo **used-cars-price-prediction** tren Kaggle. Cac file trong du an:

- `Datasets/train-dataset.csv`: du lieu train, khoang 6019 dong du lieu.
- `Datasets/test-dataset.csv`: du lieu test, khoang 1234 dong du lieu.
- `Datasets/train_cleaned.csv`: ban du lieu sau tien xu ly.

Cot muc tieu la:

- `Price` sau khi rename thanh `Gia_theo_lakh`.

Mot so cot dau vao:

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

### Kho khan voi du lieu bang

- `New_Price` thieu rat nhieu, khoang 86% trong train.
- Cac cot `Mileage`, `Engine`, `Power` co don vi dang text nhu `18.2 kmpl`, `1199 CC`, `88.7 bhp`, can tach so.
- Co gia tri bat thuong nhu so km rat lon, so ghe bang 0, cong suat `null bhp`.
- Cot ten xe co rat nhieu gia tri unique, nen can gop top xe/hang xe de tranh qua nhieu category hiem.

### 3.2. Du lieu anh cho YOLO

### Can noi

Nhanh detection dung du lieu tu bo **car-damage-detection**, phan `CarDD_COCO`. Du lieu goc o dinh dang COCO, sau do duoc chuyen sang dinh dang YOLOv8 thong qua Roboflow.

Ban dau co khoang **4000 anh**. Sau khi benchmark, em thay cac lop kho la:

- `crack`
- `scratch`
- `dent`

Nen em bo sung them **2307 anh** tu Roboflow Universe cho cac lop nay. Dataset sau merge co khoang **6307 anh**.

### 3.3. Du lieu anh cho severity classification

### Can noi

Nhanh severity classification dung anh full va nhan:

- `minor`
- `moderate`
- `severe`

Diem dang chu y la trong pipeline hien tai, CNN nhan **anh full nguoi dung upload**, khong nhan crop bbox tu YOLO. YOLO va CNN xu ly song song:

- YOLO cho biet co hu hong gi, o dau, dien tich bao nhieu.
- CNN cho biet muc do tong quat cua anh la nhe, vua hay nang.

---

## 4. Quy Trinh Lam Viec Voi Du Lieu Bang

### Can noi

Voi nhanh tabular, em lam theo cac buoc:

1. **EDA truoc xu ly**
   - Doc du lieu, kiem tra missing, kieu du lieu, thong ke co ban.
   - File tham khao: `tests/test_tabular/test-dataset-tabular.ipynb`, `src/EDA_before.py`, `Quan_sat/eda_before.txt`.

2. **Chuan hoa cot va lam sach**
   - Rename cot sang ten ro nghia bang tieng Viet.
   - Loc cac dong khong hop le.
   - Tach so tu cac cot co don vi.
   - Map category nhu hop so, nhien lieu, so lan so huu.
   - File: `src/load_data_and_cleaning.py`.

3. **Feature engineering**
   - Tao `Tuoi_xe` tu nam san xuat.
   - Tao `Hang_xe` tu ten xe.
   - Tao `Km_moi_nam`.
   - Tao `Chay_nhieu`.
   - Tao `log_Quang_duong_da_di(km)`.
   - Tao `Top_xe` de gop cac dong xe pho bien, xe it gap dua ve `Other`.
   - File: `src/feature_engineering.py`.

4. **Xu ly missing va outlier**
   - Numeric dung median.
   - Categorical dung mode/Unknown.
   - Outlier duoc clip theo IQR.
   - Quan trong: cac tham so median, imputer, top category, outlier bounds duoc fit tren train roi dung lai cho val/test de tranh data leakage.

5. **Tien xu ly cho model**
   - Numeric: impute median.
   - Categorical: `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`.
   - File: `src/preprocessing.py`.

6. **Train va danh gia**
   - Chia train/validation 80/20.
   - So sanh baseline, Random Forest va XGBoost.
   - Chon XGBoost de trien khai.
   - File: `src/train_and_evaluate.py`, `src/main.py`.

### Cac ham quan trong

| File | Ham | Vai tro |
| --- | --- | --- |
| `src/load_data_and_cleaning.py` | `doi_ten_cot(df)` | Rename cot ve schema thong nhat |
| `src/load_data_and_cleaning.py` | `loai_bo_hang_ban(df)` | Loc dong khong hop le |
| `src/load_data_and_cleaning.py` | `chuyen_cot_sang_so(df)` | Tach so tu cot co don vi |
| `src/load_data_and_cleaning.py` | `chuyen_cot_sang_category(df)` | Ma hoa nhien lieu, hop so, so huu |
| `src/feature_engineering.py` | `tao_moi_feature(df, km_median=None)` | Tao tuoi xe, hang xe, km moi nam, log km |
| `src/feature_engineering.py` | `xu_ly_gia_tri_thieu(df, imputers=None)` | Fit/transform missing value |
| `src/feature_engineering.py` | `gioi_han_xe(...)`, `gioi_han_hang_xe(...)` | Gop category hiem |
| `src/feature_engineering.py` | `xu_ly_outlier(df, bounds=None)` | Clip outlier theo IQR |
| `src/preprocessing.py` | `tien_xu_ly(num, cat)` | Tao ColumnTransformer |
| `src/train_and_evaluate.py` | `compare_models(...)` | So sanh RF va XGBoost |
| `src/train_and_evaluate.py` | `train_model(...)` | Train XGBoost final |
| `src/train_and_evaluate.py` | `feature_importance_report(...)` | Xuat feature importance |
| `src/train_and_evaluate.py` | `save(...)` | Luu model deploy vao `Models/model.pkl` |

---

## 5. Ket Qua Nhanh Tabular

### Can noi

Ket qua baseline:

| Baseline | RMSE | MAE | R2 |
| --- | ---: | ---: | ---: |
| Mean baseline | 5.7028 | 4.6119 | -0.0044 |
| Median baseline | 5.9453 | 4.0818 | -0.0916 |

Ket qua so sanh model:

| Model | RMSE | MAE | R2 validation |
| --- | ---: | ---: | ---: |
| Random Forest | 1.6074 | 0.9962 | 0.9202 |
| XGBoost | 1.4144 | 0.8856 | 0.9382 |

Em chon **XGBoost** vi:

- R2 validation cao hon Random Forest.
- RMSE va MAE thap hon.
- Phu hop voi du lieu bang co ca numeric va categorical da encode.
- De luu va tich hop vao app bang `joblib`.

### Feature quan trong

Theo `feature_importance_xgb.csv`, cac feature quan trong nhat:

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

### Cau noi goi y

> Ket qua nay cho thay mo hinh hoc duoc cac yeu to hop ly ve mat thuc te: cong suat, tuoi xe, dung tich, hang xe va hop so deu la cac yeu to anh huong manh den gia xe cu.

---

## 6. Quy Trinh Lam Viec Voi YOLO Damage Detection

### Can noi

Voi nhanh detection, muc tieu cua em la phat hien cac vung hu hong tren anh xe. Em dung YOLO vi day la mo hinh object detection nhanh, phu hop voi ung dung demo can inference truc tiep.

Em da thu cac huong:

- YOLOv8n: nhe, nhanh, dung lam baseline.
- YOLOv8s: can bang tot hon giua toc do va do chinh xac.
- YOLOv8m: lon hon nhung chi phi cao, khong phu hop bang YOLOv8s cho demo.
- Faster R-CNN ResNet50-FPN: dung de so sanh phu.

Sau benchmark, em chon **YOLOv8s** va cai thien bang cach mo rong dataset, tap trung vao cac lop kho `crack`, `scratch`, `dent`.

### Cau hinh YOLO chinh

| Thong so | Gia tri |
| --- | --- |
| Model | YOLOv8s |
| Dataset | `merge_Data` |
| Anh goc | khoang 4000 anh |
| Anh bo sung | 2307 anh |
| Tong anh | khoang 6307 anh |
| Epochs | 100 |
| Batch size | 16 |
| Image size | 640 |
| Checkpoint app | `Models/best.pt` |
| Inference conf | 0.25 |
| Inference IoU | 0.45 |

### Ket qua YOLO

Ket qua tu `Quan_sat/yolo_car_report/results.csv`:

| Moc | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| Epoch co mAP50 cao nhat | ~0.801 | ~0.682 | ~0.726 | ~0.568 |
| Epoch co mAP50-95 cao nhat | - | - | - | ~0.572 |

So sanh phu voi Faster R-CNN:

| Model | mAP@0.5 | mAP@0.5:0.95 | mAP@0.75 |
| --- | ---: | ---: | ---: |
| YOLOv8s merge_Data | 0.726 | 0.568 | - |
| Faster R-CNN ResNet50-FPN | 0.306 | 0.128 | 0.084 |

### Cau noi goi y

> Em khong chi chon model theo mot lan train, ma co benchmark nhieu bien the. YOLOv8s duoc chon vi can bang giua do chinh xac, toc do inference va kich thuoc mo hinh. Viec bo sung du lieu cho `crack`, `scratch`, `dent` la do ba lop nay thuong nho, manh va de bi bo sot hon.

### File/hang can chi

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

## 7. Quy Trinh Lam Viec Voi CNN Severity Classification

### Can noi

Nhanh CNN co nhiem vu phan loai muc do hu hong cua anh thanh 3 muc:

- `minor`
- `moderate`
- `severe`

Em thu nhieu backbone theo huong transfer learning. Quy trinh train gom:

1. Phase 1: dong bang backbone, train classifier/head.
2. Phase 2: fine-tune cac block cuoi cua backbone va classifier de mo hinh thich nghi voi du lieu hu hong xe.

### Ket qua cac backbone CNN

| Notebook | Backbone | Best validation metric | Test accuracy | Test macro F1 |
| --- | --- | ---: | ---: | ---: |
| `resnet_18_new.ipynb` | ResNet18 | val_acc = 0.8209 | 0.6564 | 0.6490 |
| `EfficientNet_B0.ipynb` | EfficientNet-B0 | val_macro_f1 = 0.7945 | 0.6821 | 0.6817 |
| `Efficient_B2.ipynb` | EfficientNet-B2 | val_macro_f1 = 0.8022 | 0.6974 | 0.6954 |
| `ResNet50.ipynb` | ResNet50 | val_acc = 0.8128 | 0.7026 | 0.7043 |
| `ConvNeXt_Tiny.ipynb` | ConvNeXt-Tiny | val_macro_f1 = 0.8342 | 0.7077 | 0.7084 |

Em chon **ConvNeXt-Tiny** vi:

- Validation macro F1 cao nhat: 0.8342.
- Test accuracy cao nhat: 0.7077.
- Test macro F1 cao nhat: 0.7084.
- Tong quat hoa tot hon ResNet18 trong bai toan co nhieu chi tiet nho va ranh gioi nhan mo.

### Demo dinh tinh

Trong bao cao co mot vi du cung mot anh test:

| Model | Du doan | Xac suat |
| --- | ---: | ---: |
| ResNet18 | SEVERE | 60.61% |
| EfficientNet-B0 | MINOR | 42.60% |
| EfficientNet-B2 | SEVERE | 42.31% |
| ResNet50 | SEVERE | 40.88% |
| ConvNeXt-Tiny | MODERATE | 44.35% |

Nhan dung cua anh do la `moderate`, nen ConvNeXt-Tiny la mo hinh dung trong vi du nay.

### File/hang can chi

- `dich_vu/muc_do_hu_hong.py`
  - `CAC_MUC_DO = ["minor", "moderate", "severe"]`
  - `DUONG_DAN_MO_HINH = Models/ConvNeXt.pkl`
  - `BIEN_DOI_DANH_GIA = ConvNeXt_Tiny_Weights.DEFAULT.transforms()`
  - `_tao_convnext_tiny()`
  - `_tai_mo_hinh_muc_do()`
  - `phan_loai_muc_do(danh_sach_anh)`
  - `tong_hop_muc_do(danh_sach_phat_hien, danh_sach_muc_do)`
- `train/ConvNeXt_Tiny.ipynb`: notebook train model chinh.

### Gioi han can noi ro

> Accuracy cua CNN khoang 70.77%, nen em khong xem severity la chan ly tuyet doi. Em xem no nhu mot tin hieu ho tro cho tang dieu chinh gia. Lop `moderate` kho nhat vi nam giua `minor` va `severe`.

---

## 8. Tang Dieu Chinh Gia Theo Hu Hong

### Can noi

Sau khi co:

- base price tu XGBoost,
- class/area/confidence tu YOLO,
- severity tu ConvNeXt,

em dung mot tang rule-based de tinh ti le tru gia. Ly do dung rule-based la vi hien tai khong co nhan ground truth cho gia xe sau hu hong.

Cong thuc tong quat trong code:

```text
diem_hu_hong = min(MUC_GIAM_TOI_DA, tong_diem * HE_SO_DIEM_SANG_TI_LE)
tien_tru = gia_co_ban * diem_hu_hong
gia_sau = gia_co_ban - tien_tru
```

Thong so chinh:

| Thong so | Gia tri | Y nghia |
| --- | ---: | --- |
| `MUC_GIAM_TOI_DA` | 0.3 | Tru toi da 30% gia tri xe |
| `HE_SO_DIEM_SANG_TI_LE` | 0.012 | Doi damage score sang deduction rate |
| `LAKH_INR_SANG_VND` | 30,000,000 | Doi gia du doan tu lakh sang VND |

Trong so theo lop hu hong:

| Lop | Trong so |
| --- | ---: |
| `scratch` | 0.75 |
| `dent` | 1.0 |
| `crack` | 1.15 |
| `tire_flat` | 1.1 |
| `glass_broken` | 1.25 |
| `lamp_broken` | 1.35 |

Ngoai ra, diem tru con phu thuoc vao:

- `severity_score`: minor = 1, moderate = 2, severe = 3.
- `area_ratio`: bbox chiem bao nhieu dien tich anh.
- `confidence`: do tin cay cua YOLO.
- `so_lan_da_gap`: neu cung mot lop lap lai nhieu lan thi co he so giam dan de tranh tru qua manh.

### Cau noi goi y

> Tang rule-based nay giup em ket hop duoc dau ra cua cac mo hinh rieng le thanh mot ket qua co y nghia voi bai toan: gia cuoi cung. No cung minh bach hon, vi em co the giai thich tai sao gia bi tru: do co bao nhieu damage, lop nao, muc do nao va dien tich anh huong bao nhieu.

### File/hang can chi

- `dich_vu/dinh_gia.py`
  - `TRONG_SO_LOP`
  - `HE_SO_DIEM_SANG_TI_LE`
  - `MUC_GIAM_TOI_DA`
  - `du_doan_gia_co_ban(thong_tin_xe)`
  - `tinh_dieu_chinh_gia(gia_co_ban, danh_sach_phat_hien, danh_sach_muc_do)`

---

## 9. Tich Hop Ung Dung Streamlit

### Can noi

Sau khi co cac mo hinh rieng le, em tich hop thanh mot app Streamlit de demo.

Nguoi dung co the:

1. Chon hang xe va dong xe.
2. Nhap thong tin xe: nam san xuat, so km, hop so, nhien lieu, so ghe, dung tich dong co, cong suat...
3. Upload anh xe neu muon tinh gia co xet damage.
4. Bam `Analyze Car`.
5. He thong hien:
   - anh goc va anh co bounding box,
   - bang detection,
   - bang severity,
   - base price,
   - damage deduction,
   - final adjusted price.

### File giao dien

| File | Vai tro |
| --- | --- |
| `app.py` | Entry point, noi pipeline |
| `giao_dien/bo_cuc.py` | CSS va layout dau trang |
| `giao_dien/thanh_phan.py` | Form input, upload anh, hien ket qua |
| `tien_ich/du_lieu_mau.py` | Gia tri mac dinh, danh sach fuel/transmission |
| `tien_ich/dinh_dang.py` | Format VND, phan tram |
| `tien_ich/trang_thai.py` | Session state Streamlit |

### Demo noi truc tiep

Khi demo, nen noi theo thu tu:

1. Day la form thong tin xe. Brand/model duoc lay tu dataset de giam sai lech category voi model.
2. Khi khong upload anh, he thong chi du doan gia co so bang XGBoost.
3. Khi upload anh, app se chay them YOLO va ConvNeXt.
4. YOLO ve bbox va tra bang class, confidence, area ratio.
5. ConvNeXt tra severity cua anh.
6. Cuoi cung, rule-based adjustment tinh tien tru va gia sau dieu chinh.

Lenh chay:

```powershell
streamlit run app.py
```

---

## 10. Nhung Gi Da Dat Duoc

### Can noi

Qua du an, em da dat duoc cac ket qua sau:

1. **Xay dung pipeline tabular hoan chinh**
   - EDA, cleaning, feature engineering, preprocessing, train/evaluate, save model.
   - XGBoost dat R2 validation **0.9382**.

2. **Xay dung nhanh damage detection**
   - Benchmark YOLOv8n, YOLOv8s, YOLOv8m.
   - Mo rong dataset co muc tieu cho `crack`, `scratch`, `dent`.
   - YOLOv8s tren `merge_Data` dat mAP50 khoang **0.726**.

3. **Xay dung nhanh severity classification**
   - Thu nhieu backbone CNN: ResNet18, EfficientNet-B0, EfficientNet-B2, ResNet50, ConvNeXt-Tiny.
   - Chon ConvNeXt-Tiny voi test accuracy **0.7077**, test macro F1 **0.7084**.

4. **Tich hop thanh ung dung demo**
   - Streamlit app nhan thong tin xe va anh.
   - Hien detection, severity, base price va final price.

5. **Co bao cao va artifact**
   - `report/dataset_report.md`
   - `report/training_report.md`
   - `Quan_sat/model_comparison_report.txt`
   - `Quan_sat/yolo_car_report/`
   - `Models/model.pkl`, `Models/best.pt`, `Models/ConvNeXt.pkl`

---

## 11. Kho Khan Trong Qua Trinh Lam

### Bang kho khan va cach xu ly

| Kho khan | Mo ta | Cach em xu ly |
| --- | --- | --- |
| Du lieu bang co nhieu don vi text | `Mileage`, `Engine`, `Power` khong phai so thuan | Viet ham tach so trong `chuyen_cot_sang_so` |
| Missing value | `New_Price` thieu rat nhieu, mot so cot nhu `Power`, `Engine`, `Seats` thieu | Loai cot qua thieu, impute median/mode |
| Category qua nhieu | Ten xe co nhieu gia tri unique | Tao `Top_xe`, `Hang_xe`, gop gia tri hiem ve `Other` |
| Outlier | So km, gia, cong suat co gia tri cuc doan | Clip theo IQR trong `xu_ly_outlier` |
| Tranh data leakage | De bi fit imputer/top category tren ca test | Fit tren train, reuse cho val/test |
| Damage nho, kho detect | `scratch`, `dent`, `crack` nho va de bi anh huong boi anh sang | Bo sung 2307 anh co chu dich cho cac lop nay |
| Annotation noise | Test set co the thieu nhan, lam mo hinh phat hien dung bi tinh thanh false positive | Khong chi nhin mAP, co xem inference dinh tinh |
| Severity kho | `moderate` nam giua minor/severe, de nham | Benchmark nhieu backbone, chon ConvNeXt-Tiny |
| Khong co nhan final price sau damage | Khong the train supervised final adjusted price | Dung rule-based adjustment minh bach |
| Tich hop model | Can dong bo schema input app voi schema train model | Viet `chuan_hoa_thong_tin_xe` trong `app.py` |

### Cau noi goi y

> Kho khan lon nhat cua em khong chi la train model, ma la lam sao ket hop ba bai toan khac nhau thanh mot he thong co the demo duoc. Vi du, du lieu gia xe va du lieu damage khong dong bo theo tung chiec xe, nen em phai thiet ke theo huong modular pipeline thay vi end-to-end.

---

## 12. Han Che Hien Tai

### Can noi

Du an hien tai van co mot so han che:

1. **Chua co dataset multimodal dong bo**
   - Anh damage va du lieu gia xe den tu cac nguon khac nhau.
   - Chua co moi quan he truc tiep theo tung chiec xe giua damage va gia ban.

2. **Chua co ground truth cho final adjusted price**
   - Gia cuoi cung sau khi xet damage hien duoc tinh bang rule-based.
   - Chua phai mo hinh supervised hoc tu nhan that.

3. **CNN severity con gioi han**
   - Test accuracy khoang 70.77%.
   - Lop `moderate` con nhap nhang.

4. **Severity hien chay tren anh full**
   - Neu anh co nhieu vung hư hong, mot severity cho ca anh co the chua chi tiet.
   - Huong cai tien la crop tung bbox tu YOLO roi phan loai severity rieng cho tung damage.

5. **App moi o muc MVP**
   - Van can dong goi path/model tot hon neu chuyen sang may khac.
   - Can them unit test/automation test that su.

---

## 13. Huong Phat Trien

### Can noi

Neu tiep tuc phat trien, em se lam cac huong:

1. **Cai thien du lieu**
   - Thu thap du lieu xe co anh hu hong va gia thuc te sau khi dinh gia.
   - Bo sung hard negatives va kiem tra duplicate/label noise.

2. **Cai thien severity**
   - Cat crop bbox tu YOLO roi cho CNN phan loai tung vung damage.
   - Hoac dung multi-task model vua detect vua danh severity.

3. **Cai thien price adjustment**
   - Neu co nhan gia sau hu hong, co the train model hoc truc tiep ti le tru gia.
   - Hien tai rule-based minh bach, nhung can du lieu that de hieu chinh trong so.

4. **Cai thien deploy**
   - Bo hard-code path.
   - Dong goi config.
   - Them test tu dong cho preprocessing va prediction.
   - Cache model va toi uu inference.

---

## 14. Neu Thay Hoi

### Vi sao chon XGBoost thay vi Random Forest?

Em co so sanh tren validation. Random Forest dat R2 0.9202, con XGBoost dat R2 0.9382, RMSE va MAE cung thap hon. Ngoai ra XGBoost phu hop voi du lieu bang da xu ly va de trien khai trong pipeline.

### Vi sao dung OrdinalEncoder cho categorical?

Vi model tree-based nhu XGBoost/Random Forest co the lam viec voi so nguyen encode tu category. Em dung `handle_unknown="use_encoded_value", unknown_value=-1` de khi app gap category moi khong bi loi.

### Em co tranh data leakage khong?

Co. Trong `src/main.py`, em chia train/validation truoc. Cac thong so nhu median km, imputer, top category, outlier bounds duoc fit tren train roi moi reuse cho validation/test.

### Vi sao khong dung mot mo hinh end-to-end cho ca anh va bang?

Vi hien tai du lieu anh damage va du lieu gia xe khong dong bo theo tung chiec xe, va khong co nhan gia sau hu hong. Neu train end-to-end se khong co target dung. Vi vay em chon thiet ke modular: XGBoost du doan gia co so, YOLO/CNN trich xuat damage, rule-based tinh dieu chinh.

### Vi sao chon YOLOv8s?

YOLOv8n nhanh nhung kha nang hoc damage nho han che. YOLOv8m lon hon nhung chi phi cao, khong phu hop bang cho demo. YOLOv8s can bang giua do chinh xac va toc do, dat mAP50 khoang 0.726 tren run `merge_Data`.

### Vi sao bo sung du lieu cho `crack`, `scratch`, `dent`?

Vi day la cac lop kho: vung hu hong nho, manh, de bi anh huong boi anh sang va goc chup. Bo sung co muc tieu giup mo hinh co them mau hoc cho cac truong hop kho thay vi tang du lieu dai tra.

### Vi sao chon ConvNeXt-Tiny?

Em benchmark 5 backbone. ConvNeXt-Tiny co validation macro F1 0.8342, test accuracy 0.7077 va test macro F1 0.7084, tot nhat trong cac model da thu.

### Tai sao CNN phan loai anh full ma khong phan loai tung bbox?

Trong ban hien tai, de phu hop voi du lieu severity va app demo, em dung anh full. Tuy nhien em nhan thuc day la han che. Huong phat trien tot hon la crop tung bbox tu YOLO roi cho CNN phan loai severity tung damage.

### Rule-based adjustment co chu quan khong?

Co mot phan chu quan, vi chua co ground truth gia sau hu hong. Tuy nhien cach nay minh bach va giai thich duoc: moi lop damage co trong so, severity co diem, bbox co dien tich, confidence co he so. Khi co du lieu that, cac he so nay co the duoc hoc hoac hieu chinh lai.

---

## 15. Kich Ban Noi 7-10 Phut

### 1 phut - Gioi thieu bai toan

Kinh thua thay, du an cua em la he thong demo du doan gia xe cu co xet den tinh trang hu hong ngoai that. Dau vao gom thong tin bang cua xe va anh xe. Dau ra la gia co so, cac hu hong phat hien tren anh, muc do hu hong va gia sau dieu chinh.

### 1 phut - Kien truc

He thong gom ba nhanh: XGBoost cho gia co so, YOLOv8s cho phat hien hu hong, ConvNeXt-Tiny cho phan loai muc do. Cuoi cung em dung rule-based adjustment de tinh gia cuoi cung. Em chon huong modular vi du lieu gia va du lieu anh khong dong bo theo tung xe va chua co nhan gia sau hu hong.

### 2 phut - Tabular pipeline

Voi du lieu bang, em lam EDA, rename cot, tach so tu cac cot co don vi, xu ly missing, tao feature moi nhu tuoi xe, hang xe, km moi nam, log km va Top_xe. Sau do em chia train/validation, preprocess numeric/categorical va train model. Ket qua XGBoost dat R2 validation 0.9382, tot hon Random Forest 0.9202, nen em chon XGBoost lam model deploy.

### 2 phut - Image pipeline

Voi damage detection, em benchmark YOLOv8n, YOLOv8s, YOLOv8m. YOLOv8s can bang tot nhat nen duoc chon. Sau do em mo rong dataset tu khoang 4000 anh len khoang 6307 anh bang cach bo sung 2307 anh cho cac lop kho `crack`, `scratch`, `dent`. Ket qua YOLOv8s tren `merge_Data` dat mAP50 khoang 0.726.

Voi severity classification, em thu ResNet18, EfficientNet-B0, EfficientNet-B2, ResNet50 va ConvNeXt-Tiny. ConvNeXt-Tiny tot nhat voi test accuracy 0.7077 va macro F1 0.7084, nen em dung lam model chinh.

### 1 phut - Tich hop app

Em tich hop cac thanh phan vao Streamlit. Nguoi dung nhap thong tin xe va upload anh. App chuan hoa input, du doan base price bang XGBoost, chay YOLO de ve bbox, chay ConvNeXt de lay severity, roi tinh damage deduction va final adjusted price.

### 1 phut - Kho khan

Kho khan lon nhat la du lieu khong dong bo va khong co nhan gia sau damage. Vi vay em khong the train mot model end-to-end, ma phai thiet ke hybrid pipeline. Ngoai ra du lieu bang co missing/outlier/don vi text, con du lieu anh co class imbalance, damage nho va label noise.

### 1 phut - Ket luan

Ket qua cuoi cung la em xay dung duoc mot MVP hoan chinh: co pipeline train tabular, co mo hinh detection, co mo hinh severity, co rule adjustment va co app demo. Huong phat trien tiep theo la thu thap du lieu multimodal dong bo hon, phan loai severity theo crop bbox va hoc truc tiep ti le tru gia khi co nhan thuc te.

---

## 16. Cac File Nen Mo Khi Bao Cao

Neu can demo code, nen mo theo thu tu:

1. `app.py`
   - Chi ham `chay_pipeline`.
   - Noi: day la noi ket noi 3 nhanh mo hinh.

2. `src/main.py`
   - Chi split train/validation va flow train tabular.
   - Noi: day la pipeline train XGBoost.

3. `src/feature_engineering.py`
   - Chi cac feature `Tuoi_xe`, `Hang_xe`, `Km_moi_nam`, `Chay_nhieu`.

4. `dich_vu/phat_hien_hu_hong.py`
   - Chi `phat_hien_hu_hong` va thong so YOLO.

5. `dich_vu/muc_do_hu_hong.py`
   - Chi ConvNeXt-Tiny va 3 class severity.

6. `dich_vu/dinh_gia.py`
   - Chi `TRONG_SO_LOP`, `MUC_GIAM_TOI_DA`, `tinh_dieu_chinh_gia`.

7. `Quan_sat/model_comparison_report.txt`
   - Chi bang XGBoost vs Random Forest.

8. `Quan_sat/yolo_car_report/results.png` hoac `results.csv`
   - Chi ket qua train YOLO.

9. `report/training_report.md`
   - Chi bang CNN va ket qua chon ConvNeXt-Tiny.

---

## 17. Cau Ket

Em co the ket thuc bang cau nay:

> Tong ket lai, du an cua em da xay dung duoc mot pipeline hybrid cho bai toan dinh gia xe cu co xet den hu hong ngoai that. Phan gia xe duoc hoc bang XGBoost, phan hinh anh duoc xu ly bang YOLOv8s va ConvNeXt-Tiny, sau do ket hop bang mot tang dieu chinh gia minh bach. Du an van con han che ve du lieu multimodal va nhan gia sau hu hong, nhung da dat duoc muc tieu xay dung mot he thong demo hoan chinh, co ket qua dinh luong va co kha nang mo rong tiep.

