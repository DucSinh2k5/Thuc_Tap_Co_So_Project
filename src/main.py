import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from EDA_after import eda_after
from EDA_before import eda_before
from feature_engineering import (
    gioi_han_hang_xe,
    gioi_han_xe,
    tao_moi_feature,
    xu_ly_gia_tri_thieu,
    xu_ly_outlier,
)
from load_data_and_cleaning import (
    chuyen_cot_sang_category,
    chuyen_cot_sang_so,
    doi_ten_cot,
    load_data,
    loai_bo_hang_ban,
)
from preprocessing import tien_xu_ly
from train_and_evaluate import (
    compare_models,
    evaluate_model,
    feature_importance_report,
    run_baseline,
    save,
    train_model,
)


PROJECT_DIR = r"F:\Documents\CODE\TTCS\Thuc_Tap_Co_So_Project"
REPORT_DIR = os.path.join(PROJECT_DIR, "Quan_sat")

RUN_BASELINE = True
RUN_FEATURE_IMPORTANCE = True
USE_XGB_EARLY_STOPPING = True
EARLY_STOPPING_ROUNDS = 50


def clean_pipeline(df):
    df = doi_ten_cot(df)
    df = loai_bo_hang_ban(df)
    df = chuyen_cot_sang_so(df)
    df = chuyen_cot_sang_category(df)
    return df


def export_cleaned_dataset(df_source, km_median, imputers, top_names, top_brands, bounds, output_path):
    df_export = df_source.copy()
    df_export, _ = tao_moi_feature(df_export, km_median=km_median)
    df_export, _ = xu_ly_gia_tri_thieu(df_export, imputers=imputers)
    df_export, _ = gioi_han_xe(df_export, top_names=top_names)
    df_export, _ = gioi_han_hang_xe(df_export, top_brands=top_brands)
    df_export, _ = xu_ly_outlier(df_export, bounds=bounds)
    df_export.to_csv(output_path, index=False)
    print(f"Da luu du lieu da cleaning vao {output_path}")


def main():
    print("\n" + "=" * 60)
    print("DU DOAN GIA XE O TO CU - MACHINE LEARNING")
    print("=" * 60 + "\n")
    os.makedirs(REPORT_DIR, exist_ok=True)

    df_train = load_data(os.path.join(PROJECT_DIR, "Datasets", "train-dataset.csv"))
    df_test = load_data(os.path.join(PROJECT_DIR, "Datasets", "test-dataset.csv"))

    eda_before(df_train)
    df_train = clean_pipeline(df_train)
    df_test = clean_pipeline(df_test)

    df_tr, df_val = train_test_split(df_train, test_size=0.2, random_state=42)
    df_tr = df_tr.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    print(f"Train={len(df_tr)}, Validation={len(df_val)}, Test={len(df_test)}\n")

    # Feature engineering and cleaning are fitted only on train, then reused for val/test.
    df_tr, km_median = tao_moi_feature(df_tr)
    df_val, _ = tao_moi_feature(df_val, km_median=km_median)
    df_test, _ = tao_moi_feature(df_test, km_median=km_median)

    df_tr, imputers = xu_ly_gia_tri_thieu(df_tr)
    df_val, _ = xu_ly_gia_tri_thieu(df_val, imputers=imputers)
    df_test, _ = xu_ly_gia_tri_thieu(df_test, imputers=imputers)

    df_tr, top_names = gioi_han_xe(df_tr)
    df_val, _ = gioi_han_xe(df_val, top_names=top_names)
    df_test, _ = gioi_han_xe(df_test, top_names=top_names)

    df_tr, top_brands = gioi_han_hang_xe(df_tr, top_n=15)
    df_val, _ = gioi_han_hang_xe(df_val, top_brands=top_brands)
    df_test, _ = gioi_han_hang_xe(df_test, top_brands=top_brands)

    df_tr, bounds = xu_ly_outlier(df_tr)
    df_val, _ = xu_ly_outlier(df_val, bounds=bounds)
    df_test, _ = xu_ly_outlier(df_test, bounds=bounds)

    train_cleaned_path = os.path.join(PROJECT_DIR, "Datasets", "train_cleaned.csv")
    df_tr.to_csv(train_cleaned_path, index=False)
    print(f"Da luu ban train da cleaning: {train_cleaned_path}")

    export_cleaned_dataset(
        df_train,
        km_median=km_median,
        imputers=imputers,
        top_names=top_names,
        top_brands=top_brands,
        bounds=bounds,
        output_path=os.path.join(PROJECT_DIR, "Datasets", "test.csv"),
    )

    eda_after(df_tr)

    target_col = "Gia_theo_lakh"
    excluded_cols = [
        "Gia_theo_lakh",
        "Gia_moi_lakh",
        "Unnamed: 0",
        "Nam_san_xuat",
        "Quang_duong_da_di(km)",
        "Ten_xe",
    ]
    candidate_features = [c for c in df_tr.columns if c not in excluded_cols]
    if not candidate_features:
        print("Khong tim thay cot feature trong train")
        return

    if df_tr[target_col].isna().any() or df_val[target_col].isna().any():
        before_tr, before_val = len(df_tr), len(df_val)
        df_tr = df_tr[df_tr[target_col].notna()].reset_index(drop=True)
        df_val = df_val[df_val[target_col].notna()].reset_index(drop=True)
        print(f"Da loai bo dong thieu target: train {before_tr}->{len(df_tr)}, val {before_val}->{len(df_val)}")

    numeric_features = [c for c in candidate_features if pd.api.types.is_numeric_dtype(df_tr[c])]
    categorical_features = [c for c in candidate_features if c not in numeric_features]

    print(f"\nNumeric features     ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}\n")

    preprocessor = tien_xu_ly(numeric_features, categorical_features)
    X_tr = df_tr[candidate_features]
    y_tr = df_tr[target_col].values
    X_val = df_val[candidate_features]
    y_val = df_val[target_col].values

    if RUN_BASELINE:
        baseline_path = os.path.join(REPORT_DIR, "baseline_report.txt")
        run_baseline(y_tr, y_val, output_path=baseline_path)

    for col in candidate_features:
        if col not in df_test.columns:
            df_test[col] = np.nan
    X_test = df_test[candidate_features]

    X_tr_trans = preprocessor.fit_transform(X_tr)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)

    # OrdinalEncoder keeps one column per categorical feature.
    feature_names = numeric_features + categorical_features
    selected_features = feature_names
    selected_indices = list(range(len(feature_names)))
    print(f"Using all features ({len(selected_features)}): {selected_features}")

    comparison_path = os.path.join(REPORT_DIR, "model_comparison_report.txt")
    _, best_model_name = compare_models(
        X_tr_trans,
        y_tr,
        X_val_trans,
        y_val,
        selected_indices,
        output_path=comparison_path,
    )
    if best_model_name != "XGBoost":
        print(f"Luu y: {best_model_name} dang co R2 validation cao hon trong lan chay nay.")
    print("Model duoc chon de trien khai: XGBoost")

    eval_set = None
    if USE_XGB_EARLY_STOPPING:
        eval_set = [(X_val_trans[:, selected_indices], y_val)]

    xgb = train_model(
        X_tr_trans,
        y_tr,
        selected_indices,
        eval_set=eval_set,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )

    evaluate_model(xgb, X_val_trans, y_val, selected_indices)

    if RUN_FEATURE_IMPORTANCE:
        feature_importance_report(
            xgb,
            X_val_trans[:, selected_indices],
            y_val,
            selected_features,
            output_dir=REPORT_DIR,
        )

    X_sel = X_test_trans[:, selected_indices]
    df_test["Du_doan_XGB"] = xgb.predict(X_sel)
    preview_cols = [c for c in ["Ten_xe", "Du_doan_XGB"] if c in df_test.columns]
    print("\nDu doan tren tap test (10 dong dau):")
    print(df_test[preview_cols].head(10).to_string(index=False))

    save(df_tr, preprocessor, xgb, candidate_features, selected_features, selected_indices)

    print("\nHOAN THANH!\n")


if __name__ == "__main__":
    main()
