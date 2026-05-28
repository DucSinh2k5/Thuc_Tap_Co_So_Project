import inspect
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

RF_COMPARE_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 2,
    "min_samples_split": 5,
    "max_features": 0.5,
    "max_samples": 0.7,
    "random_state": 42,
    "n_jobs": -1,
}


XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}


def _chon_feature(X, selected_indices):
    return X[:, selected_indices]


def _format_metrics(label, metrics):
    rmse, mae, r2 = metrics
    return f"{label:<18} RMSE={rmse:.4f} | MAE={mae:.4f} | R2={r2:.4f}"


def tinh_metrics(y_true, y_pred, n_features):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def run_baseline(y_train, y_val, output_path=None):
   
    mean_pred = np.full_like(y_val, np.mean(y_train), dtype=float)
    median_pred = np.full_like(y_val, np.median(y_train), dtype=float)

    mean_metrics = tinh_metrics(y_val, mean_pred, 0)
    median_metrics = tinh_metrics(y_val, median_pred, 0)

    lines = [
        "BASELINE REPORT",
        "-" * 60,
        _format_metrics("Mean baseline", mean_metrics),
        _format_metrics("Median baseline", median_metrics),
    ]

    report = "\n".join(lines)
    print(report)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report + "\n")
    return {
        "mean": mean_metrics,
        "median": median_metrics,
    }


def compare_models(X_train_clean, y_train, X_val_clean, y_val, selected_indices, output_path=None):
    """
    Train Random Forest va XGBoost tren cung tap feature, sau do chon model tot hon
    dua tren R2 validation. Linear Regression da duoc loai bo de pipeline gon hon.
    """
    X_train_sel = _chon_feature(X_train_clean, selected_indices)
    X_val_sel = _chon_feature(X_val_clean, selected_indices)
    n_features = X_train_sel.shape[1]

    print("Dang train Random Forest de so sanh...")
    rf = RandomForestRegressor(**RF_COMPARE_PARAMS)
    rf.fit(X_train_sel, y_train)
    rf_metrics = tinh_metrics(y_val, rf.predict(X_val_sel), n_features)

    print("Dang train XGBoost de so sanh...")
    xgb = XGBRegressor(**XGB_PARAMS)
    xgb.fit(X_train_sel, y_train)
    xgb_metrics = tinh_metrics(y_val, xgb.predict(X_val_sel), n_features)

    results = {
        "Random Forest": rf_metrics,
        "XGBoost": xgb_metrics,
    }
    best_model = max(results, key=lambda name: results[name][2])

    lines = [
        "=" * 80,
        "SO SANH MO HINH TABULAR",
        "=" * 80,
        "Baseline chi dung de lam moc tham chieu. Hai mo hinh ML duoc so sanh truc tiep la RF va XGBoost.",
        "",
        f"{'Model':<18} {'RMSE':>12} {'MAE':>12} {'R2':>12}",
        "-" * 80,
    ]
    for name, metrics in results.items():
        rmse, mae, r2 = metrics
        lines.append(f"{name:<18} {rmse:>12.4f} {mae:>12.4f} {r2:>12.4f}")
    lines.extend([
        "-" * 80,
        f"Model tot nhat theo R2 validation: {best_model}",
        "=" * 80,
    ])

    report = "\n".join(lines)
    print(report)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report + "\n")

    return results, best_model


def train_model(X_train_clean, y_train, selected_indices, eval_set=None, early_stopping_rounds=50):
    """
    Train model trien khai cuoi cung. Sau buoc so sanh, project chon XGBoost nen
    model.pkl chi luu XGBoost va cac thanh phan tien xu ly can thiet.
    """
    X_train_sel = _chon_feature(X_train_clean, selected_indices)
    params = XGB_PARAMS.copy()
    if eval_set is not None:
        params["n_estimators"] = 2000

    print("Dang train XGBoost final...")
    xgb = XGBRegressor(**params)
    if eval_set is None:
        xgb.fit(X_train_sel, y_train)
    else:
        fit_sig = inspect.signature(xgb.fit)
        if "early_stopping_rounds" in fit_sig.parameters:
            xgb.fit(
                X_train_sel,
                y_train,
                eval_set=eval_set,
                early_stopping_rounds=early_stopping_rounds,
                verbose=False,
            )
        else:
            print("XGBoost version hien tai khong ho tro early_stopping_rounds trong fit().")
            xgb.fit(
                X_train_sel,
                y_train,
                eval_set=eval_set,
                verbose=False,
            )
    print("Da train xong XGBoost final!")
    return xgb


def evaluate_model(xgb, X_val_clean, y_val, selected_indices):
    X_val_sel = _chon_feature(X_val_clean, selected_indices)
    metrics = tinh_metrics(y_val, xgb.predict(X_val_sel), X_val_sel.shape[1])
    print("=" * 80)
    print("DANH GIA XGBOOST FINAL TREN VALIDATION")
    print("=" * 80)
    print(_format_metrics("XGBoost final", metrics))
    print("=" * 80)
    return metrics


def feature_importance_report(
    xgb,
    X_val,
    y_val,
    feature_names,
    output_dir,
    n_repeats=10,
    random_state=42,
):
    os.makedirs(output_dir, exist_ok=True)
    notes = []

    if hasattr(xgb, "feature_importances_"):
        df_xgb = pd.DataFrame({
            "feature": feature_names,
            "importance": xgb.feature_importances_,
        }).sort_values("importance", ascending=False)
        path_xgb = os.path.join(output_dir, "feature_importance_xgb.csv")
        df_xgb.to_csv(path_xgb, index=False)
        notes.append(f"XGB impurity importance -> {path_xgb}")

    perm = permutation_importance(
        xgb,
        X_val,
        y_val,
        scoring="r2",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )
    df_perm = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)
    path_perm = os.path.join(output_dir, "permutation_importance_xgb.csv")
    df_perm.to_csv(path_perm, index=False)
    notes.append(f"Permutation importance (xgb) -> {path_perm}")

    print("Feature importance reports:")
    for line in notes:
        print(f"- {line}")
    return notes


def save(df, preprocessor, xgb, candidate_features, selected_features, selected_indices):
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    quan_sat_dir = os.path.join(project_dir, "Quan_sat")
    models_dir = os.path.join(project_dir, "Models")
    os.makedirs(quan_sat_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    test_path = os.path.join(quan_sat_dir, "test.csv")
    selected_features_path = os.path.join(quan_sat_dir, "selected_features.txt")
    model_path = os.path.join(models_dir, "model.pkl")

    df.to_csv(test_path, index=False)
    print(f"Da luu du lieu vao {test_path}")

    with open(selected_features_path, "w", encoding="utf-8") as f:
        for feat in selected_features:
            f.write(feat + "\n")
    print(f"Da luu danh sach feature: {selected_features_path}")

    model = {
        "preprocessor": preprocessor,
        "model_name": "XGBoost",
        "model_xgb": xgb,
        "candidate_features": candidate_features,
        "selected_indices": selected_indices,
        "selected_features": selected_features,
    }
    joblib.dump(model, model_path)
    print(f"Da luu model: {model_path}")
