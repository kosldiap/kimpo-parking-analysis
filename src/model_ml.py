"""1차 모델링: ML 모델 (Linear 베이스라인 / RandomForest / XGBoost).

- target: 시간당 입차대수, 유형별 개별 모델
- 분할: train = ~2017, test = 2018 (시계열 순서 보존)
- 결측(target NaN)·lag 워밍업 행 제외
- 평가: MAE/RMSE/MAPE/WAPE/R2  → output/metrics_ml.csv
- 산출: 예측 parquet, 예측vs실제 그림, XGB 특성중요도 그림

실행:  python src/model_ml.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from evaluate import metrics_row, metrics_table  # noqa: E402

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

OUT = Path(__file__).resolve().parent.parent / "output"
SPLIT = pd.Timestamp("2018-01-01")

# 학습에 쓸 특성 (year 제외: 트리가 미래연도 외삽 못함)
FEATURES = [
    "month", "day", "hour", "dow", "is_weekend", "quarter", "doy", "season",
    "is_holiday", "is_pre_holiday", "is_post_holiday", "is_dayoff",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "lag_1", "lag_24", "lag_168", "roll_mean_24", "roll_mean_168", "roll_std_24",
]

# 외부 데이터가 결합돼 있으면 자동 포함
OPTIONAL_FEATURES = ["f_flights", "f_pax", "f_cargo",
                     "temp", "precip", "snow", "wind", "humidity", "vis"]


LAG_FEATURES = ["lag_1", "lag_24", "lag_168", "roll_mean_24", "roll_mean_168", "roll_std_24"]


def active_features(df, use_lags: bool = True) -> list:
    base = FEATURES if use_lags else [f for f in FEATURES if f not in LAG_FEATURES]
    return base + [c for c in OPTIONAL_FEATURES if c in df.columns]


def _models():
    return {
        "Linear": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=18, min_samples_leaf=3,
            n_jobs=-1, random_state=42,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=600, max_depth=7, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42,
        ),
    }


def _prep(df_cat: pd.DataFrame, feats: list):
    d = df_cat.dropna(subset=["target"] + feats).copy()
    train = d[d["datetime"] < SPLIT]
    test = d[d["datetime"] >= SPLIT]
    return train, test


def run(use_lags: bool = True):
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(config.PROCESSED_DIR / "features.parquet")
    feats = active_features(df, use_lags=use_lags)
    tag = "" if use_lags else "_structural"
    print(f"[{'lag포함(운영)' if use_lags else 'lag제외(구조적/시나리오용)'}] 특성 {len(feats)}개")
    rows, preds, importances = [], [], {}

    for cat in config.CATEGORIES:
        train, test = _prep(df[df["category"] == cat], feats)
        Xtr, ytr = train[feats], train["target"]
        Xte, yte = test[feats], test["target"]
        print(f"\n[{cat}] train {len(train):,} / test {len(test):,}")
        pr = pd.DataFrame({"category": cat, "datetime": test["datetime"].values, "actual": yte.values})
        for name, model in _models().items():
            model.fit(Xtr, ytr)
            yp = np.clip(model.predict(Xte), 0, None)  # 음수 예측 0으로
            rows.append(metrics_row(cat, name, yte, yp))
            pr[name] = yp
            print(f"  {name:13} MAE={rows[-1]['MAE']:.2f}  WAPE={rows[-1]['WAPE']:.1f}%  R2={rows[-1]['R2']:.3f}")
            if name == "XGBoost":
                importances[cat] = pd.Series(model.feature_importances_, index=feats)
        preds.append(pr)

    table = metrics_table(rows)
    table.to_csv(OUT / f"metrics_ml{tag}.csv", index=False, encoding="utf-8-sig")
    pred_df = pd.concat(preds, ignore_index=True)
    pred_df.to_parquet(config.PROCESSED_DIR / f"predictions_ml{tag}.parquet", index=False)

    if use_lags:
        _plot_pred(pred_df)
        _plot_importance(importances)

    print(f"\n===== ML 성능 (2018 test, {'lag포함' if use_lags else '구조적'}) =====")
    print(table.to_string(index=False))
    print(f"\n산출물 → {OUT}")


def _plot_pred(pred_df, days=14):
    fig, axes = plt.subplots(len(config.CATEGORIES), 1, figsize=(13, 11), sharex=False)
    for ax, cat in zip(axes, config.CATEGORIES):
        p = pred_df[pred_df["category"] == cat].sort_values("datetime")
        p = p[p["datetime"] < SPLIT + pd.Timedelta(days=days)]
        ax.plot(p["datetime"], p["actual"], color="k", lw=1.3, label="실제")
        ax.plot(p["datetime"], p["XGBoost"], color="#e53e3e", lw=1.0, label="XGBoost")
        ax.plot(p["datetime"], p["Linear"], color="#3182ce", lw=0.8, alpha=0.7, label="Linear")
        ax.set_ylabel(cat, rotation=0, ha="right", va="center")
        ax.grid(alpha=0.25)
    axes[0].set_title(f"예측 vs 실제 (2018-01, 첫 {days}일)")
    axes[0].legend(ncol=3, fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "08_pred_vs_actual_ml.png"); plt.close(fig)


def _plot_importance(importances):
    fig, axes = plt.subplots(1, len(importances), figsize=(16, 6))
    for ax, (cat, imp) in zip(np.atleast_1d(axes), importances.items()):
        top = imp.sort_values().tail(12)
        ax.barh(top.index, top.values, color="#38a169")
        ax.set_title(f"{cat} (XGB 중요도)", fontsize=10)
        ax.tick_params(labelsize=8)
    fig.tight_layout(); fig.savefig(OUT / "09_feature_importance_ml.png"); plt.close(fig)


if __name__ == "__main__":
    # python src/model_ml.py             # lag포함(운영 단기예측)
    # python src/model_ml.py structural  # lag제외(시나리오용 구조적 예측)
    run(use_lags="structural" not in sys.argv)
