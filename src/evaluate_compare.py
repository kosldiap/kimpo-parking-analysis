"""#6 모델 성능 평가·비교 정식화.

1) 통합 비교표: 운영형(lag) Linear/RF/XGB, 구조형(no-lag) RF/XGB, Prophet
   — 모두 동일 2018 holdout 기준. MAE/RMSE/MAPE/WAPE/R2.
2) 시계열 교차검증(TimeSeriesSplit 5fold): XGB 운영/구조 — 평균±표준편차로 안정성.
3) 비교 그래프: 유형×모델 R²/WAPE, CV 안정성.

산출: output/metrics_all.csv, cv_summary.csv, 12_model_compare.png, 13_cv_stability.png

실행: python src/evaluate_compare.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from evaluate import metrics  # noqa: E402
from model_ml import FEATURES, LAG_FEATURES, OPTIONAL_FEATURES, active_features, _models  # noqa: E402

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
OUT = Path(__file__).resolve().parent.parent / "output"
P = config.PROCESSED_DIR


# ---------- 1) holdout 통합 비교 ----------
def consolidate() -> pd.DataFrame:
    specs = [
        ("predictions_ml.parquet", "운영(lag)", ["Linear", "RandomForest", "XGBoost"]),
        ("predictions_ml_structural.parquet", "구조(no-lag)", ["RandomForest", "XGBoost"]),
        ("predictions_ts.parquet", "구조(no-lag)", ["Prophet"]),
    ]
    rows = []
    for fname, regime, models in specs:
        fp = P / fname
        if not fp.exists():
            continue
        df = pd.read_parquet(fp)
        for cat, g in df.groupby("category"):
            for m in models:
                if m not in g:
                    continue
                mt = metrics(g["actual"], g[m])
                rows.append({"category": cat, "regime": regime, "model": m,
                             **{k: round(v, 3) for k, v in mt.items()}})
    return pd.DataFrame(rows)


# ---------- 2) 시계열 교차검증 ----------
def _xgb():
    return XGBRegressor(n_estimators=600, max_depth=7, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42)


def cv(df_all: pd.DataFrame, n_splits=5) -> pd.DataFrame:
    rows = []
    for use_lags, tag in [(True, "운영(lag)"), (False, "구조(no-lag)")]:
        feats = active_features(df_all, use_lags=use_lags)
        for cat in config.CATEGORIES:
            d = df_all[df_all["category"] == cat].dropna(subset=["target"] + feats)
            d = d.sort_values("datetime")
            X, y = d[feats].values, d["target"].values
            if len(d) < n_splits * 50:
                continue
            tss = TimeSeriesSplit(n_splits=n_splits)
            r2s, wapes = [], []
            for tr, te in tss.split(X):
                m = _xgb().fit(X[tr], y[tr])
                p = np.clip(m.predict(X[te]), 0, None)
                mt = metrics(y[te], p)
                r2s.append(mt["R2"]); wapes.append(mt["WAPE"])
            rows.append({"regime": tag, "category": cat,
                         "R2_mean": round(np.mean(r2s), 3), "R2_std": round(np.std(r2s), 3),
                         "WAPE_mean": round(np.mean(wapes), 1), "WAPE_std": round(np.std(wapes), 1),
                         "folds": n_splits})
    return pd.DataFrame(rows)


# ---------- 3) 그래프 ----------
def plot_compare(tab: pd.DataFrame):
    tab = tab.copy()
    tab["label"] = tab["model"] + "·" + tab["regime"].str.replace("(no-lag)", "", regex=False).str.replace("(lag)", "", regex=False)
    key = tab[tab["model"].isin(["XGBoost", "Prophet"]) | (tab["model"] == "Linear")]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, metric in zip(axes, ["R2", "WAPE"]):
        piv = key.pivot_table(index="category", columns="label", values=metric)
        piv = piv.reindex(config.CATEGORIES)
        piv.plot(kind="bar", ax=ax, width=0.8)
        ax.set_title(f"유형별 {metric} (모델·체제 비교)")
        ax.set_xlabel(""); ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        ax.tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(OUT / "12_model_compare.png"); plt.close(fig)


def plot_cv(cvtab: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    cats = config.CATEGORIES
    x = np.arange(len(cats))
    for i, (tag, sub) in enumerate(cvtab.groupby("regime")):
        s = sub.set_index("category").reindex(cats)
        ax.bar(x + (i - 0.5) * 0.35, s["R2_mean"], 0.35, yerr=s["R2_std"],
               capsize=4, label=tag)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("R² (5-fold 평균±표준편차)")
    ax.set_title("시계열 교차검증 안정성 (XGBoost)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "13_cv_stability.png"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tab = consolidate()
    tab.to_csv(OUT / "metrics_all.csv", index=False, encoding="utf-8-sig")
    print("===== 통합 비교 (2018 holdout) =====")
    show = tab.sort_values(["category", "regime", "model"])
    print(show[["category", "regime", "model", "MAE", "WAPE", "R2"]].to_string(index=False))

    print("\n===== 시계열 교차검증 (XGBoost, 5-fold) =====")
    df_all = pd.read_parquet(P / "features.parquet")
    cvtab = cv(df_all)
    cvtab.to_csv(OUT / "cv_summary.csv", index=False, encoding="utf-8-sig")
    print(cvtab.to_string(index=False))

    plot_compare(tab)
    plot_cv(cvtab)
    print(f"\n산출물 → {OUT}")


if __name__ == "__main__":
    main()
