"""2차 모델링: 시계열 모형 (Prophet). 시간당 입차대수, 유형별.

Prophet: 추세 + 다중계절성(일/주/연) + 한국공휴일 + 외부회귀변수(항공/기상).
특히 직원 유형의 성장추세를 명시적 trend로 외삽 → ML 트리의 약점 보완.

분할: train=~2017, test=2018 (ML과 동일).
산출: output/metrics_ts.csv, predictions_ts.parquet, 비교 그림.

실행:  python src/model_ts.py
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from evaluate import metrics_row, metrics_table  # noqa: E402

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

OUT = Path(__file__).resolve().parent.parent / "output"
SPLIT = pd.Timestamp("2018-01-01")

# 유형별 외부 회귀변수 (있을 때만 사용)
REGRESSORS = {
    "국내선": ["f_pax", "temp", "precip"],
    "국제선": ["f_pax", "temp", "precip"],
    "화물": ["f_cargo", "precip"],
    "직원": ["temp"],
}


def _holidays_df():
    from features import _holiday_set
    days = sorted(_holiday_set())
    return pd.DataFrame({
        "holiday": "kr_holiday",
        "ds": pd.to_datetime(list(days)),
        "lower_window": 0, "upper_window": 1,
    })


def run():
    from prophet import Prophet
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(config.PROCESSED_DIR / "features.parquet")
    hol = _holidays_df()
    rows, preds = [], []

    for cat in config.CATEGORIES:
        regs = [r for r in REGRESSORS.get(cat, []) if r in df.columns]
        cols = ["datetime", "target"] + regs
        d = df[df["category"] == cat][cols].dropna(subset=["target"]).copy()
        # 회귀변수 결측 보간 (Prophet은 NaN 불가)
        for r in regs:
            d[r] = d[r].ffill().bfill()
            d[r] = d[r].fillna(d[r].mean())
        d = d.rename(columns={"datetime": "ds", "target": "y"})
        train = d[d["ds"] < SPLIT]
        test = d[d["ds"] >= SPLIT]

        m = Prophet(growth="linear", yearly_seasonality=True,
                    weekly_seasonality=True, daily_seasonality=True,
                    holidays=hol, changepoint_prior_scale=0.1)
        for r in regs:
            m.add_regressor(r)
        m.fit(train)

        fc = m.predict(test[["ds"] + regs])
        yp = np.clip(fc["yhat"].values, 0, None)
        rows.append(metrics_row(cat, "Prophet", test["y"].values, yp))
        preds.append(pd.DataFrame({"category": cat, "datetime": test["ds"].values,
                                   "actual": test["y"].values, "Prophet": yp}))
        print(f"[{cat}] regs={regs}  MAE={rows[-1]['MAE']:.2f}  "
              f"WAPE={rows[-1]['WAPE']:.1f}%  R2={rows[-1]['R2']:.3f}", flush=True)

    table = metrics_table(rows)
    table.to_csv(OUT / "metrics_ts.csv", index=False, encoding="utf-8-sig")
    pred_df = pd.concat(preds, ignore_index=True)
    pred_df.to_parquet(config.PROCESSED_DIR / "predictions_ts.parquet", index=False)

    print("\n===== Prophet 성능 (2018 test) =====")
    print(table.to_string(index=False))
    print(f"\n산출물 → {OUT}")


if __name__ == "__main__":
    run()
