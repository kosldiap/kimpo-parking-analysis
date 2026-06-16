"""예측 평가 지표 (전 모델 공통)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def metrics(y_true, y_pred) -> dict:
    y = np.asarray(y_true, float)
    p = np.asarray(y_pred, float)
    e = y - p
    mae = np.mean(np.abs(e))
    rmse = np.sqrt(np.mean(e ** 2))
    # WAPE: sum|e|/sum|y| — 0이 많은 시계열에 강건
    wape = np.sum(np.abs(e)) / max(np.sum(np.abs(y)), 1e-9) * 100
    # MAPE: 실측>0 인 지점만 (0 나눗셈 회피)
    mask = y > 0
    mape = np.mean(np.abs(e[mask] / y[mask])) * 100 if mask.any() else np.nan
    ss_res = np.sum(e ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "WAPE": wape, "R2": r2}


def metrics_row(category, model, y_true, y_pred) -> dict:
    m = metrics(y_true, y_pred)
    return {"category": category, "model": model, **{k: round(v, 3) for k, v in m.items()}}


def metrics_table(rows) -> pd.DataFrame:
    return pd.DataFrame(rows)[["category", "model", "MAE", "RMSE", "MAPE", "WAPE", "R2"]]
