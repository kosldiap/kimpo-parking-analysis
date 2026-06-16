"""자율주행/카셰어링 시나리오 — '점유'를 체류시간대별로 분해해 제거가능 수요 추정.

핵심 원리:
  점유시간(occupancy-hours) = Σ 각 차량의 체류시간. 즉 5일 주차 1대 = 120 점유시간
  = 1시간 주차 120대와 동일한 '공간 점유 부담'. → 점유는 장기 여행주차가 좌우.

카셰어링/AV는 '드롭오프 후 차량이 떠나는' 모델 → **장기 여행주차 점유가 제거됨**
(짧은 드롭오프는 이미 태워주고 떠나므로 불변). 따라서:
  제거가능 점유비중 = 장기체류(≥임계)가 점유에서 차지하는 비중  ← 데이터
  주차수요 감소 ≈ 대체율(시나리오) × 제거가능 점유비중      ← 가정×데이터

산출: output/15_av_scenario.png + 체류분해/시나리오 표 출력
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
OUT = Path(__file__).resolve().parent.parent / "output"

# 체류시간 버킷 (시간)
EDGES = [0, 1, 4, 12, 24, 48, 96, 1e9]
LABELS = ["~1h", "1~4h", "4~12h", "12~24h", "1~2일", "2~4일", "4일+"]
ELIMINABLE_FROM = "12~24h"  # 12h↑(숙박급 여행주차)부터 카셰어링 제거대상으로 간주
SUBST = [0.2, 0.4, 0.6, 0.8]  # 카셰어링 대체율 시나리오


def load():
    tx = pd.read_parquet(config.PROCESSED_DIR / "transactions.parquet",
                         columns=["lot", "입차일시", "출차일시", "주차시간_분", "요금"])
    tx = tx[pd.to_numeric(tx["요금"], errors="coerce").fillna(0) > 0].copy()
    tx["category"] = tx["lot"].map(config.LOT_TO_CATEGORY)
    dur = pd.to_numeric(tx["주차시간_분"], errors="coerce")
    # 결측/이상은 출차-입차로 보정
    alt = (tx["출차일시"] - tx["입차일시"]).dt.total_seconds() / 60
    tx["dur_h"] = dur.fillna(alt).clip(lower=0) / 60
    return tx.dropna(subset=["dur_h"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tx = load()
    tx["bucket"] = pd.cut(tx["dur_h"], bins=EDGES, labels=LABELS, right=False)

    # 점유시간(=Σ체류) 기준 분해 vs 대수(건수) 기준 분해
    occ = tx.groupby(["category", "bucket"], observed=True)["dur_h"].sum().unstack(0)
    cnt = tx.groupby(["category", "bucket"], observed=True).size().unstack(0)
    occ_share = (occ / occ.sum() * 100).reindex(LABELS)
    cnt_share = (cnt / cnt.sum() * 100).reindex(LABELS)

    print("== 점유시간 비중 (%) — 장기주차가 점유를 좌우 ==")
    print(occ_share[config.CATEGORIES].round(1).to_string())
    print("\n== 대수(건수) 비중 (%) — 참고 ==")
    print(cnt_share[config.CATEGORIES].round(1).to_string())

    # 제거가능 점유비중 (ELIMINABLE_FROM 이상 버킷 합)
    idx_from = LABELS.index(ELIMINABLE_FROM)
    elim = occ_share.iloc[idx_from:][config.CATEGORIES].sum()
    print(f"\n== 제거가능 점유비중 (체류 {ELIMINABLE_FROM}↑ = 숙박급 여행주차) ==")
    print(elim.round(1).to_string())

    # 시나리오: 주차수요(점유) 감소율 = 대체율 × 제거가능비중
    print("\n== 카셰어링/AV 시나리오별 점유수요 감소율 (%) ==")
    sc = pd.DataFrame({f"대체{int(s*100)}%": (s * elim).round(1) for s in SUBST})
    print(sc.to_string())

    # 그래프
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    occ_share[config.CATEGORIES].T.plot(kind="barh", stacked=True, ax=axes[0],
                                        colormap="RdYlBu_r")
    axes[0].set_title("점유시간 구성 (체류시간대별, %) — 장기주차 비중")
    axes[0].set_xlabel("점유시간 비중 %"); axes[0].legend(fontsize=7, ncol=4, loc="lower right")
    for cat in config.CATEGORIES:
        axes[1].plot([int(s * 100) for s in SUBST], [s * elim[cat] for s in SUBST],
                     marker="o", label=cat)
    axes[1].set_title(f"카셰어링 대체율별 점유수요 감소 (체류 {ELIMINABLE_FROM}↑ 기준)")
    axes[1].set_xlabel("카셰어링 대체율 %"); axes[1].set_ylabel("점유수요 감소율 %")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "15_av_scenario.png"); plt.close(fig)
    print(f"\n산출물 → {OUT/'15_av_scenario.png'}")


if __name__ == "__main__":
    main()
