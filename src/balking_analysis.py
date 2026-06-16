"""가설: 점유율(만차 근접)이 높을수록 요금0원 비주차(자리 못찾아 회차=balking) 증가?

방법:
  - 시간단위 패널: (유형, 시각) 별 요금0 비율(dropoff_share), 점유율%(점유/실제정원)
  - 점유율은 요금>0 차량으로 산출 → dropoff와 기계적 상관 없음(독립)
  - 운영시간(6~23시)만. 단순상관 + 시간대 통제 회귀(OLS: share ~ occ% + C(hour))
  - 만차에 닿는 유형은 국내선뿐(최대 122%) → 핵심 검증 대상

산출: output/14_balking_vs_occupancy.png + 상관·회귀계수 출력
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
OUT = Path(__file__).resolve().parent.parent / "output"
P = config.PROCESSED_DIR


def build_panel() -> pd.DataFrame:
    tx = pd.read_parquet(P / "transactions.parquet", columns=["lot", "입차일시", "요금"])
    tx["category"] = tx["lot"].map(config.LOT_TO_CATEGORY)
    tx = tx.dropna(subset=["입차일시"])
    tx["hour"] = tx["입차일시"].dt.floor("h")
    tx["drop"] = (pd.to_numeric(tx["요금"], errors="coerce").fillna(0) <= 0).astype(int)
    g = tx.groupby(["category", "hour"]).agg(total=("drop", "size"),
                                             drops=("drop", "sum")).reset_index()
    g["dropoff_share"] = g["drops"] / g["total"] * 100

    occ = pd.read_parquet(P / "occupancy.parquet")
    occ = occ.rename(columns={"datetime": "hour"})
    g = g.merge(occ, on=["category", "hour"], how="left")
    g["occ_pct"] = g.apply(lambda r: r["occupancy"] / config.CAPACITY[r["category"]] * 100, axis=1)
    g["hod"] = g["hour"].dt.hour
    g = g[(g["hod"] >= 6) & (g["hod"] <= 23) & (g["total"] >= 5)]  # 운영시간·소표본 제외
    return g


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g = build_panel()

    print("== 점유율% vs 요금0 비율 (유형별) ==")
    print(f"{'유형':8}{'단순상관':>9}{'시간대통제 회귀계수':>18}{'p값':>10}{'관측':>8}")
    for c in config.CATEGORIES:
        d = g[g["category"] == c].dropna(subset=["occ_pct", "dropoff_share"])
        if len(d) < 100:
            continue
        r = d["occ_pct"].corr(d["dropoff_share"])
        m = smf.ols("dropoff_share ~ occ_pct + C(hod)", data=d).fit()
        coef, p = m.params["occ_pct"], m.pvalues["occ_pct"]
        print(f"{c:8}{r:9.3f}{coef:18.4f}{p:10.2e}{len(d):8,}")

    # 그래프: 점유율 구간별 평균 요금0 비율
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, c in zip(axes.ravel(), config.CATEGORIES):
        d = g[g["category"] == c].dropna(subset=["occ_pct", "dropoff_share"])
        bins = pd.cut(d["occ_pct"], bins=np.arange(0, 131, 10))
        prof = d.groupby(bins, observed=True)["dropoff_share"].mean()
        centers = [b.mid for b in prof.index]
        ax.plot(centers, prof.values, marker="o", color="#e53e3e")
        ax.axvline(100, color="gray", ls=":", lw=1)
        ax.set_title(f"{c} (정원 {config.CAPACITY[c]}대)")
        ax.set_xlabel("점유율 %"); ax.set_ylabel("요금0 비율 %"); ax.grid(alpha=0.3)
    fig.suptitle("점유율 구간별 요금0원 비주차 비율 (점선=만차100%)", y=1.0)
    fig.tight_layout(); fig.savefig(OUT / "14_balking_vs_occupancy.png"); plt.close(fig)
    print(f"\n산출물 → {OUT/'14_balking_vs_occupancy.png'}")


if __name__ == "__main__":
    main()
