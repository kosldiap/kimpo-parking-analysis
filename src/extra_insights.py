"""추가 인사이트 5종 (현재 2016~18 데이터, 교란 없음).

1) 점유율 부하지속곡선 + 설계기준시간(연 30번째 피크)
2) 회전율(1면당 일 회전수)
3) 입·출차 동시 혼잡(게이트 부하) — 시간대 프로파일
4) 매출·감면 구조 (요금 vs 수입)
5) 명절·연휴 수요 배율

산출: output/17~21_*.png + 수치 출력
주의: 2016은 DB누락으로 수준 과소 → 2017~18 위주 해석.
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
from features import _holiday_set, LUNAR_DATES  # noqa: E402

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
OUT = Path(__file__).resolve().parent.parent / "output"
P = config.PROCESSED_DIR
CATS = config.CATEGORIES
W0, W1 = pd.Timestamp("2016-01-01"), pd.Timestamp("2019-01-01")


def _tx():
    tx = pd.read_parquet(P / "transactions.parquet",
                         columns=["lot", "입차일시", "출차일시", "요금", "수입"])
    tx = tx[pd.to_numeric(tx["요금"], errors="coerce").fillna(0) > 0].copy()
    tx["category"] = tx["lot"].map(config.LOT_TO_CATEGORY)
    tx["요금"] = pd.to_numeric(tx["요금"], errors="coerce")
    tx["수입"] = pd.to_numeric(tx["수입"], errors="coerce")
    return tx.dropna(subset=["입차일시"])


# ---------- 1. 부하지속곡선 + 설계기준시간 ----------
def load_duration(occ):
    fig, ax = plt.subplots(figsize=(11, 6))
    print("== 1. 부하지속곡선 / 설계기준 ==")
    for c in CATS:
        s = occ[occ["category"] == c]["occupancy"].dropna().values
        cap = config.CAPACITY[c]
        pct = np.sort(s)[::-1] / cap * 100
        x = np.arange(1, len(pct) + 1) / len(pct) * 100
        ax.plot(x, pct, label=c)
        years = len(pct) / 8760
        over100 = (pct >= 100).sum() / max(years, 1)
        over95 = (pct >= 95).sum() / max(years, 1)
        rank30 = pct[min(int(30 * years), len(pct) - 1)]  # 연 30번째 피크
        print(f"  {c}: 만차(≥100%) 연 {over100:.0f}h, ≥95% 연 {over95:.0f}h, "
              f"설계기준(연30위) 점유 {rank30:.0f}%")
    ax.axhline(100, color="k", ls="--", lw=1, label="정원(100%)")
    ax.set_xlabel("시간 비율 (%, 내림차순)"); ax.set_ylabel("점유율 %")
    ax.set_title("점유율 부하지속곡선 (capacity=100%)"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_xlim(0, 30)
    fig.tight_layout(); fig.savefig(OUT / "17_load_duration.png"); plt.close(fig)


# ---------- 2. 회전율 ----------
def turnover(tx):
    print("\n== 2. 회전율 (1면당 일 회전수) ==")
    d = tx.copy(); d["date"] = d["입차일시"].dt.normalize()
    daily = d.groupby(["category", "date"]).size().groupby("category").mean()
    for c in CATS:
        t = daily[c] / config.CAPACITY[c]
        print(f"  {c}: 일평균 입차 {daily[c]:.0f} / 정원 {config.CAPACITY[c]} = 회전 {t:.2f}회/면/일")


# ---------- 3. 입·출차 동시 혼잡 ----------
def gate_load(tx):
    print("\n== 3. 게이트 부하 (입+출차 동시 피크 시각) ==")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, c in zip(axes.ravel(), CATS):
        d = tx[tx["category"] == c]
        ein = d["입차일시"].dt.hour.value_counts().reindex(range(24), fill_value=0)
        eout = d["출차일시"].dropna().dt.hour.value_counts().reindex(range(24), fill_value=0)
        ax.plot(range(24), ein, label="입차"); ax.plot(range(24), eout, label="출차")
        ax.plot(range(24), ein + eout, label="입+출 합", lw=2, color="k")
        ax.set_title(f"{c} (입피크 {ein.idxmax()}시 / 합피크 {(ein+eout).idxmax()}시)")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
        print(f"  {c}: 입차피크 {ein.idxmax()}시, 출차피크 {eout.idxmax()}시, 입+출 합피크 {(ein+eout).idxmax()}시")
    fig.suptitle("입·출차 시간대 프로파일 (게이트 부하)", y=1.0)
    fig.tight_layout(); fig.savefig(OUT / "18_gate_load.png"); plt.close(fig)


# ---------- 4. 매출·감면 ----------
def revenue(tx):
    print("\n== 4. 매출·감면 구조 ==")
    g = tx.groupby("category").agg(요금합=("요금", "sum"), 수입합=("수입", "sum"))
    g["감면율%"] = (1 - g["수입합"] / g["요금합"]) * 100
    g["건당수입"] = g["수입합"] / tx.groupby("category").size()
    print(g.assign(요금합=lambda x: (x["요금합"]/1e8).round(1),
                   수입합=lambda x: (x["수입합"]/1e8).round(1)).reindex(CATS)
          .rename(columns={"요금합": "요금(억)", "수입합": "수입(억)"})
          .round(1).to_string())
    # 시간대별 수입 프로파일
    fig, ax = plt.subplots(figsize=(11, 5))
    tx2 = tx.copy(); tx2["h"] = tx2["입차일시"].dt.hour
    for c in CATS:
        prof = tx2[tx2["category"] == c].groupby("h")["수입"].sum()
        ax.plot(prof.index, prof.values / 1e6, label=c)
    ax.set_xlabel("입차 시각"); ax.set_ylabel("수입 합(백만원)")
    ax.set_title("시간대별 수입"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "19_revenue.png"); plt.close(fig)


# ---------- 5. 명절·연휴 배율 ----------
def holiday_mult(tx):
    print("\n== 5. 명절·연휴 수요 배율 (평일=1.0, 2017~18 기준) ==")
    hol = _holiday_set(); lunar = {pd.Timestamp(x) for x in LUNAR_DATES}
    d = tx.copy(); d["date"] = d["입차일시"].dt.normalize()
    daily = d.groupby(["category", "date"]).size().reset_index(name="entries")
    daily = daily[daily["date"].dt.year >= 2017]  # 2016 누락 제외
    dd = daily["date"]
    daily["dow"] = dd.dt.dayofweek
    daily["type"] = "평일"
    daily.loc[daily["dow"] >= 5, "type"] = "주말"
    daily.loc[dd.isin(hol), "type"] = "공휴일"
    daily.loc[dd.apply(lambda x: any(abs((x - l).days) <= 2 for l in lunar)), "type"] = "명절연휴"
    piv = daily.pivot_table(index="category", columns="type", values="entries", aggfunc="mean")
    base = piv["평일"]
    mult = piv.div(base, axis=0)[["평일", "주말", "공휴일", "명절연휴"]].reindex(CATS)
    print(mult.round(2).to_string())
    fig, ax = plt.subplots(figsize=(9, 5))
    mult.T.plot(kind="bar", ax=ax)
    ax.axhline(1, color="k", lw=0.8); ax.set_ylabel("평일 대비 배율")
    ax.set_title("일유형별 입차수요 배율 (평일=1)"); ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(OUT / "20_holiday_mult.png"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    occ = pd.read_parquet(P / "occupancy.parquet")
    tx = _tx()
    load_duration(occ)
    turnover(tx)
    gate_load(tx)
    revenue(tx)
    holiday_mult(tx)
    print(f"\n산출물 → {OUT} (17~20)")


if __name__ == "__main__":
    main()
