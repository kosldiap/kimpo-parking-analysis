"""탐색적 데이터 분석(EDA). 집계 parquet를 읽어 패턴을 시각화한다.

산출물(output/):
  01_daily_timeseries.png   주차장별 일별 입차대수(2016~2018)
  02_monthly_trend.png      월별 추세 + 공항 전체
  03_hourly_profile.png     시간대(0~23시)별 평균 입차
  04_dow_profile.png        요일별 평균 입차
  05_month_seasonality.png  월(계절)별 평균 입차
  06_stay_duration.png      체류시간 분포
  07_corr_heatmap.png       주차장 간 일별수요 상관

실행:  python src/eda.py
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
LOTS = config.PARKING_LOTS
DOW_KR = ["월", "화", "수", "목", "금", "토", "일"]


def _load():
    P = config.PROCESSED_DIR
    daily = pd.read_parquet(P / "daily.parquet")
    hourly = pd.read_parquet(P / "hourly.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    hourly["hour"] = pd.to_datetime(hourly["hour"])
    return daily, hourly


def _lot_order(present):
    return [l for l in LOTS if l in set(present)]


def plot_daily_timeseries(daily):
    lots = _lot_order(daily["lot"].unique())
    fig, axes = plt.subplots(len(lots), 1, figsize=(13, 11), sharex=True)
    for ax, lot in zip(axes, lots):
        s = daily[daily["lot"] == lot].set_index("date")["entries"].sort_index()
        ax.plot(s.index, s.values, lw=0.6, color="#2b6cb0")
        ax.plot(s.index, s.rolling(7).mean().values, lw=1.4, color="#e53e3e")
        ax.set_ylabel(lot, rotation=0, ha="right", va="center")
        ax.grid(alpha=0.25)
    axes[0].set_title("주차장별 일별 입차대수 (파랑=일별, 빨강=7일 이동평균)")
    fig.tight_layout()
    fig.savefig(OUT / "01_daily_timeseries.png"); plt.close(fig)


def plot_monthly_trend(daily):
    d = daily.copy()
    d["ym"] = d["date"].dt.to_period("M").dt.to_timestamp()
    piv = d.groupby(["ym", "lot"])["entries"].sum().unstack()
    piv = piv[_lot_order(piv.columns)]
    fig, ax = plt.subplots(figsize=(13, 6))
    for lot in piv.columns:
        ax.plot(piv.index, piv[lot], marker="o", ms=3, label=lot)
    ax.plot(piv.index, piv.sum(axis=1), color="k", lw=2.2, ls="--", label="전체")
    ax.set_title("월별 입차대수 추세"); ax.set_ylabel("월 입차대수")
    ax.legend(ncol=4, fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "02_monthly_trend.png"); plt.close(fig)


def plot_hourly_profile(hourly):
    h = hourly.copy()
    h["hr"] = h["hour"].dt.hour
    prof = h.groupby(["lot", "hr"])["entries"].mean().unstack(0)
    prof = prof[_lot_order(prof.columns)]
    fig, ax = plt.subplots(figsize=(12, 6))
    for lot in prof.columns:
        ax.plot(prof.index, prof[lot], marker="o", ms=3, label=lot)
    ax.set_xticks(range(0, 24)); ax.set_xlabel("시각(시)")
    ax.set_ylabel("시간당 평균 입차대수")
    ax.set_title("시간대별 평균 입차 프로파일")
    ax.legend(ncol=3); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "03_hourly_profile.png"); plt.close(fig)


def plot_dow_profile(daily):
    d = daily.copy()
    d["dow"] = d["date"].dt.dayofweek
    prof = d.groupby(["lot", "dow"])["entries"].mean().unstack(0)
    prof = prof[_lot_order(prof.columns)]
    fig, ax = plt.subplots(figsize=(11, 6))
    for lot in prof.columns:
        ax.plot(prof.index, prof[lot], marker="o", label=lot)
    ax.set_xticks(range(7)); ax.set_xticklabels(DOW_KR)
    ax.set_ylabel("일평균 입차대수"); ax.set_title("요일별 평균 입차")
    ax.legend(ncol=3); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "04_dow_profile.png"); plt.close(fig)


def plot_month_seasonality(daily):
    d = daily.copy()
    d["mon"] = d["date"].dt.month
    prof = d.groupby(["lot", "mon"])["entries"].mean().unstack(0)
    prof = prof[_lot_order(prof.columns)]
    fig, ax = plt.subplots(figsize=(11, 6))
    for lot in prof.columns:
        ax.plot(prof.index, prof[lot], marker="o", label=lot)
    ax.set_xticks(range(1, 13)); ax.set_xlabel("월")
    ax.set_ylabel("일평균 입차대수"); ax.set_title("월(계절)별 평균 입차")
    ax.legend(ncol=3); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "05_month_seasonality.png"); plt.close(fig)


def plot_stay_duration():
    P = config.PROCESSED_DIR
    df = pd.read_parquet(P / "transactions.parquet", columns=["lot", "주차시간_분"])
    df = df.dropna()
    df["hours"] = pd.to_numeric(df["주차시간_분"], errors="coerce") / 60
    df = df[(df["hours"] >= 0) & (df["hours"] <= 72)]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(df["hours"], bins=144, color="#3182ce")
    ax.set_xlabel("체류시간(시간, 0~72h)"); ax.set_ylabel("건수")
    ax.set_title(f"체류시간 분포  (중앙값 {df['hours'].median():.1f}h, "
                 f"평균 {df['hours'].mean():.1f}h)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "06_stay_duration.png"); plt.close(fig)
    return df["hours"].describe()


def plot_corr(daily):
    piv = daily.pivot_table(index="date", columns="lot", values="entries")
    piv = piv[_lot_order(piv.columns)]
    corr = piv.corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("주차장 간 일별 입차수요 상관")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(OUT / "07_corr_heatmap.png"); plt.close(fig)
    return corr


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    daily, hourly = _load()
    print("일별 레코드:", daily.shape, "| 기간:", daily["date"].min().date(), "~", daily["date"].max().date())

    plot_daily_timeseries(daily); print("  01 일별 시계열 ✓")
    plot_monthly_trend(daily);    print("  02 월별 추세 ✓")
    plot_hourly_profile(hourly);  print("  03 시간대 프로파일 ✓")
    plot_dow_profile(daily);      print("  04 요일별 ✓")
    plot_month_seasonality(daily);print("  05 월 계절성 ✓")
    stay = plot_stay_duration();  print("  06 체류시간 ✓")
    corr = plot_corr(daily);      print("  07 상관 ✓")

    print("\n== 주차장별 일평균 입차 ==")
    print(daily.groupby("lot")["entries"].mean().reindex(_lot_order(daily['lot'].unique())).round(0).to_string())
    print("\n== 체류시간(시간) 요약 ==")
    print(stay.round(2).to_string())
    print("\n== 상관행렬 ==")
    print(corr.round(2).to_string())
    print(f"\n그림 → {OUT}")


if __name__ == "__main__":
    main()
