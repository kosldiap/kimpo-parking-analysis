"""주차 점유율(동시주차대수) 재구성 — 입·출차 트랜잭션에서 직접 계산.

원리: 시점 t의 점유 = (t까지 누적 입차) - (t까지 누적 출차).
  입차시각에 +1, 출차시각에 -1 이벤트를 시간단위로 집계 후 누적합.

산출물: data/processed/occupancy.parquet  [category, datetime, occupancy]
  + output/10_occupancy.png (유형별 점유 추이 + 일중 프로파일)

주의(경계효과):
  - 2016-01-01 시작 시 기존 주차차량을 0으로 보아 초기 며칠 과소추정(장기주차 채워질 때까지).
  - 데이터 끝(2018-12-31)에 미출차 차량은 -1 이벤트가 없어 말미 과대 가능.
  - 화물 2017-09(누락), 직원 운영전 구간은 점유도 결측/왜곡.

실행:  python src/occupancy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
OUT = Path(__file__).resolve().parent.parent / "output"

WIN_START = pd.Timestamp("2016-01-01 00:00")
WIN_END = pd.Timestamp("2018-12-31 23:00")


def build_occupancy() -> pd.DataFrame:
    tx = pd.read_parquet(config.PROCESSED_DIR / "transactions.parquet",
                         columns=["lot", "입차일시", "출차일시"])
    tx["category"] = tx["lot"].map(config.LOT_TO_CATEGORY)
    full_idx = pd.date_range(WIN_START, WIN_END, freq="h")
    frames = []
    for cat in config.CATEGORIES:
        sub = tx[tx["category"] == cat]
        ent = sub["입차일시"].dropna().dt.floor("h").value_counts()
        ex = sub["출차일시"].dropna().dt.floor("h").value_counts()
        delta = ent.reindex(full_idx, fill_value=0) - ex.reindex(full_idx, fill_value=0)
        occ = delta.cumsum().clip(lower=0)
        frames.append(pd.DataFrame({"category": cat, "datetime": full_idx,
                                    "occupancy": occ.values}))
    return pd.concat(frames, ignore_index=True)


def practical_capacity(occ: pd.DataFrame) -> pd.Series:
    """실측 최대점유의 상위분위로 실효 수용능력 추정 (공식 정원 미상이라 근사)."""
    return occ.groupby("category")["occupancy"].quantile(0.995).round()


def plot(occ: pd.DataFrame, cap: pd.Series):
    lots = config.CATEGORIES
    fig, axes = plt.subplots(len(lots), 1, figsize=(13, 10), sharex=True)
    for ax, c in zip(axes, lots):
        s = occ[occ["category"] == c].set_index("datetime")["occupancy"]
        ax.plot(s.index, s.values, lw=0.5, color="#2b6cb0")
        ax.axhline(cap[c], color="#e53e3e", ls="--", lw=1, label=f"실효정원~{cap[c]:.0f}")
        ax.set_ylabel(c, rotation=0, ha="right", va="center")
        ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.25)
    axes[0].set_title("유형별 동시주차대수(점유) 추이  (빨강=실효정원 근사, 상위0.5%)")
    fig.tight_layout(); fig.savefig(OUT / "10_occupancy.png"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    occ = build_occupancy()
    occ.to_parquet(config.PROCESSED_DIR / "occupancy.parquet", index=False)
    cap = practical_capacity(occ)
    plot(occ, cap)
    print("점유율 재구성 완료:", occ.shape)
    print("\n== 유형별 점유 통계 (대) ==")
    g = occ.groupby("category")["occupancy"]
    summ = pd.DataFrame({"평균": g.mean().round(0), "중앙": g.median().round(0),
                         "최대": g.max().round(0), "실효정원≈": cap})
    print(summ.reindex(config.CATEGORIES).to_string())
    # 만차 근접(>=95% 실효정원) 시간 비율
    print("\n== 혼잡(실효정원 95%↑) 시간 비율 ==")
    for c in config.CATEGORIES:
        s = occ[occ["category"] == c]["occupancy"]
        ratio = (s >= 0.95 * cap[c]).mean() * 100
        print(f"  {c}: {ratio:.1f}%")
    print(f"\n산출물 → {config.PROCESSED_DIR/'occupancy.parquet'}, {OUT/'10_occupancy.png'}")


if __name__ == "__main__":
    main()
