"""주차 입차가 항공 출발을 몇 시간 선행하는지 데이터로 추정.

방법: 유형별 '시간대별 주차 입차 프로파일'과 '시간대별 출발여객 프로파일'을
      시차 k(0~6h) 만큼 어긋나게 상관 → corr(parking[h], dep[h+k]) 최대인 k = 선행시간.

대상: 국내선·국제선 (여객 주차). 화물/직원은 여객 선행 개념이 약해 제외.
산출: output/11_lead_xcorr.png + 유형별 최적 선행시간 출력
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
OUT = Path(__file__).resolve().parent.parent / "output"
EXTERNAL = config.PROCESSED_DIR.parent / "external"
LINE = {"국내선": "국내선", "국제선": "국제선"}


def parking_profile():
    h = pd.read_parquet(config.PROCESSED_DIR / "hourly.parquet")  # entries=요금>0
    h["category"] = h["lot"].map(config.LOT_TO_CATEGORY)
    h["hr"] = pd.to_datetime(h["hour"]).dt.hour
    return h.groupby(["category", "hr"])["entries"].mean().unstack(0)


def dep_profile():
    f = pd.read_csv(EXTERNAL / "flights_hourly_profile.csv")
    g = f.groupby(["line", "hour"])["pax_dep"].mean().unstack(0)
    return g


def xcorr(park: pd.Series, dep: pd.Series, max_lag=6):
    p = (park - park.mean()) / park.std()
    d = (dep - dep.mean()) / dep.std()
    res = {}
    for k in range(0, max_lag + 1):
        # parking[h] vs dep[h+k]  (dep를 k시간 앞으로 당겨 정렬)
        shifted = np.roll(d.values, -k)
        res[k] = float(np.corrcoef(p.values, shifted)[0, 1])
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pk = parking_profile(); dp = dep_profile()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    best = {}
    for i, cat in enumerate(["국내선", "국제선"]):
        park = pk[cat]; dep = dp[LINE[cat]]
        cc = xcorr(park, dep)
        k_best = max(cc, key=cc.get); best[cat] = (k_best, cc[k_best])
        # xcorr curve
        ax = axes[i, 0]
        ax.bar(list(cc.keys()), list(cc.values()), color="#3182ce")
        ax.axvline(k_best, color="#e53e3e", ls="--")
        ax.set_title(f"{cat}: 시차별 상관 (최적 선행 {k_best}h, r={cc[k_best]:.2f})")
        ax.set_xlabel("선행시간(h)"); ax.set_ylabel("상관계수"); ax.grid(alpha=0.3)
        # overlay profiles (dep shifted by best lead)
        ax2 = axes[i, 1]
        ax2.plot(range(24), park / park.max(), marker="o", ms=3, label="주차 입차", color="#2b6cb0")
        ax2.plot(range(24), dep / dep.max(), marker="s", ms=3, label="출발여객", color="#dd6b20", alpha=0.6)
        ax2.plot(range(24), np.roll(dep.values, -k_best) / dep.max(), marker="^", ms=3,
                 label=f"출발여객(-{k_best}h)", color="#e53e3e")
        ax2.set_title(f"{cat}: 프로파일 정렬 (정규화)")
        ax2.set_xlabel("시각(시)"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "11_lead_xcorr.png"); plt.close(fig)

    print("== 데이터 기반 주차→출발 선행시간 ==")
    for cat, (k, r) in best.items():
        print(f"  {cat}: 최적 선행 {k}시간 (상관 r={r:.3f})")
    return best


if __name__ == "__main__":
    main()
