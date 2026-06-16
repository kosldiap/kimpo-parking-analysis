"""8번 — 김포 국내선 항공사 구성(LCC vs FSC) 월별 추세 수집·분석.

airportal selectDailyStatisticsByRoute.do 를 월 단위로 호출(범위=합산) →
항공사(al_ICAO)별 운항/여객 집계 → FSC/LCC 분류 → 2016~18 LCC 점유율 추세.

가설: LCC 확대 → 운항 분산. 주차수요 패턴(피크)과의 연관 해석.

산출: data/external/airlines_monthly.csv [ym, group, flights, pax]
      output/16_lcc_trend.png + LCC 점유율 추세 출력
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
OUT = Path(__file__).resolve().parent.parent / "output"
EXTERNAL = config.PROCESSED_DIR.parent / "external"
URL = "https://www.airportal.go.kr/stats/transport/selectDailyStatisticsByRoute.do"

FSC = {"KAL", "AAR"}                       # 대한항공, 아시아나
LCC = {"JJA", "JNA", "TWB", "ABL", "ESR"}  # 제주, 진에어, 티웨이, 에어부산, 이스타


def group_of(al):
    if al in FSC:
        return "FSC"
    if al in LCC:
        return "LCC"
    return "기타"


def fetch_month(ym_s, ym_e, line="D", retries=3):
    body = json.dumps({"startDt": ym_s, "endDt": ym_e, "koreaAirport": "RKSS", "airlineType": line})
    for a in range(retries):
        try:
            req = urllib.request.Request(URL, data=body.encode(), headers={
                "User-Agent": "Mozilla/5.0", "Content-Type": "application/json", "X-Requested-With": "XMLHttpRequest"})
            return json.loads(urllib.request.urlopen(req, timeout=40).read().decode("utf-8", "ignore")).get("content", [])
        except Exception as e:  # noqa: BLE001
            if a == retries - 1:
                print(f"  [ERR] {ym_s}: {e}"); return None
            time.sleep(1 + a)


def collect():
    rows = []
    for p in pd.period_range("2016-01", "2018-12", freq="M"):
        ym = f"{p.year}{p.month:02d}"; last = f"{p.year}{p.month:02d}{p.days_in_month:02d}"
        data = fetch_month(ym + "01", last)
        if not data:
            continue
        agg = defaultdict(lambda: [0, 0])
        for r in data:
            g = group_of(r.get("al_ICAO"))
            agg[g][0] += int(float(r.get("total_FP", 0) or 0))
            agg[g][1] += int(float(r.get("total_PERSON", 0) or 0))
        for g, (fp, pax) in agg.items():
            rows.append({"ym": ym, "group": g, "flights": fp, "pax": pax})
        time.sleep(0.25)
    return pd.DataFrame(rows)


def main():
    EXTERNAL.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    df = collect()
    df.to_csv(EXTERNAL / "airlines_monthly.csv", index=False, encoding="utf-8-sig")
    df["dt"] = pd.to_datetime(df["ym"], format="%Y%m")

    fp = df.pivot_table(index="dt", columns="group", values="flights", aggfunc="sum").fillna(0)
    px = df.pivot_table(index="dt", columns="group", values="pax", aggfunc="sum").fillna(0)
    lcc_fp = fp.get("LCC", 0) / fp.sum(axis=1) * 100
    lcc_px = px.get("LCC", 0) / px.sum(axis=1) * 100

    print("== 김포 국내선 LCC 점유율 (%) 분기 ==")
    q = pd.DataFrame({"운항_LCC%": lcc_fp, "여객_LCC%": lcc_px})
    print(q.resample("Q").mean().round(1).to_string())  # pandas<2.2 호환

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    fp.plot.area(ax=ax[0]); ax[0].set_title("김포 국내선 운항편 구성 (FSC/LCC/기타)"); ax[0].set_ylabel("월 운항편")
    ax[1].plot(lcc_fp.index, lcc_fp, marker="o", label="운항 LCC%")
    ax[1].plot(lcc_px.index, lcc_px, marker="s", label="여객 LCC%")
    ax[1].set_title("LCC 점유율 추세"); ax[1].set_ylabel("%"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "16_lcc_trend.png"); plt.close(fig)
    print(f"\n2016-01 LCC운항%={lcc_fp.iloc[0]:.1f} → 2018-12 ={lcc_fp.iloc[-1]:.1f}")
    print(f"산출물 → {OUT/'16_lcc_trend.png'}")


if __name__ == "__main__":
    main()
