"""KAC 시간대별 항공통계(김포) 수집 → 월×시간대 운항/여객/화물 프로파일.

출처: https://www.airport.co.kr/www/cms/frFlightStatsCon/hourStats.do
엔드포인트: POST /www/ajaxf/frFlightStatsSvc/hourStatsList.do
  params: ST_YY,ST_MM,EN_YY,EN_MM, AIRPORT_CHK=GMP(김포), LINE_TYPE(0국내/1국제), MENU_ID=1250
  응답 data[]: {T_IO(I도착/O출발), T_TYPE(A운항편/B여객명/C화물kg), T_T1..T_T24(0~23시)}

월 단위로 받으면 그 달의 hour-of-day 합계 → 일평균(÷일수)으로 변환.
시간대별 운항/여객/화물 = 구조형 모델의 intra-day 외생 동인.

산출물: data/external/flights_hourly_profile.csv
  [ym, line, hour, flights, pax, cargo_ton]  (line=국내선/국제선, 일평균값)

실행: python src/fetch_flights_hourly.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

URL = "https://www.airport.co.kr/www/ajaxf/frFlightStatsSvc/hourStatsList.do"
EXTERNAL_DIR = config.PROCESSED_DIR.parent / "external"
LINES = {"0": "국내선", "1": "국제선"}
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
           "X-Requested-With": "XMLHttpRequest"}


def fetch_month(ym: str, line: str, retries: int = 3):
    p = {"ST_YY": ym[:4], "ST_MM": ym[4:6], "EN_YY": ym[:4], "EN_MM": ym[4:6],
         "AIRPORT_CHK": "GMP", "LINE_TYPE": line, "USE_TYPE": "", "PASS_TYPE": "",
         "CAGO_TYPE": "", "RAGUL_TYPE": "", "AL_TYPE": "", "CTR_DIV_CD": "", "MENU_ID": "1250"}
    body = urllib.parse.urlencode(p).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request(URL, data=body, headers=HEADERS)
            return json.loads(urllib.request.urlopen(req, timeout=40).read().decode("utf-8", "ignore"))["data"]
        except Exception as e:  # noqa: BLE001
            if a == retries - 1:
                print(f"    [ERR] {ym} line{line}: {e}")
                return None
            time.sleep(1.0 + a)


def _agg(rows, t_type, io):
    """T_TYPE(A운항/B여객/C화물) × T_IO(I도착/O출발) → 24시간 배열."""
    out = [0] * 24
    for r in rows:
        if r.get("T_TYPE") == t_type and r.get("T_IO") == io:
            for h in range(1, 25):
                out[h - 1] += int(r.get(f"T_T{h}", 0) or 0)
    return out


def collect(delay: float = 0.3) -> pd.DataFrame:
    months = pd.period_range("2016-01", "2018-12", freq="M")
    recs = []
    t0 = time.time()
    for p in months:
        ym = f"{p.year}{p.month:02d}"
        nd = p.days_in_month
        for line, nm in LINES.items():
            data = fetch_month(ym, line)
            if not data:
                continue
            fa, fd = _agg(data, "A", "I"), _agg(data, "A", "O")  # 운항 도착/출발
            pa, pd_ = _agg(data, "B", "I"), _agg(data, "B", "O")  # 여객 도착/출발
            ca, cd = _agg(data, "C", "I"), _agg(data, "C", "O")  # 화물 도착/출발
            for h in range(24):
                recs.append({"ym": ym, "line": nm, "hour": h,
                             "flights_arr": fa[h] / nd, "flights_dep": fd[h] / nd,
                             "pax_arr": pa[h] / nd, "pax_dep": pd_[h] / nd,
                             "cargo_arr_ton": ca[h] / 1000 / nd, "cargo_dep_ton": cd[h] / 1000 / nd})
            time.sleep(delay)
        print(f"  {ym} 완료 ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(recs)


def main():
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    print("== KAC 김포 시간대별 항공통계 수집 2016-01~2018-12 ...")
    df = collect()
    out = EXTERNAL_DIR / "flights_hourly_profile.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out}  ({len(df):,}행)")
    # 검증용: 국내선 시간대 출발/도착 여객 프로파일
    piv = df[df["line"] == "국내선"].groupby("hour")[["pax_dep", "pax_arr"]].mean().round(0)
    print("\n국내선 시간대별 일평균 여객 (출발/도착):")
    print(piv.to_string())


if __name__ == "__main__":
    main()
