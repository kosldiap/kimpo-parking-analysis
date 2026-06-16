"""항공정보포털에서 김포공항 일별 항공통계를 수집한다.

출처: https://www.airportal.go.kr/stats/transport/dailyRoute.do
엔드포인트: POST /stats/transport/selectDailyStatisticsByRoute.do
  body(JSON): {startDt, endDt, koreaAirport:"RKSS"(김포), airlineType:"D"|"I"}
  응답 content[]: 노선×항공사별 dep/arr/total 의 FP(운항편)·PERSON(여객)·WEIGHT(화물톤)

범위로 주면 합산되어 날짜가 사라지므로 '하루씩(startDt=endDt)' 호출해 일별로 모은다.
국내선(D)/국제선(I)을 각각 받아 일자별 합계를 만든다.

산출물: data/external/flights_daily.csv
  [date, dom_flights, dom_pax, dom_cargo, intl_flights, intl_pax, intl_cargo]

실행:
  python src/fetch_flights.py                 # 2016-01-01~2018-12-31
  python src/fetch_flights.py 20180101 20180107   # 검증용 구간
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

URL = "https://www.airportal.go.kr/stats/transport/selectDailyStatisticsByRoute.do"
AIRPORT = "RKSS"  # 김포
HEADERS = {
    "User-Agent": "Mozilla/5.0 (research; parking-demand-study)",
    "Content-Type": "application/json",
    "Accept": "application/json",
    "X-Requested-With": "XMLHttpRequest",
}
EXTERNAL_DIR = config.PROCESSED_DIR.parent / "external"


def _num(x) -> float:
    if x in (None, "", "-"):
        return 0.0
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return 0.0


def fetch_day(yyyymmdd: str, airline_type: str, retries: int = 3):
    body = json.dumps({
        "startDt": yyyymmdd, "endDt": yyyymmdd,
        "koreaAirport": AIRPORT, "airlineType": airline_type,
    }).encode("utf-8")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(URL, data=body, headers=HEADERS)
            raw = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
            return json.loads(raw).get("content", []) or []
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                print(f"    [ERR] {yyyymmdd} {airline_type}: {e}")
                return None
            time.sleep(1.0 + attempt)


def _sum_day(content) -> dict:
    flights = sum(_num(r.get("total_FP")) for r in content)
    pax = sum(_num(r.get("total_PERSON")) for r in content)
    cargo = sum(_num(r.get("total_WEIGHT")) for r in content)
    return {"flights": flights, "pax": pax, "cargo": cargo}


def collect(start: str, end: str, delay: float = 0.25) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    rows = []
    t0 = time.time()
    for i, d in enumerate(dates):
        ymd = d.strftime("%Y%m%d")
        dom = fetch_day(ymd, "D")
        time.sleep(delay)
        intl = fetch_day(ymd, "I")
        time.sleep(delay)
        rec = {"date": d.date()}
        ds = _sum_day(dom) if dom is not None else {"flights": None, "pax": None, "cargo": None}
        is_ = _sum_day(intl) if intl is not None else {"flights": None, "pax": None, "cargo": None}
        rec.update({"dom_flights": ds["flights"], "dom_pax": ds["pax"], "dom_cargo": ds["cargo"],
                    "intl_flights": is_["flights"], "intl_pax": is_["pax"], "intl_cargo": is_["cargo"]})
        rows.append(rec)
        if (i + 1) % 30 == 0 or i == len(dates) - 1:
            el = time.time() - t0
            print(f"  {i+1}/{len(dates)}  {ymd}  ({el:.0f}s)  "
                  f"국내선 운항{rec['dom_flights']:.0f}/여객{rec['dom_pax']:.0f}  "
                  f"국제선 운항{rec['intl_flights']:.0f}", flush=True)
    return pd.DataFrame(rows)


def main():
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) >= 3:
        start, end = sys.argv[1], sys.argv[2]
        out = EXTERNAL_DIR / f"flights_daily_{start}_{end}.csv"
    else:
        start, end = "2016-01-01", "2018-12-31"
        out = EXTERNAL_DIR / "flights_daily.csv"
    print(f"== 김포 일별 항공통계 수집 {start} ~ {end}")
    df = collect(start, end)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out}  ({len(df)}일)")
    print(df.describe().round(0).to_string())


if __name__ == "__main__":
    main()
