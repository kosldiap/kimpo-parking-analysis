"""기상청 ASOS 시간자료(서울 108)를 data.go.kr OpenAPI로 수집한다.

서비스: 기상청_지상(종관, ASOS) 시간자료 조회서비스
엔드포인트: /1360000/AsosHourlyInfoService/getWthrDataList
인증키: data/external/datagokr_key.txt (Decoding 키) 또는 env WEATHER_API_KEY (git 제외)

산출물: data/external/weather_hourly.csv
  [datetime, temp, precip, snow, wind, humidity, vis]

실행:
  python src/fetch_weather.py                 # 2016-01 ~ 2018-12
  python src/fetch_weather.py 201601 201601   # 검증용(한 달)
"""
from __future__ import annotations

import os
import sys
import time
import urllib.parse
import urllib.request
import json
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

ENDPOINT = "https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
STN = "108"  # 서울 (김포 대용; 종관ASOS 미설치)
EXTERNAL_DIR = config.PROCESSED_DIR.parent / "external"
KEY_FILE = EXTERNAL_DIR / "datagokr_key.txt"


def _load_key() -> str:
    k = os.environ.get("WEATHER_API_KEY")
    if k:
        return k.strip()
    if KEY_FILE.exists():
        return KEY_FILE.read_text(encoding="utf-8").strip()
    raise RuntimeError(f"인증키 없음: env WEATHER_API_KEY 또는 {KEY_FILE}")


def _num(x):
    s = str(x).strip()
    if s in ("", "-", "None"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def fetch_month(key: str, ym: str, retries: int = 3):
    """ym='YYYYMM' 한 달치 시간자료 반환(리스트). 실패 시 None."""
    y, m = int(ym[:4]), int(ym[4:6])
    start = f"{ym}01"
    end_day = pd.Period(f"{y}-{m:02d}").days_in_month
    end = f"{ym}{end_day:02d}"
    params = {
        "serviceKey": key, "pageNo": "1", "numOfRows": "999",
        "dataType": "JSON", "dataCd": "ASOS", "dateCd": "HR",
        "startDt": start, "startHh": "00", "endDt": end, "endHh": "23",
        "stnIds": STN,
    }
    url = ENDPOINT + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            raw = urllib.request.urlopen(url, timeout=40).read().decode("utf-8", "ignore")
            data = json.loads(raw)
            body = data["response"]["body"]
            items = body["items"]
            if items in ("", None):
                return []
            it = items["item"]
            return it if isinstance(it, list) else [it]
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                head = raw[:200] if "raw" in dir() else ""
                print(f"    [ERR] {ym}: {e}  {head}")
                return None
            time.sleep(1.5 + attempt)


PARTS_DIR = EXTERNAL_DIR / "_weather_parts"
OUT_FILE = EXTERNAL_DIR / "weather_hourly.csv"


def _items_to_df(items) -> pd.DataFrame:
    rows = [{
        "datetime": r.get("tm"),
        "temp": _num(r.get("ta")),
        "precip": _num(r.get("rn")) or 0.0,
        "snow": _num(r.get("dsnw")) or 0.0,
        "wind": _num(r.get("ws")),
        "humidity": _num(r.get("hm")),
        "vis": _num(r.get("vs")),
    } for r in items]
    return pd.DataFrame(rows)


def collect(start_ym: str, end_ym: str, delay: float = 0.4, resume: bool = True):
    """월별로 받아 _weather_parts/ 에 저장(이어받기). 전 월 완료 시 True."""
    key = _load_key()
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    months = list(pd.period_range(start_ym, end_ym, freq="M"))
    t0 = time.time()
    for p in months:
        ym = f"{p.year}{p.month:02d}"
        part = PARTS_DIR / f"weather_{ym}.csv"
        if resume and part.exists():
            continue
        items = fetch_month(key, ym)
        if items is None:
            # 403(키 미활성) 또는 일시 오류 → 중단(다음 실행에서 이 달부터 재개)
            print(f"  {ym} 실패(키 미활성 가능) → 중단, 다음 시도에 재개", flush=True)
            return False
        _items_to_df(items).to_csv(part, index=False, encoding="utf-8-sig")
        print(f"  {ym}: {len(items)}행 저장  ({time.time()-t0:.0f}s)", flush=True)
        time.sleep(delay)
    return True


def assemble():
    parts = sorted(PARTS_DIR.glob("weather_*.csv"))
    if not parts:
        return None
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    return df


def main():
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    auto = "--auto" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    s, e = (args[0], args[1]) if len(args) >= 2 else ("201601", "201812")

    n_months = len(list(pd.period_range(s, e, freq="M")))
    done_parts = len(list(PARTS_DIR.glob("weather_*.csv"))) if PARTS_DIR.exists() else 0
    if auto and OUT_FILE.exists() and done_parts >= n_months:
        print(f"이미 완료됨: {OUT_FILE} (parts {done_parts}/{n_months}) → 종료")
        return

    print(f"== 서울(108) ASOS 시간자료 수집 {s}~{e}  (parts {done_parts}/{n_months})")
    ok = collect(s, e)
    if ok:
        df = assemble()
        print(f"\n✅ 완료: {OUT_FILE}  ({len(df):,}행)")
        print(f"기간: {df['datetime'].min()} ~ {df['datetime'].max()}")
        print(df[["temp", "precip", "snow", "wind", "humidity"]].describe().round(1).to_string())
    else:
        print(f"⏳ 미완료(parts {len(list(PARTS_DIR.glob('weather_*.csv')))}/{n_months}). "
              f"키 활성화 후 자동 재개됩니다.")


if __name__ == "__main__":
    # python src/fetch_weather.py            # 1회 수집
    # python src/fetch_weather.py --auto      # 자동/이어받기 (스케줄러용)
    main()
