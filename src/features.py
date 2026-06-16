"""특성공학. 시간당 입차대수(4유형)에 대한 모델 입력 행렬을 만든다.

파이프라인:
  1) hourly.parquet → 주차장을 4유형(국내선/국제선/화물/직원)으로 합산
  2) 유형별 완전한 시간 인덱스 구성
       - 관측 없는 시간 = 0 (실제 무입차)
       - 알려진 누락/운영전 구간 = NaN (0이 아님!)
  3) 달력 특성: 시/요일/주말/월/분기/계절/공휴일/연휴/전후일
  4) 순환 인코딩: 시·요일·월 sin/cos
  5) 시계열 특성: lag(1,24,168h), rolling mean/std(24,168h)
  6) (선택) 외부 join: 기상 / 항공스케줄  ← data/external/ 에 파일 있으면 결합

산출물:
  data/processed/features.parquet   (long: category, datetime, target, 특성들)

실행:  python src/features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

WINDOW_START = pd.Timestamp("2016-01-01 00:00")
WINDOW_END = pd.Timestamp("2018-12-31 23:00")

# NaN(0이 아닌 결측) 으로 둘 구간: (category, 시작, 끝(미만))
KNOWN_NAN = [
    ("화물", "2017-09-01", "2017-10-01"),   # 9월 파일이 10월 중복 → 9월 실누락
]
# 유형별 운영 시작 (이전은 NaN)
CATEGORY_OP_START = {
    "직원": pd.Timestamp("2016-07-20"),   # 항공지원센터
    # 국제선버스(2016-02-06)는 국제선에 흡수되어 국제선은 상시 운영
}

EXTERNAL_DIR = config.PROCESSED_DIR.parent / "external"


# ---------- 공휴일 ----------
def _holiday_set() -> set:
    import holidays
    kr = holidays.SouthKorea(years=[2016, 2017, 2018])
    days = set(pd.Timestamp(d) for d in kr)
    # 패키지 누락 가능한 임시공휴일(선거·연휴연결) 보강
    extra = ["2016-04-13", "2017-05-09", "2017-10-02", "2018-06-13"]
    days |= {pd.Timestamp(d) for d in extra}
    return days


# ---------- 유형별 완전 시간 시계열 ----------
def build_category_hourly() -> pd.DataFrame:
    hourly = pd.read_parquet(config.PROCESSED_DIR / "hourly.parquet")
    hourly["category"] = hourly["lot"].map(config.LOT_TO_CATEGORY)
    cat = (
        hourly.groupby(["category", "hour"])["entries"].sum().reset_index()
    )
    full_idx = pd.date_range(WINDOW_START, WINDOW_END, freq="h")
    frames = []
    for c in config.CATEGORIES:
        s = (
            cat[cat["category"] == c]
            .set_index("hour")["entries"]
            .reindex(full_idx, fill_value=0)
            .astype(float)
        )
        # 운영 시작 전 → NaN
        op = CATEGORY_OP_START.get(c)
        if op is not None:
            s[s.index < op] = np.nan
        # 알려진 누락 구간 → NaN
        for cc, a, b in KNOWN_NAN:
            if cc == c:
                s[(s.index >= pd.Timestamp(a)) & (s.index < pd.Timestamp(b))] = np.nan
        frames.append(pd.DataFrame({"category": c, "datetime": full_idx, "target": s.values}))
    return pd.concat(frames, ignore_index=True)


# ---------- 달력 + 순환 + 시계열 특성 ----------
def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    hol = _holiday_set()
    dt = df["datetime"]
    d = df["datetime"].dt.normalize()
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["dow"] = dt.dt.dayofweek           # 0=월
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["quarter"] = dt.dt.quarter
    df["doy"] = dt.dt.dayofyear
    df["season"] = (df["month"] % 12 // 3)  # 0겨울 1봄 2여름 3가을
    df["is_holiday"] = d.isin(hol).astype(int)
    df["is_pre_holiday"] = (d + pd.Timedelta(days=1)).isin(hol).astype(int)
    df["is_post_holiday"] = (d - pd.Timedelta(days=1)).isin(hol).astype(int)
    df["is_dayoff"] = ((df["is_weekend"] == 1) | (df["is_holiday"] == 1)).astype(int)
    # 순환 인코딩
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    return df


def add_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for c, g in df.groupby("category", sort=False):
        g = g.sort_values("datetime").copy()
        t = g["target"]
        for lag in (1, 24, 168):
            g[f"lag_{lag}"] = t.shift(lag)
        g["roll_mean_24"] = t.shift(1).rolling(24, min_periods=12).mean()
        g["roll_mean_168"] = t.shift(1).rolling(168, min_periods=84).mean()
        g["roll_std_24"] = t.shift(1).rolling(24, min_periods=12).std()
        out.append(g)
    return pd.concat(out, ignore_index=True)


# ---------- 외부 데이터(선택) ----------
def merge_flights(df: pd.DataFrame) -> pd.DataFrame:
    """일별 항공통계(flights_daily.csv)를 유형별로 결합.

    파일: [date, dom_flights, dom_pax, dom_cargo, intl_flights, intl_pax, intl_cargo]
    유형별 매핑(그날 김포 항공활동 → 해당 주차 수요 동인):
      국내선: 국내선 운항/여객/화물
      국제선: 국제선 운항/여객/화물
      화물  : 전체 운항, 전체 화물톤
      직원  : 전체 운항/여객/화물 (공항 전체 활동의 약한 대리변수)
    """
    fpath = EXTERNAL_DIR / "flights_daily.csv"
    if not fpath.exists():
        print(f"  [skip] 항공 파일 없음 → {fpath}")
        return df
    fd = pd.read_csv(fpath, parse_dates=["date"])
    fd["date"] = fd["date"].dt.normalize()
    # 운항 0인 날 = 포털 데이터 공백 → 결측 처리
    fd.loc[fd["dom_flights"] == 0, ["dom_flights", "dom_pax", "dom_cargo"]] = np.nan
    fd.loc[fd["intl_flights"] == 0, ["intl_flights", "intl_pax", "intl_cargo"]] = np.nan
    tot_f = fd["dom_flights"] + fd["intl_flights"]
    tot_p = fd["dom_pax"] + fd["intl_pax"]
    tot_c = fd["dom_cargo"] + fd["intl_cargo"]
    per_cat = {
        "국내선": pd.DataFrame({"date": fd["date"], "f_flights": fd["dom_flights"], "f_pax": fd["dom_pax"], "f_cargo": fd["dom_cargo"]}),
        "국제선": pd.DataFrame({"date": fd["date"], "f_flights": fd["intl_flights"], "f_pax": fd["intl_pax"], "f_cargo": fd["intl_cargo"]}),
        "화물": pd.DataFrame({"date": fd["date"], "f_flights": tot_f, "f_pax": tot_p, "f_cargo": tot_c}),
        "직원": pd.DataFrame({"date": fd["date"], "f_flights": tot_f, "f_pax": tot_p, "f_cargo": tot_c}),
    }
    flong = pd.concat([d.assign(category=c) for c, d in per_cat.items()], ignore_index=True)
    df["date"] = df["datetime"].dt.normalize()
    df = df.merge(flong, on=["date", "category"], how="left").drop(columns="date")
    print(f"  항공 결합: {fpath.name} ({len(fd):,}일) → f_flights/f_pax/f_cargo")
    return df


def merge_external(df: pd.DataFrame) -> pd.DataFrame:
    """data/external/ 에 파일이 있으면 결합. 없으면 건너뜀.

    기상:  weather_hourly.csv  [datetime, temp, precip, snow, wind, humidity]
    항공:  flights_daily.csv   [date, dom_*, intl_*]  (fetch_flights.py 생성)
    """
    wpath = EXTERNAL_DIR / "weather_hourly.csv"
    if wpath.exists():
        w = pd.read_csv(wpath, parse_dates=["datetime"])
        df = df.merge(w, on="datetime", how="left")
        print(f"  기상 결합: {wpath.name} ({len(w):,}행)")
    else:
        print(f"  [skip] 기상 파일 없음 → {wpath}")
    df = merge_flights(df)
    return df


def main():
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("== 유형별 시간당 시계열 구성 ...")
    df = build_category_hourly()
    print("== 달력/순환 특성 ...")
    df = add_calendar(df)
    print("== 시계열(lag/rolling) 특성 ...")
    df = add_timeseries(df)
    print("== 외부 데이터 결합 시도 ...")
    df = merge_external(df)

    out = config.PROCESSED_DIR / "features.parquet"
    df.to_parquet(out, index=False)

    print(f"\n특성행렬: {df.shape}  → {out}")
    print("컬럼:", list(df.columns))
    print("\n== 유형별 시간당 target 요약 (NaN=결측구간) ==")
    g = df.groupby("category")["target"]
    summ = pd.DataFrame({
        "관측시간수": g.count(),
        "결측시간수": g.apply(lambda s: s.isna().sum()),
        "평균": g.mean().round(1),
        "최대": g.max(),
    }).reindex(config.CATEGORIES)
    print(summ.to_string())


if __name__ == "__main__":
    main()
