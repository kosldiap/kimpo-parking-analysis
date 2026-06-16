"""전 raw data를 읽어 트랜잭션 마스터 + 시계열 집계를 생성한다.

산출물 (PROCESSED_DIR):
  transactions.parquet  : 전 거래 정제본 (입차시각 기준, 9999999 제외)
  daily.parquet         : 주차장×일 단위 입차대수/매출/평균체류
  hourly.parquet        : 주차장×시각(시간) 단위 입차대수
  quality_report.csv    : 파일별 행수·기간·결측 등 품질 리포트

실행:  python src/aggregate.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from etl import read_parking_xls  # noqa: E402

KEEP_COLS = [
    "lot", "주차장명", "차량번호", "주차권번호",
    "입차일시", "출차일시", "주차시간_분", "요금", "수입", "지불유형",
    "year", "source_file",
]


def _iter_files():
    for year, d in config.YEAR_DIRS.items():
        if not d.exists():
            continue
        for f in sorted(d.glob("*.xls")):
            yield year, f


def load_all_transactions(verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """전 파일 읽어 트랜잭션 마스터 + 품질 리포트 반환."""
    frames = []
    report = []
    for year, f in _iter_files():
        t = time.time()
        try:
            df = read_parking_xls(f)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  [ERR] {f.name}: {e}")
            report.append({"file": f.name, "year": year, "rows": 0, "error": str(e)})
            continue
        if not df.empty:
            df["year"] = year
            df["lot"] = df["주차장명"].map(config.normalize_lot)
            for c in KEEP_COLS:
                if c not in df:
                    df[c] = pd.NA
            frames.append(df[KEEP_COLS])
        rep = {
            "file": f.name,
            "year": year,
            "rows": len(df),
            "entry_min": df["입차일시"].min() if not df.empty else None,
            "entry_max": df["입차일시"].max() if not df.empty else None,
            "exit_na": int(df["출차일시"].isna().sum()) if "출차일시" in df else None,
            "fee_na": int(df["요금"].isna().sum()) if "요금" in df else None,
            "sec": round(time.time() - t, 1),
            "error": "",
        }
        report.append(rep)
        if verbose:
            print(f"  {f.name:42} {len(df):>8,}행  {rep['sec']:>5}s")

    master = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return master, pd.DataFrame(report)


def build_daily(master: pd.DataFrame) -> pd.DataFrame:
    m = master.dropna(subset=["입차일시"]).copy()
    m["date"] = m["입차일시"].dt.normalize()
    g = m.groupby(["lot", "date"])
    daily = g.agg(
        entries=("입차일시", "size"),
        revenue=("수입", "sum"),
        fee=("요금", "sum"),
        avg_stay_min=("주차시간_분", "mean"),
    ).reset_index()
    return daily


def build_hourly(master: pd.DataFrame) -> pd.DataFrame:
    m = master.dropna(subset=["입차일시"]).copy()
    m["hour"] = m["입차일시"].dt.floor("h")
    hourly = (
        m.groupby(["lot", "hour"]).size().reset_index(name="entries")
    )
    return hourly


def main():
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("== raw data 읽는 중 (전 파일) ...")
    t0 = time.time()
    master, report = load_all_transactions()
    print(f"\n총 {len(master):,} 거래, {time.time() - t0:.0f}s 소요")

    print("== 저장 중 ...")
    master.to_parquet(config.PROCESSED_DIR / "transactions.parquet", index=False)
    daily = build_daily(master)
    hourly = build_hourly(master)
    daily.to_parquet(config.PROCESSED_DIR / "daily.parquet", index=False)
    hourly.to_parquet(config.PROCESSED_DIR / "hourly.parquet", index=False)
    report.to_csv(config.PROCESSED_DIR / "quality_report.csv", index=False, encoding="utf-8-sig")

    print("\n== 주차장별 거래수 ==")
    print(master["lot"].value_counts().to_string())
    print("\n== 연도별 입차 기간 ==")
    for y, sub in master.groupby("year"):
        print(f"  {y}: {sub['입차일시'].min()} ~ {sub['입차일시'].max()}  ({len(sub):,})")
    print(f"\n산출물 → {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
