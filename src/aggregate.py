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


# 분석 유효 구간 (이 밖은 불완전: 출차기준 분류로 인한 잔여/경계)
WINDOW_START = "2016-01-01"
WINDOW_END = "2019-01-01"  # 미만

# 완전중복 판정 키 (화물청사 2017년9월=10월 중복파일 등 제거)
DUP_KEY = ["lot", "차량번호", "주차권번호", "입차일시", "출차일시"]

# 알려진 실제 누락 구간 (모델링 시 0이 아닌 결측으로 취급)
KNOWN_MISSING = [
    ("화물청사", "2017-09", "9월 파일이 10월 데이터 중복 → 9월 실제 누락"),
    # 16년 1~4월은 참고사항상 누락 경고이나 일별로는 데이터 존재(부분 손실 가능)
]


def clean_master(master: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """완전중복 제거 + 분석구간 필터.

    차량번호(개인정보)가 없는 데이터(예: 향후 정보공개 분)도 동일 작동 —
    DUP_KEY 중 존재하는 컬럼만 사용. 주차권번호+입출차일시로 충분(차량번호 영향 0.02%).
    """
    n0 = len(master)
    keys = [c for c in DUP_KEY if c in master.columns]
    m = master.drop_duplicates(keys)
    n1 = len(m)
    m = m[(m["입차일시"] >= WINDOW_START) & (m["입차일시"] < WINDOW_END)]
    n2 = len(m)
    if verbose:
        print(f"  정제: 원본 {n0:,} → 중복제거 {n1:,} (-{n0-n1:,}) "
              f"→ 구간필터 {n2:,} (-{n1-n2:,})")
    return m.reset_index(drop=True)


# 요금 0원 = 유예시간(~10분) 내 비주차 통행(드롭오프/U턴/만차회차). 전 유형 검증됨.
# 주차수요(entries)는 요금>0 만 카운트, 0원 통행은 dropoffs로 별도 보존.
def _is_parking(m: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(m["요금"], errors="coerce").fillna(0) > 0


def build_daily(master: pd.DataFrame) -> pd.DataFrame:
    m = master.dropna(subset=["입차일시"]).copy()
    m["date"] = m["입차일시"].dt.normalize()
    m["_park"] = _is_parking(m)
    park = m[m["_park"]]
    daily = park.groupby(["lot", "date"]).agg(
        entries=("입차일시", "size"),
        revenue=("수입", "sum"),
        fee=("요금", "sum"),
        avg_stay_min=("주차시간_분", "mean"),
    ).reset_index()
    drop = (m[~m["_park"]].groupby(["lot", "date"]).size()
            .reset_index(name="dropoffs"))
    return daily.merge(drop, on=["lot", "date"], how="outer").fillna({"dropoffs": 0, "entries": 0})


def build_hourly(master: pd.DataFrame) -> pd.DataFrame:
    m = master.dropna(subset=["입차일시"]).copy()
    m["hour"] = m["입차일시"].dt.floor("h")
    m["_park"] = _is_parking(m)
    park = (m[m["_park"]].groupby(["lot", "hour"]).size()
            .reset_index(name="entries"))
    drop = (m[~m["_park"]].groupby(["lot", "hour"]).size()
            .reset_index(name="dropoffs"))
    return park.merge(drop, on=["lot", "hour"], how="outer").fillna(0)


def _save_all(master: pd.DataFrame, report: pd.DataFrame | None):
    P = config.PROCESSED_DIR
    P.mkdir(parents=True, exist_ok=True)
    master.to_parquet(P / "transactions.parquet", index=False)
    build_daily(master).to_parquet(P / "daily.parquet", index=False)
    build_hourly(master).to_parquet(P / "hourly.parquet", index=False)
    if report is not None:
        report.to_csv(P / "quality_report.csv", index=False, encoding="utf-8-sig")
    print(f"산출물 → {P}")


def rebuild_from_parquet():
    """저장된 transactions.parquet를 정제→재집계 (xls 재독 없이 빠름)."""
    P = config.PROCESSED_DIR
    print("== transactions.parquet 로드 후 정제/재집계 ...")
    master = pd.read_parquet(P / "transactions.parquet")
    master = clean_master(master)
    _save_all(master, report=None)
    _print_summary(master)


def _print_summary(master: pd.DataFrame):
    print("\n== 주차장별 거래수 ==")
    print(master["lot"].value_counts().to_string())
    print("\n== 연도별 입차 기간 ==")
    for y, sub in master.groupby(master["입차일시"].dt.year):
        print(f"  {y}: {sub['입차일시'].min()} ~ {sub['입차일시'].max()}  ({len(sub):,})")


def main():
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("== raw data 읽는 중 (전 파일) ...")
    t0 = time.time()
    master, report = load_all_transactions()
    print(f"\n총 {len(master):,} 거래(원본), {time.time() - t0:.0f}s 소요")

    master = clean_master(master)
    print("== 저장 중 ...")
    _save_all(master, report)
    _print_summary(master)


if __name__ == "__main__":
    # python src/aggregate.py          -> 전 xls 재독 후 집계
    # python src/aggregate.py rebuild  -> 저장된 parquet 정제/재집계(빠름)
    if len(sys.argv) > 1 and sys.argv[1] == "rebuild":
        rebuild_from_parquet()
    else:
        main()
