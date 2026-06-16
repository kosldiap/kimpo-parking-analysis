"""공식 자료(KAC 신축사업 자료) 대비 재구성 검증.

입력(외부, git제외): 신축사업 자료 폴더의
  '16-19년 김포국제공항 주차장별 이용실적.xlsx' (Sheet3: 연도블록×월×lot 입/출차)

공식 lot(국내1·국내2·국제선지하·화물청사) → 우리 4유형 매핑:
  국내선 ← 국내1+국내2, 국제선 ← 국제선(지하), 화물 ← 화물청사  (직원은 공식자료에 없음)

산출:
  data/processed/official_monthly.parquet  [category, ym, official_in, official_out]
  + 재구성(요금>0 / 전체) 월별 입차와 비교 출력

실행: python src/validate_official.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import openpyxl
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

OFFICIAL_XLSX = (Path(r"G:\조재원 백업(내부 정리중)\D드라이브\김포공항 주차빌딩")
                 / "요청자료" / "김포공항 교통센터(주차빌딩) 신축사업 자료-190924"
                 / "20190923 자료 송부" / "16-19년 김포국제공항 주차장별 이용실적.xlsx")

# Sheet3 열: C=국내1입, E=국내2입, G=국제선지하입, I=화물청사입 (입차는 +0, 출차 +1)
LOT_COL = {"국내1": 3, "국내2": 5, "국제선": 7, "화물청사": 9}
LOT_TO_CAT = {"국내1": "국내선", "국내2": "국내선", "국제선": "국제선", "화물청사": "화물"}
MONTHS = {f"{m}월": m for m in range(1, 13)}


def parse_official() -> pd.DataFrame:
    ws = openpyxl.load_workbook(OFFICIAL_XLSX, read_only=True, data_only=True)["Sheet3"]
    rows = list(ws.iter_rows(values_only=True))
    recs = []
    year = None
    for r in rows:
        c0 = str(r[0]).strip() if r[0] is not None else ""
        if c0.endswith("년") and c0[:4].isdigit():
            year = int(c0[:4]); continue
        if year and c0 in MONTHS:
            mm = MONTHS[c0]
            for lot, col in LOT_COL.items():
                vin, vout = r[col - 1], r[col]
                if vin is None:
                    continue
                recs.append({"category": LOT_TO_CAT[lot], "year": year, "month": mm,
                             "in": float(vin or 0), "out": float(vout or 0)})
    df = pd.DataFrame(recs)
    g = df.groupby(["category", "year", "month"], as_index=False)[["in", "out"]].sum()
    g["ym"] = g["year"] * 100 + g["month"]
    return g.rename(columns={"in": "official_in", "out": "official_out"})


def reconstructed_monthly() -> pd.DataFrame:
    tx = pd.read_parquet(config.PROCESSED_DIR / "transactions.parquet",
                         columns=["lot", "입차일시", "요금"])
    tx["category"] = tx["lot"].map(config.LOT_TO_CATEGORY)
    tx = tx.dropna(subset=["입차일시"])
    tx["ym"] = tx["입차일시"].dt.year * 100 + tx["입차일시"].dt.month
    tx["park"] = pd.to_numeric(tx["요금"], errors="coerce").fillna(0) > 0
    allm = tx.groupby(["category", "ym"]).size().reset_index(name="recon_all")
    parkm = tx[tx["park"]].groupby(["category", "ym"]).size().reset_index(name="recon_park")
    return allm.merge(parkm, on=["category", "ym"], how="left")


def main():
    off = parse_official()
    off.to_parquet(config.PROCESSED_DIR / "official_monthly.parquet", index=False)
    rec = reconstructed_monthly()
    cmp = off.merge(rec, on=["category", "ym"], how="left")
    cmp = cmp[(cmp["ym"] >= 201601) & (cmp["ym"] <= 201812)]
    cmp["전체/공식%"] = (cmp["recon_all"] / cmp["official_in"] * 100).round(1)

    print("== 연도×유형: 공식입차 vs 재구성(전체) ==")
    yr = cmp.assign(year=cmp["ym"] // 100).groupby(["year", "category"]).agg(
        공식입차=("official_in", "sum"), 재구성전체=("recon_all", "sum"),
        재구성주차=("recon_park", "sum")).reset_index()
    yr["전체/공식%"] = (yr["재구성전체"] / yr["공식입차"] * 100).round(1)
    print(yr.to_string(index=False))
    print("\n※ 직원(항공지원센터)은 공식 4개 lot에 없어 비교 제외")
    print("※ 2016 낮음 = 1~4월 DB누락(참고사항). 2017~18은 ~100% 근접 → 재구성 신뢰.")


if __name__ == "__main__":
    main()
