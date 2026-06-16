"""주차관제시스템 .xls raw data 공통 리더.

연도/주차장마다 export 포맷이 다르다. 이 모듈은 그 차이를 흡수한다:

  - 헤더 위치가 파일마다 다름 (0행 / 2행 / 5행) → 자동 탐지
  - 헤더에 줄바꿈·병합셀 (예: '주차권\\n번호'), 빈 열 섞임 → 정규화 후 이름으로 매핑
  - 한 파일이 시트 여러 개로 분할됨 (.xls 65,536행 한계) → 전 시트 합침
  - 입/출차일시가 문자열(2016) 또는 Excel 시리얼(2017~18) 혼재 → 양쪽 파싱
  - 주차권번호 '9999999'(허수) 제외, 입차일시 없는 행 제외

핵심 함수:
  read_parking_xls(path) -> pd.DataFrame  (표준화된 트랜잭션 테이블)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import xlrd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import EXCLUDE_TICKET_NO  # noqa: E402

# 표준 컬럼명 -> 정규화된 헤더에서 찾을 이름(들)
CORE_COLUMNS: dict[str, list[str]] = {
    "순번": ["순번"],
    "주차장명": ["주차장명"],
    "차량번호": ["차량번호"],
    "주차권번호": ["주차권번호"],
    "입차일시": ["입차일시"],
    "정산일시": ["정산일시"],
    "출차일시": ["출차일시"],
    "주차시간": ["주차시간"],
    "요금": ["요금"],
    "수입": ["수입"],
    "지불유형": ["지불유형"],
}


def _norm(v) -> str:
    """공백/줄바꿈 제거해 헤더 셀을 비교 가능한 형태로."""
    return re.sub(r"\s+", "", str(v))


def _find_header(sheet, max_scan: int = 15):
    """'순번'과 '입차일시'가 함께 있는 행을 헤더로 본다."""
    for r in range(min(max_scan, sheet.nrows)):
        cells = [_norm(sheet.cell_value(r, c)) for c in range(sheet.ncols)]
        if "순번" in cells and "입차일시" in cells:
            return r, cells
    return None, None


def _col_index(header_cells: list[str]) -> dict[str, int]:
    idx: dict[str, int] = {}
    for canon, variants in CORE_COLUMNS.items():
        for c, name in enumerate(header_cells):
            if name in variants:
                idx[canon] = c
                break
    return idx


def _to_datetime(value, datemode):
    if value in ("", None):
        return pd.NaT
    if isinstance(value, (int, float)):
        try:
            return pd.Timestamp(xlrd.xldate.xldate_as_datetime(value, datemode))
        except Exception:
            return pd.NaT
    return pd.to_datetime(str(value).strip(), errors="coerce")


def _duration_to_min(value):
    m = re.match(r"\s*(\d+):(\d+)", str(value))
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    return pd.NA


def _clean_ticket(s: pd.Series) -> pd.Series:
    # float로 읽힌 '73438.0' -> '73438'
    return (
        s.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def read_parking_xls(path) -> pd.DataFrame:
    """한 .xls 파일을 표준화된 트랜잭션 DataFrame으로 읽는다."""
    path = Path(path)
    book = xlrd.open_workbook(path)
    datemode = book.datemode

    frames: list[pd.DataFrame] = []
    prev_idx: dict[str, int] | None = None  # 헤더 없는 분할 시트가 재사용
    for sheet in book.sheets():
        hdr_row, hdr_cells = _find_header(sheet)
        if hdr_row is not None:
            idx = _col_index(hdr_cells)
            data_start = hdr_row + 1
            prev_idx = idx
        elif prev_idx is not None and sheet.nrows > 0:
            # .xls 65,536행 한계로 분할된 헤더 없는 연속 시트 → 직전 매핑 재사용
            idx = prev_idx
            data_start = 0
        else:
            continue
        if "입차일시" not in idx:
            continue
        recs = []
        for r in range(data_start, sheet.nrows):
            recs.append({c: sheet.cell_value(r, i) for c, i in idx.items()})
        if recs:
            frames.append(pd.DataFrame(recs))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # 날짜 파싱
    for col in ("입차일시", "출차일시", "정산일시"):
        if col in df:
            df[col] = df[col].apply(lambda v: _to_datetime(v, datemode))

    # 파생/수치
    if "주차시간" in df:
        df["주차시간_분"] = df["주차시간"].apply(_duration_to_min)
    for col in ("요금", "수입"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "주차권번호" in df:
        df["주차권번호"] = _clean_ticket(df["주차권번호"])

    # 필터: 9999999 허수 제외 + 입차일시 없는 행(서브헤더/빈행) 제외
    if "주차권번호" in df:
        df = df[df["주차권번호"] != EXCLUDE_TICKET_NO]
    df = df[df["입차일시"].notna()]

    df["source_file"] = path.name
    return df.reset_index(drop=True)


def _summary(df: pd.DataFrame) -> str:
    if df.empty:
        return "  (행 없음)"
    lines = [
        f"  행수: {len(df):,}",
        f"  입차 기간: {df['입차일시'].min()} ~ {df['입차일시'].max()}",
    ]
    if "주차장명" in df:
        lots = df["주차장명"].value_counts()
        lines.append(f"  주차장: {dict(lots)}")
    if "출차일시" in df:
        lines.append(f"  출차 결측: {df['출차일시'].isna().sum():,}")
    if "수입" in df:
        lines.append(f"  총수입: {df['수입'].sum():,.0f}")
    return "\n".join(lines)


if __name__ == "__main__":
    # 빠른 점검:  python src/etl.py <xls경로>
    for p in sys.argv[1:]:
        print(f"== {p}")
        print(_summary(read_parking_xls(p)))
