"""프로젝트 공통 설정 및 데이터 상수.

데이터 폴더(DATA_DIR)는 PC마다 다르므로 코드에 하드코딩하지 않는다.
아래 순서로 경로를 찾는다:

  1) 환경변수 KIMPO_DATA_DIR  (권장 — PC별로 한 번만 설정)
  2) src/data_dir.local.txt   (git에 안 올라가는 로컬 파일, 경로 한 줄)
  3) 기본값(회사 PC 경로)      (위 둘 다 없을 때)

집 PC 등 새 환경에서는 1) 또는 2)만 지정하면 코드는 그대로 동작한다.
"""
import os
from pathlib import Path

# 위 둘 다 없을 때 쓰는 기본값 (회사 PC)
_DEFAULT_DATA_DIR = (
    r"G:\조재원 백업(내부 정리중)\D드라이브\김포공항 주차빌딩"
    r"\주차관제시스템 데이터\교통센터 관련 (3)"
)

_LOCAL_FILE = Path(__file__).with_name("data_dir.local.txt")


def _resolve_data_dir() -> Path:
    # 1) 환경변수
    env = os.environ.get("KIMPO_DATA_DIR")
    if env:
        return Path(env.strip().strip('"'))
    # 2) 로컬 파일
    if _LOCAL_FILE.exists():
        line = _LOCAL_FILE.read_text(encoding="utf-8").strip().strip('"')
        if line:
            return Path(line)
    # 3) 기본값
    return Path(_DEFAULT_DATA_DIR)


DATA_DIR = _resolve_data_dir()

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"데이터 폴더를 찾을 수 없습니다: {DATA_DIR}\n"
        "환경변수 KIMPO_DATA_DIR 를 설정하거나 "
        f"{_LOCAL_FILE} 파일에 경로 한 줄을 적어주세요."
    )

YEAR_DIRS = {
    2016: DATA_DIR / "2016년",
    2017: DATA_DIR / "2017년",
    2018: DATA_DIR / "2018년",
}

# 집계 산출물 저장 위치 (프로젝트 내부, git 제외 — data/ 는 .gitignore됨)
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# 표준 주차장 키 (원본 주차장명 표기가 제각각이라 정규화)
PARKING_LOTS = ["국내선1", "국내선2", "국제선", "화물청사", "국제선버스", "항공지원센터"]


def normalize_lot(name) -> str:
    """원본 주차장명 -> 표준 키. 매칭 안되면 원본 반환."""
    s = str(name)
    if "항공지원" in s:
        return "항공지원센터"
    if "버스" in s:
        return "국제선버스"
    if "화물" in s:
        return "화물청사"
    if "국제선" in s:
        return "국제선"
    if "국내선" in s:
        if "2" in s or "제2" in s:
            return "국내선2"
        if "1" in s or "제1" in s:
            return "국내선1"
    return s

# ===== 데이터 주의사항 (참고사항.hwp 기준) =====

# 통계 작성 시 반드시 제외할 주차권번호 (출차완료 차량 재출차 허수 내역)
EXCLUDE_TICKET_NO = "9999999"

# DB서버 장애로 데이터 누락된 구간
MISSING_PERIOD = (2016, 1, 4)  # 2016년 1~4월

# 주차장별 운영 시작일 (이전 데이터 부재는 정상)
OPERATION_START = {
    "국제선버스": "2016-02-06",
    "항공지원센터": "2016-07-20",
}
