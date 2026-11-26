#!/usr/bin/env python
"""보건의료 결과파일 반출 전 점검 도구.

결과파일(csv·xlsx·docx)에 식별자, 소수 집계, 민감정보, 숨은 누출 경로가 남았는지 찾는다.
셀 값은 절대 출력하지 않는다. 규칙 이름, 열 이름, 건수, 행 번호만 출력한다.

    python export_check.py result.xlsx
    python export_check.py result.csv --config rules.toml --qi 성별,연령대
    python export_check.py results_folder/

근거 (2026-11 기준 공개된 최신본):
  - 「보건의료데이터 활용 가이드라인」(2025.12, 보건복지부·개인정보보호위원회, 5차 개정)
      본문 p.12  재식별 시 동의가 필요한 건강정보(정신질환, HIV, 성매개감염병, 희귀질환, 학대·낙태)
      부록 8     사망자 정형 의료데이터 네거티브 가명처리 기준 1~23번 (참고 기준, 칼럼명 예시 포함)
  - 「개인정보 보호법」 제23조(민감정보), 제24조(고유식별정보), 제28조의2~제28조의5(가명정보 특례)
  - 「가명정보 처리 가이드라인」(2026.03 전면개정, 개인정보보호위원회) — 반출(제3자 제공)은 처리 환경에 따라
    위험도가 올라가므로 적정성 검토를 거친다
소수 셀 기준(k) 등 수치 기준은 위 문서에 일률적으로 정해져 있지 않으므로 설정 파일로 받는다.
이 도구는 의심 항목을 찾는 보조 도구다. 반출 가능 여부는 기관 DRB와 개인정보보호책임자가 판단한다.
"""
from __future__ import annotations

import argparse
import re
import sys
import tomllib
import zipfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

BLOCK, WARN, INFO = '위반', '주의', '참고'


@dataclass
class Finding:
    rule: str       # 규칙 이름
    level: str      # 위반 / 주의 / 참고
    basis: str      # 근거
    column: str = ''
    count: int = 0
    rows: list = field(default_factory=list)  # 위치: 표는 행 번호(헤더 = 1행), 문서·OCR은 줄 번호, 영상은 초
    unit: str = '행'

    def line(self) -> str:
        where = f' [{self.column}]' if self.column else ''
        cnt = f' {self.count}건' if self.count else ''
        rows = f' ({self.unit} {", ".join(map(str, self.rows[:5]))}{" …" if len(self.rows) > 5 else ""})' if self.rows else ''
        return f'[{self.level}] {self.rule}{cnt}{where}{rows} — {self.basis}'


# ---------------------------------------------------------------- 열 이름 규칙 (부록 8)
# (부록8 번호, 항목, 열 이름 정규식, 등급, 처리원칙)
COLUMN_RULES = [
    (1, '성명', r'성명|이름|환자명|PAT_?NM|PATIENT_?NAME|^NAME$|CAREGIVER_?NM', BLOCK, '완전 삭제'),
    (2, '고유식별번호', r'주민|RES_?NO|JUMIN|RRN|여권|PASS_?NO|운전면허|외국인등록|FOREIG', BLOCK, '완전 삭제'),
    (3, '환자등록번호', r'등록번호|환자번호|차트번호|PT_?NO|EMR_?ID|CHART_?NO|MRN', BLOCK, '가명처리 후 원본 삭제'),
    (4, '전화번호', r'전화|연락처|휴대폰|TEL_?NO|MOB_?NO|HOME_?TEL|EMGCY|PHONE', BLOCK, '완전 삭제'),
    (5, '이메일·계정', r'이메일|EMAIL|E_MAIL|WBS_?ID|PORTAL_?ID', BLOCK, '완전 삭제'),
    (6, '상세주소', r'주소|ADDR|ZIP|우편번호', WARN, '시·도, 시·군·구까지만 유지'),
    (7, '계좌·카드', r'계좌|카드번호|ACCT_?NO|CARD_?NO', BLOCK, '완전 삭제'),
    (8, '보험증 번호', r'보험증|INS_?CARD|PRIVATE_?INS', BLOCK, '완전 삭제'),
    (9, '보호자·유족', r'보호자|GUARDIAN|사망신고|DTH_?RPT|CERT_?APPL', BLOCK, '완전 삭제'),
    (10, '생년월일', r'생년월일|BIRTH', WARN, '가명처리'),
    (12, '입·퇴원일자', r'입원일|퇴원일|ADM_?DT|ADMISSION_?DATE|DSCH_?DT|DISCHARGE_?DATE', WARN, '가명처리'),
    (14, '수술·시술일', r'수술일|시술일|OP_?DT|PROC_?DT', WARN, '가명처리'),
    (15, '검사일시', r'검사일|채취일|SPECIMEN_?DT|RESULT_?DT|STUDY_?DT', WARN, '가명처리'),
    (17, '의료인 성명', r'의사명|간호사명|처방의|DOC_?NM|NURSE_?NM|ORDERER_?NM', BLOCK, '완전 삭제'),
    (18, 'IP·URL', r'^IP$|IP_?ADDR|URL|HOME_?PAGE', BLOCK, '완전 삭제'),
    (19, '보험자·피보험자 번호', r'피보험|POLICY_?NO|SUBSCRIBER', BLOCK, '완전 삭제'),
    (20, '차량번호', r'차량|CAR_?NO', BLOCK, '완전 삭제'),
    (21, '장비·이식체 일련번호', r'DEVICE_?SN|IMPLANT_?SN|일련번호', BLOCK, '완전 삭제'),
    (22, '생체정보', r'지문|홍채|얼굴|FINGER|IRIS|FACE_?TEMPLATE', BLOCK, '완전 삭제'),
]


# 부록 8이 본보기로 든 HIPAA Safe Harbor 18개 식별자 중 영문 열 이름으로 흔한 것 (45 CFR §164.514(b)(2))
HIPAA_RULES = [
    ('성명(영문)', r'^(FIRST|LAST|MIDDLE|MAIDEN)(_?NAME)?$|^(GIVEN|FAMILY|FULL)_?NAME$', BLOCK, '(A) Names'),
    ('SSN', r'^SSN$|SOCIAL_?SECURITY', BLOCK, '(G) Social Security numbers'),
    ('운전면허·여권', r'^DRIVERS?$|DRIVERS?_?LICEN|^PASSPORT', BLOCK, '(R) Any other unique identifying number'),
    ('주 미만 지역', r'^(ADDRESS|CITY|COUNTY|FIPS|ZIP|ZIPCODE|POSTAL_?CODE|LAT|LON|LATITUDE|LONGITUDE|BIRTHPLACE)$', WARN, '(B) Geographic subdivisions smaller than a state'),
    ('날짜(연도 제외)', r'^(BIRTH|DEATH|ADMIT|DISCHARGE)_?DATE$|^BIRTHDATE$|^DEATHDATE$|^DOB$', WARN, '(C) All elements of dates except year'),
    ('의무기록·계정번호', r'^(MRN|MEDICAL_?RECORD|ACCOUNT_?NO|HEALTH_?PLAN)', BLOCK, '(H)(I)(J) Record, plan, account numbers'),
    ('의료인 등록번호', r'^NPI$', BLOCK, '부록8-17 의료인 실명 및 등록번호'),
]


def key(col) -> str:
    return re.sub(r'\s+', '', str(col)).upper()


# ---------------------------------------------------------------- 값 형식 검사
# 숫자 경계: 앞뒤가 숫자·영문이 아니고 소수(0.9001011234567)의 일부도 아니다. 문장 끝 마침표는 허용
_NUM_L, _NUM_R = r'(?<![0-9A-Za-z])(?<!\d\.)', r'(?![0-9A-Za-z])(?!\.\d)'


def _valid_birth(yy: int, mm: int, dd: int, g: int) -> bool:
    century = 1900 if g in (1, 2, 5, 6) else 2000
    try:
        date(century + yy, mm, dd)
        return True
    except ValueError:
        return False


def rrn_like(v: str) -> bool:
    """주민등록번호 형식: 앞 6자리가 실제 날짜이고 7번째 자리가 1~8."""
    return any(_valid_birth(*(int(x) for x in m.groups()))
               for m in re.finditer(_NUM_L + r'(\d{2})(\d{2})(\d{2})\s?-?\s?([1-8])\d{6}' + _NUM_R, v))


def rrn_partial(v: str) -> bool:
    """뒷자리를 가린 주민번호(900101-1******): 생년월일과 성별이 그대로 드러난다."""
    return any(_valid_birth(*(int(x) for x in m.groups()))
               for m in re.finditer(_NUM_L + r'(\d{2})(\d{2})(\d{2})\s?-\s?([1-8])[*xX●○#]{1,6}(?![0-9A-Za-z])', v))


def luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for d in reversed(digits):
        n = int(d)
        if alt:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
        alt = not alt
    return total % 10 == 0


def card_like(v: str) -> bool:
    return any(luhn_ok(re.sub(r'\D', '', m.group()))
               for m in re.finditer(r'(?<!\d)\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}(?!\d)', v))


PHONE = re.compile(_NUM_L + r'0(?:1[016789]|2|[3-6][1-5])[-) ]?\d{3,4}[- ]?\d{4}' + _NUM_R)
EMAIL = re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+')
IPV4 = re.compile(r'(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])')
ADDRESS = re.compile(r'[가-힣\d]+(?:로|길)\s?\d+(?:-\d+)?|\d+동\s?\d+호')
# 구분자는 앞뒤가 같아야 하고(2024-03-15, 2024.03.15, 20240315), 숫자·영문·소수점에 붙어 있으면 날짜로 보지 않는다
# (금액 201301.12, UUID 안의 숫자열 오탐 방지)
_EDGE_L, _EDGE_R = r'(?<![0-9A-Za-z.])', r'(?![0-9A-Za-z]|\.\d)'
_TIME = r'(?:[T ]\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?'  # ISO 시각 꼬리 2016-12-28T22:25:54Z
FULL_DATE = re.compile(_EDGE_L + r'(?:19|20)\d{2}(?P<sep>[-./]?)(?:0[1-9]|1[0-2])(?P=sep)(?:0[1-9]|[12]\d|3[01])' + _TIME + _EDGE_R
                       + r'|(?:19|20)\d{2}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일')
SURNAMES = set('김이박최정강조윤장임한오서신권황안송류전홍고문양손배백허유남심노하곽성차주우구민진나엄채원천방공현함변염여추도소석선설마길연위표명기반왕금옥육인맹제모탁국어은편용예봉경사부가복태목형피두감음빈동온호범좌팽승간상시갈단견당화창')

# 문장 속 성명: "홍길동 환자", "김민서님", "성명: 이준우". 성씨로 시작해야 하고 일반 명사는 뺀다
NAME_SUFFIX = re.compile(r'(?<![가-힣])([가-힣]{2,4})\s?(?:환자|님|씨|보호자|선생님|교수님)(?![가-힣])')
NAME_LABEL = re.compile(r'(?:성명|이름|환자명|보호자명|수진자)\s*[:：]?\s*([가-힣]{2,4}?)(?=[^가-힣]|$|주민|연락|생년|성별|나이|등록)')
NOT_NAMES = {'외래', '입원', '소아', '성인', '해당', '신규', '기존', '전체', '일반', '중증', '경증', '응급', '대상', '남자', '여자',
             '남성', '여성', '노인', '고령', '담당', '주치', '전공', '간호', '이번', '이전', '정상', '전원', '조기', '고위험', '저위험',
             '당뇨', '고혈압', '암', '모든', '각', '한', '위', '이', '그', '신장', '심장', '성별', '연령', '이상', '방문'}


def name_in_text(v: str) -> bool:
    for rx in (NAME_SUFFIX, NAME_LABEL):
        for m in rx.finditer(v):
            w = m.group(1)
            if w[0] in SURNAMES and w not in NOT_NAMES and w[:2] not in NOT_NAMES:
                return True
    return False


UUID = re.compile(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}')
US_SSN = re.compile(r'(?<![\d.-])(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?![\d.-])')

VALUE_RULES = [
    ('주민등록번호 형식', rrn_like, BLOCK, '부록8-2 고유식별번호'),
    ('가린 주민번호(생년월일·성별 노출)', rrn_partial, BLOCK, '부록8-2·8-10, 앞자리만으로 생년월일·성별 특정'),
    ('문장 속 성명', name_in_text, WARN, '부록8-1 성명 (텍스트·로그 포함)'),
    ('미국 SSN 형식', lambda v: bool(US_SSN.search(v)), BLOCK, 'HIPAA (G) Social Security numbers'),
    ('전화번호 형식', lambda v: bool(PHONE.search(v)), BLOCK, '부록8-4 연락처'),
    ('이메일 형식', lambda v: bool(EMAIL.search(v)), BLOCK, '부록8-5 이메일'),
    ('카드번호 형식', card_like, BLOCK, '부록8-7 재정정보 (Luhn 검증 통과)'),
    ('IP 주소 형식', lambda v: bool(IPV4.search(v)), BLOCK, '부록8-18 디지털'),
    ('상세주소 형식', lambda v: bool(ADDRESS.search(v)), WARN, '부록8-6 상세주소'),
    ('연-월-일 날짜', lambda v: bool(FULL_DATE.search(v)), WARN, '부록8-10~16 일자는 가명처리'),
]

# ---------------------------------------------------------------- 민감정보 (본문 p.12)
SENSITIVE_ICD = [
    ('정신질환', r'F\d'),
    ('후천성면역결핍증(HIV)', r'B2[0-4]|Z21'),
    ('성매개감염병(근사: A50~A64)', r'A5\d|A6[0-4]'),
    ('학대', r'T74'),
    ('낙태', r'O04'),
    ('희귀질환 산정특례코드', r'V\d{3}'),
]
ICD_COL = r'ICD|KCD|상병|진단|DIAG|^DX|질환코드|특례'


# ---------------------------------------------------------------- 설정
DEFAULT_CONFIG = {
    'min_cell': 3,             # 예시값. 집계 인원이 이 값 미만이면 소수 셀. 기관 DRB 기준으로 바꿀 것
    'site_terms': [],          # 결과에 남으면 안 되는 기관명·약칭
    'rare_disease_codes': '',  # 희귀질환 코드 목록 CSV 경로(첫 열 = 코드). 질병관리청 희귀질환 헬프라인 자료로 만든다
    'age_top': 90,             # 이 나이 이상은 상단코딩 대상으로 표시
    'ocr': True,               # 이미지·영상·DICOM 화면 속 글자 읽기 (rapidocr 필요)
    'face': True,              # 얼굴 탐지 (opencv 필요)
    'skin_ratio': 0.25,        # 피부색 영역이 이 비율 이상이면 문신·흉터 수동 확인 대상으로 표시
    'video_interval_sec': 1.0, # 영상에서 몇 초마다 한 장씩 볼지
    'video_max_frames': 300,   # 영상 한 개에서 최대 몇 장까지 볼지
}


def load_config(path: str | None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if path:
        with open(path, 'rb') as f:
            cfg.update(tomllib.load(f))
    codes = set()
    if cfg['rare_disease_codes']:
        codes = {key(c).replace('.', '') for c in pd.read_csv(cfg['rare_disease_codes'], dtype=str).iloc[:, 0].dropna()}
    cfg['_rare'] = codes
    return cfg


# ---------------------------------------------------------------- 표 점검
COUNT_COL = r'^(N|CNT|COUNT|FREQ|FREQUENCY)$|_(CNT|COUNT|N)$|^N_|환자수|대상자수|인원|명수|빈도|건수|N_?PAT'
PERSONAL_VALUE_COL = r'SUM|합계|평균|MEAN|AVG|중앙값|MEDIAN|용량|금액|일수|최대|최소|MAX|MIN'
TOTAL_WORDS = {'합계', '전체', '소계', '계', 'TOTAL', 'ALL', 'OVERALL'}
QI_COL = r'성별|^SEX$|GENDER|연령|나이|^AGE|_AGE$|지역|시도|시군구|^ZIP|기관|^SITE'


ROW_OFFSET = 2  # 표: 헤더가 1행이라 데이터 첫 행 = 2행. 문서·OCR 줄은 1부터


def scan_values(s: pd.Series, rule) -> pd.Series:
    """정규식은 고유값에만 적용하고 행으로 펼친다 (대용량 대비)."""
    _, test, _, _ = rule
    uniq = s.dropna().unique()
    bad = {u for u in uniq if test(u)}
    return s.isin(bad)


def check_lines(lines: list[str], cfg: dict, source: str) -> list[Finding]:
    """문장·OCR 결과 같은 비정형 텍스트 점검. rows = 줄 번호(1부터)."""
    df = pd.DataFrame({source: list(lines)})
    found = check_table(df, cfg, prose=True, row_offset=1)
    for f in found:
        f.unit = '줄'
    return found


def check_table(df: pd.DataFrame, cfg: dict, qi: list[str] | None = None, prose: bool = False,
                row_offset: int = ROW_OFFSET) -> list[Finding]:
    out: list[Finding] = []
    df = df.astype('string').reset_index(drop=True)
    cols = list(df.columns)

    def row_numbers(mask: pd.Series) -> list[int]:
        return [int(i) + row_offset for i in mask[mask].index]

    # 1) 열 이름 (문서·OCR 줄은 열 이름이 없음)
    for no, item, rx, lvl, action in ([] if prose else COLUMN_RULES):
        for c in cols:
            if re.search(rx, key(c)):
                out.append(Finding(f'{item} 열', lvl, f'부록8-{no}, {action}', column=str(c)))
    for item, rx, lvl, basis in ([] if prose else HIPAA_RULES):
        for c in cols:
            if re.search(rx, key(c)) and not any(f.column == str(c) and f.level == BLOCK for f in out):
                out.append(Finding(f'{item} 열', lvl, f'HIPAA Safe Harbor {basis}', column=str(c)))

    # 2) 값 형식 + 기관명 + 성명 추정
    site_rx = re.compile('|'.join(re.escape(t) for t in cfg['site_terms']), re.I) if cfg['site_terms'] else None
    for c in cols:
        s = df[c].fillna('')
        is_code = re.search(r'코드|CODE|EDI|ATC', key(c))
        if len(s[s != '']) and s[s != ''].str.fullmatch(UUID).mean() > 0.9:
            continue  # 무작위 UUID 열은 가명 키로 보고 값 형식 검사를 건너뛴다 (16진수 숫자열 오탐 방지)
        for rule in VALUE_RULES:
            if is_code and rule[0] != '이메일 형식':
                continue
            m = scan_values(s, rule)
            if m.any():
                out.append(Finding(rule[0], rule[2], rule[3], str(c), int(m.sum()), row_numbers(m)))
        if site_rx:
            m = s.str.contains(site_rx, na=False)
            if m.any():
                out.append(Finding('기관명', BLOCK, '기관 특정 가능, 삭제·코드화', str(c), int(m.sum()), row_numbers(m)))
        filled = s[s != '']
        if len(filled) >= 3 and filled.nunique() / len(filled) > 0.5:
            name_like = filled.str.fullmatch(r'[가-힣]{2,4}') & filled.str[0].isin(SURNAMES)
            if name_like.mean() > 0.8:
                out.append(Finding('성명 의심 열', BLOCK, '부록8-1, 값 대부분이 한국인 성씨로 시작하는 2~4자', str(c)))

    if prose:
        return out

    # 3) 나이
    for c in cols:
        if re.search(r'연령|나이|^AGE$|_AGE$', key(c)) and not re.search(r'대|GROUP|BAND|구간', key(c)):
            a = pd.to_numeric(df[c], errors='coerce')
            if a.notna().mean() > 0.9:
                top = a >= cfg['age_top']
                if top.any():
                    out.append(Finding(f'{cfg["age_top"]}세 이상', WARN, '부록8-11, 상단코딩', str(c), int(top.sum()), row_numbers(top)))
                if a.nunique() > 20:
                    out.append(Finding('정수 나이', WARN, '부록8-11, 연령대로 범주화', str(c)))

    # 4) 집계표 / 행 단위 판별
    count_cols = [c for c in cols if re.search(COUNT_COL, key(c))]
    if count_cols:
        k = cfg['min_cell']
        for c in count_cols:
            n = pd.to_numeric(df[c], errors='coerce')
            small = n.between(1, k - 1)
            if small.any():
                out.append(Finding(f'소수 셀 (1~{k - 1}명)', BLOCK, f'설정 min_cell={k}, 억제·병합',
                                   str(c), int(small.sum()), row_numbers(small)))
            one = n == 1
            value_cols = [v for v in cols if v != c and re.search(PERSONAL_VALUE_COL, key(v))]
            if one.any() and value_cols:
                out.append(Finding('1명 행의 요약값 = 개인값', BLOCK, f'{value_cols} 함께 억제', str(c),
                                   int(one.sum()), row_numbers(one)))
        is_total = pd.Series(False, index=df.index)
        for c in cols:
            is_total |= df[c].fillna('').str.strip().str.upper().isin(TOTAL_WORDS)
        if is_total.any():
            out.append(Finding('합계 행', WARN, '합계에서 빼면 억제한 셀이 복원되는지 확인(2차 억제)',
                               count=int(is_total.sum()), rows=row_numbers(is_total)))
    else:
        out.append(Finding('행 단위 자료', INFO, '인원 열이 없어 개인 단위 자료로 판단, 준식별자 조합 점검'))
        q = [c for c in (qi or [c for c in cols if re.search(QI_COL, key(c))]) if c in cols]
        if q and len(df):
            size = df.groupby(q, dropna=False)[q[0]].transform('size')
            rare = size < cfg['min_cell']
            if rare.any():
                out.append(Finding(f'준식별자 조합 {cfg["min_cell"]}명 미만', BLOCK, f'{q} 범주화·병합',
                                   ','.join(map(str, q)), int(rare.sum()), row_numbers(rare)))

    # 5) 민감 상병·희귀질환
    for c in cols:
        if not re.search(ICD_COL, key(c)):
            continue
        codes = df[c].fillna('').str.upper().str.replace('.', '', regex=False).str.strip()
        for label, rx in SENSITIVE_ICD:
            m = codes.str.match(rf'^(?:{rx})', na=False)
            if m.any():
                out.append(Finding(label, WARN, '본문 p.12, 동의 원칙 또는 추가 보호조치 승인', str(c),
                                   int(m.sum()), row_numbers(m)))
        if cfg['_rare']:
            m = codes.isin(cfg['_rare'])
            if m.any():
                out.append(Finding('희귀질환 코드', WARN, '본문 p.12·부록8-23, 지정 희귀질환', str(c),
                                   int(m.sum()), row_numbers(m)))
    return out


# ---------------------------------------------------------------- 파일 구조 점검 (숨은 누출)
def check_container(path: Path) -> list[Finding]:
    out: list[Finding] = []
    suf = path.suffix.lower()
    if suf not in ('.xlsx', '.xlsm', '.docx'):
        return out
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        core = z.read('docProps/core.xml').decode('utf-8', 'ignore') if 'docProps/core.xml' in names else ''
        if suf == '.docx':
            body = ''.join(z.read(n).decode('utf-8', 'ignore') for n in names
                           if re.fullmatch(r'word/(document|header\d*|footer\d*)\.xml', n))
            if '<w:delText' in body:
                out.append(Finding('변경추적으로 지운 글자 남음', BLOCK, '모두 적용 후 저장'))
            if '<w:vanish/>' in body:
                out.append(Finding('숨긴 글자', WARN, '숨김 해제 후 확인'))
            if 'word/comments.xml' in names:
                out.append(Finding('메모', WARN, '내용 확인 후 삭제'))
            if any(n.startswith('word/embeddings/') for n in names):
                out.append(Finding('삽입된 개체', WARN, '원본 표가 통째로 들어 있을 수 있음'))
        else:
            if any('pivotCache' in n for n in names):
                out.append(Finding('피벗 캐시', WARN, '원본 데이터가 캐시에 남을 수 있음'))
            if any('externalLink' in n for n in names):
                out.append(Finding('외부 링크', WARN, '다른 파일 경로 참조'))
            if any('comments' in n.lower() for n in names):
                out.append(Finding('메모', WARN, '내용 확인 후 삭제'))
    if re.search(r'<dc:creator>[^<]+|<cp:lastModifiedBy>[^<]+', core):
        out.append(Finding('문서 속성에 작성자', INFO, '문서 검사로 제거'))
    if suf in ('.xlsx', '.xlsm'):
        from openpyxl import load_workbook
        wb = load_workbook(path, read_only=False)
        for ws in wb.worksheets:
            if ws.sheet_state != 'visible':
                out.append(Finding(f'숨김 시트 [{ws.title}]', WARN, '숨김 해제 후 확인'))
            hidden_rows = sum(1 for d in ws.row_dimensions.values() if d.hidden)
            hidden_cols = [c for c, d in ws.column_dimensions.items() if d.hidden]
            if hidden_rows or hidden_cols:
                out.append(Finding(f'숨긴 행·열 [{ws.title}]', WARN, f'숨긴 행 {hidden_rows}개, 숨긴 열 {hidden_cols}'))
    return out


def read_text(path: Path) -> str:
    for enc in ('utf-8-sig', 'cp949'):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError('utf-8 / cp949 로 읽을 수 없음')


def looks_tabular(text: str) -> str | None:
    """앞 200줄이 같은 수의 구분자로 나뉘면 그 구분자, 아니면 None(문서형 텍스트)."""
    lines = [l for l in text.splitlines()[:200] if l.strip()]
    for sep in (',', '	', '|', ';'):
        k = lines[0].count(sep) if lines else 0
        if k and sum(l.count(sep) == k for l in lines) >= 0.9 * len(lines):
            return sep
    return None


def load_tables(path: Path) -> dict[str, tuple[pd.DataFrame, bool]]:
    """{이름: (표, 문서형 여부)}. 문서형이면 한 열에 줄(문단)이 하나씩 들어 있다."""
    suf = path.suffix.lower()
    if suf in ('.xlsx', '.xlsm', '.xls'):
        return {k: (v, False) for k, v in pd.read_excel(path, sheet_name=None, dtype=str).items()}
    if suf in ('.rds', '.rdata', '.rda'):
        import rdata  # MIT 라이선스 R 파일 리더
        obj = rdata.read_rds(path) if suf == '.rds' else rdata.read_rda(path)
        frames = {'RDS': obj} if suf == '.rds' else dict(obj)
        frames = {k: v for k, v in frames.items() if isinstance(v, pd.DataFrame)}
        if not frames:
            raise ValueError('R 파일 안에 data.frame 이 없음 (list·모델 객체는 data.frame 으로 저장 후 점검)')
        return {k: (v.reset_index(drop=True).astype('string'), False) for k, v in frames.items()}
    if suf == '.docx':
        import docx
        d = docx.Document(path)
        tables = {'본문': (pd.DataFrame({'본문': [p.text for p in d.paragraphs]}), True)}
        hf = [p.text for s in d.sections for part in (s.header, s.footer) for p in part.paragraphs if p.text]
        if hf:
            tables['머리글·바닥글'] = (pd.DataFrame({'머리글·바닥글': hf}), True)
        for i, t in enumerate(d.tables, 1):
            rows = [[c.text.strip() for c in r.cells] for r in t.rows]
            if len(rows) > 1:
                head = [h or f'열{j}' for j, h in enumerate(rows[0], 1)]
                head = [h if head.count(h) == 1 else f'{h}_{j}' for j, h in enumerate(head, 1)]
                tables[f'표{i}'] = (pd.DataFrame(rows[1:], columns=head), False)
        return tables
    text = read_text(path)
    sep = looks_tabular(text)
    if sep:
        from io import StringIO
        return {'표': (pd.read_csv(StringIO(text), dtype=str, sep=sep), False)}
    return {'본문': (pd.DataFrame({'본문': text.splitlines()}), True)}


def check_file(path: Path, cfg: dict, qi: list[str] | None = None) -> list[tuple[str, Finding]]:
    import media
    if media.kind(path):
        return media.check(path, cfg)
    results = [('파일', f) for f in check_container(path)]
    for name, (df, prose) in load_tables(path).items():
        if prose:
            results += [(name, f) for f in check_table(df, cfg, prose=True, row_offset=1)]
        else:
            results += [(name, f) for f in check_table(df, cfg, qi)]
    return results


TABLE_SUFFIXES = {'.csv', '.tsv', '.txt', '.md', '.log', '.xlsx', '.xlsm', '.xls', '.docx', '.rds', '.rdata', '.rda'}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='반출 전 결과파일 점검 (값은 출력하지 않음)')
    ap.add_argument('paths', nargs='+', help='파일 또는 폴더')
    ap.add_argument('--config', help='설정 파일(TOML). 예: rules.example.toml')
    ap.add_argument('--qi', help='행 단위 자료의 준식별자 열, 쉼표 구분')
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    qi = [q.strip() for q in a.qi.split(',')] if a.qi else None

    import media
    suffixes = TABLE_SUFFIXES | media.SUFFIXES
    files = []
    for p in map(Path, a.paths):
        files += sorted(f for f in p.rglob('*') if f.suffix.lower() in suffixes) if p.is_dir() else [p]

    blocked = False
    for f in files:
        print(f'\n■ {f}')
        try:
            res = check_file(f, cfg, qi)
        except Exception as e:  # 한 파일이 실패해도 나머지는 계속
            print(f'  [읽기 실패] {e}')
            blocked = True
            continue
        for lvl in (BLOCK, WARN, INFO):
            for sheet, fd in res:
                if fd.level == lvl:
                    print(f'  ({sheet}) {fd.line()}')
        nb = sum(fd.level == BLOCK for _, fd in res)
        nw = sum(fd.level == WARN for _, fd in res)
        print(f'  → 위반 {nb} / 주의 {nw}' if res else '  → 발견 없음')
        blocked |= nb > 0
    print('\n※ 셀 값은 출력하지 않았습니다. 반출 가능 여부는 기관 DRB가 판단합니다.')
    return 1 if blocked else 0


if __name__ == '__main__':
    sys.exit(main())
