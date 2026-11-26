"""위반 심기 시험: 깨끗한 합성 결과표에 위반을 하나씩 심고, 도구가 잡는 비율을 잰다.

    python benchmark.py            # 결과를 마크다운 표로 출력
    python benchmark.py --trials 500

모든 값은 무작위로 만든 가짜다. 실제 사람·기관과 관련 없다.
"""
import argparse
import random
import string

import pandas as pd

from export_check import BLOCK, DEFAULT_CONFIG, check_table, luhn_ok

CFG = {**DEFAULT_CONFIG, '_rare': set()}
SURNAMES = '김이박최정강조윤장임한오서신권황안송류전홍'
SYLLABLES = '민서지현준우예은하윤도연수아진영성호재원태희경미정'
SAFE_ICD = ['I10', 'E11', 'J45', 'K21', 'M54', 'N18', 'C50', 'I48', 'E78', 'K29']


def fake_name(r):
    return r.choice(SURNAMES) + ''.join(r.choice(SYLLABLES) for _ in range(r.choice((1, 2))))


def fake_rrn(r):
    y, m, d = r.randint(40, 99), r.randint(1, 12), r.randint(1, 28)
    return f'{y:02d}{m:02d}{d:02d}-{r.choice("12")}{r.randint(0, 999999):06d}'


def fake_card(r):
    body = [r.randint(0, 9) for _ in range(15)]
    for check in range(10):
        s = ''.join(map(str, body + [check]))
        if luhn_ok(s):
            return f'{s[:4]}-{s[4:8]}-{s[8:12]}-{s[12:]}'


def clean_rows(r, n=1000):
    """행 단위 결과표 (가명 연구ID, 범주화된 준식별자, 함정 값 포함)."""
    return pd.DataFrame({
        '연구ID': [f'R{i:05d}' for i in range(n)],
        '성별': [r.choice('MF') for _ in range(n)],
        '연령대': [f'{b}-{b + 9}' for b in (r.choice(range(20, 80, 10)) for _ in range(n))],
        '진단코드': [r.choice(SAFE_ICD) for _ in range(n)],
        'EDI코드': [f'{r.randint(0, 999999999):09d}' for _ in range(n)],          # 전화번호 오탐 함정
        '검사값': [f'{r.random():.13f}' for _ in range(n)],                       # 긴 소수 오탐 함정
        '비고': [r.choice(['', '특이사항 없음', '외래 추적', '재진']) for _ in range(n)],
    })


def clean_agg(r):
    """집계 결과표 (모든 셀 10명 이상, 통계값 포함)."""
    rows = [{'연령대': f'{b}-{b + 9}', '성별': s, '환자수': str(r.randint(10, 400)),
             'HR': f'{r.uniform(0.3, 2):.2f}', 'p': f'{r.random():.4f}'}
            for b in range(20, 80, 10) for s in 'MF']
    return pd.DataFrame(rows)


def put(df, col, r, value):
    df = df.copy()
    df.loc[r.randrange(len(df)), col] = value
    return df


# (위반 유형, 표 종류, 심는 함수, 잡았다고 볼 규칙 이름)
PLANTS = [
    ('주민등록번호 (비고 칸)', 'rows', lambda df, r: put(df, '비고', r, f'본인 {fake_rrn(r)} 확인'), '주민등록번호 형식'),
    ('휴대전화 (비고 칸)', 'rows', lambda df, r: put(df, '비고', r, f'연락 010-{r.randint(1000, 9999)}-{r.randint(1000, 9999)}'), '전화번호 형식'),
    ('이메일 (비고 칸)', 'rows', lambda df, r: put(df, '비고', r, f'{"".join(r.choices(string.ascii_lowercase, k=6))}@example.com'), '이메일 형식'),
    ('카드번호 (비고 칸)', 'rows', lambda df, r: put(df, '비고', r, f'환불 {fake_card(r)}'), '카드번호 형식'),
    ('연-월-일 날짜 (비고 칸)', 'rows', lambda df, r: put(df, '비고', r, f'{r.randint(2015, 2025)}-{r.randint(1, 12):02d}-{r.randint(1, 28):02d} 퇴원'), '연-월-일 날짜'),
    ('성명 열 (열 이름 평범)', 'rows', lambda df, r: df.assign(col_x=[fake_name(r) for _ in range(len(df))]), '성명 의심 열'),
    ('성명 열 (열 이름 명시)', 'rows', lambda df, r: df.assign(환자명=''), '성명 열'),
    ('정신질환 상병', 'rows', lambda df, r: put(df, '진단코드', r, r.choice(['F32', 'F20', 'F41'])), '정신질환'),
    ('HIV 상병', 'rows', lambda df, r: put(df, '진단코드', r, r.choice(['B20', 'B24', 'Z21'])), '후천성면역결핍증(HIV)'),
    ('준식별자 희소 조합', 'rows', lambda df, r: put(df, '연령대', r, '95-99'), '준식별자 조합'),
    ('소수 셀', 'agg', lambda df, r: put(df, '환자수', r, str(r.choice((1, 2)))), '소수 셀'),
]


def found(df, rule):
    return any(f.rule.startswith(rule) for f in check_table(df, CFG))


def run(trials, seed=7):
    r = random.Random(seed)
    results = []
    for label, kind, plant, rule in PLANTS:
        hit = sum(found(plant(clean_rows(r, 300) if kind == 'rows' else clean_agg(r), r), rule) for _ in range(trials))
        results.append((label, hit, trials))
    # 오탐: 깨끗한 표에서 '위반' 등급이 하나라도 나온 횟수
    fp_rows = sum(any(f.level == BLOCK for f in check_table(clean_rows(r, 300), CFG)) for _ in range(trials))
    fp_agg = sum(any(f.level == BLOCK for f in check_table(clean_agg(r), CFG)) for _ in range(trials))
    return results, fp_rows, fp_agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=200)
    a = ap.parse_args()
    results, fp_rows, fp_agg = run(a.trials)
    print('| 심은 위반 | 검출 | 검출률 |')
    print('|---|---|---|')
    for label, hit, n in results:
        print(f'| {label} | {hit}/{n} | {hit / n:.0%} |')
    print()
    print('| 깨끗한 표 (위반 없음) | 오탐 발생 | 오탐률 |')
    print('|---|---|---|')
    print(f'| 행 단위 결과표 300행 | {fp_rows}/{a.trials} | {fp_rows / a.trials:.0%} |')
    print(f'| 집계 결과표 | {fp_agg}/{a.trials} | {fp_agg / a.trials:.0%} |')


if __name__ == '__main__':
    main()
