import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_check import BLOCK, DEFAULT_CONFIG, check_file, check_table, card_like, rrn_like  # noqa: E402

CFG = {**DEFAULT_CONFIG, '_rare': set()}


def rules(df, **kw):
    return {f.rule for f in check_table(df, {**CFG, **kw})}


def test_rrn_needs_real_date():
    assert rrn_like('주민 900101-1234567')
    assert not rrn_like('901399-1234567')        # 13월 없음
    assert not rrn_like('0.9001011234567')       # 긴 소수는 아님


def test_card_needs_luhn():
    assert card_like('4111-1111-1111-1111')
    assert not card_like('1234-5678-9012-3456')


def test_codes_and_decimals_are_not_phones():
    df = pd.DataFrame({'EDI코드': ['052400010', '010123456'], '값': ['0.0101234567', '<0.001']})
    assert not any(f.level == BLOCK for f in check_table(df, CFG))


def test_phone_in_free_text():
    df = pd.DataFrame({'비고': ['', '보호자 010-1234-5678']})
    found = [f for f in check_table(df, CFG) if f.rule == '전화번호 형식']
    assert found and found[0].rows == [3]


def test_column_names_from_appendix8():
    r = rules(pd.DataFrame({'PAT_NM': ['x'], 'RES_NO': ['x'], 'DSCH_DT': ['x']}))
    assert {'성명 열', '고유식별번호 열', '입·퇴원일자 열'} <= r


def test_name_values_detected_without_name_column():
    df = pd.DataFrame({'col': ['김민서', '이준우', '박지현', '최서연']})
    assert '성명 의심 열' in rules(df)


def test_small_cell_uses_config():
    df = pd.DataFrame({'연령대': ['20-29', '30-39'], '환자수': ['4', '25']})
    assert not any(r.startswith('소수 셀') for r in rules(df, min_cell=3))
    assert any(r.startswith('소수 셀') for r in rules(df, min_cell=5))


def test_one_person_row_exposes_value():
    df = pd.DataFrame({'구분': ['A', 'B'], 'N': ['1', '40'], '평균 용량': ['12.5', '10.1']})
    assert '1명 행의 요약값 = 개인값' in rules(df)


def test_quasi_identifier_combination():
    df = pd.DataFrame({'성별': ['M'] * 5 + ['F'], '연령대': ['30-34'] * 5 + ['95-99']})
    assert any(r.startswith('준식별자 조합') for r in rules(df))


@pytest.mark.parametrize('code,label', [('F32.1', '정신질환'), ('B20', '후천성면역결핍증(HIV)'), ('T74.1', '학대'), ('O04', '낙태')])
def test_sensitive_icd(code, label):
    assert label in rules(pd.DataFrame({'ICD10': [code, 'I10']}))


def test_site_terms_from_config():
    df = pd.DataFrame({'메모': ['OO병원 전원', '없음']})
    assert '기관명' in rules(df, site_terms=['OO병원'])


def test_hidden_sheet(tmp_path):
    from openpyxl import Workbook
    wb = Workbook()
    wb.active.append(['연령대', '환자수'])
    wb.active.append(['20-29', '30'])
    ws = wb.create_sheet('원본')
    ws.sheet_state = 'hidden'
    p = tmp_path / 'out.xlsx'
    wb.save(p)
    assert any(f.rule.startswith('숨김 시트') for _, f in check_file(p, CFG))


def test_clean_aggregate_has_no_block():
    df = pd.DataFrame({'성별': ['M', 'F'], '연령대': ['50-59', '50-59'], '환자수': ['30', '41'], 'HR': ['0.82', '1.10']})
    assert not any(f.level == BLOCK for f in check_table(df, CFG))


def test_rrn_with_sentence_period_and_masked():
    from export_check import rrn_partial
    assert rrn_like('번호는 900101-1234567.')
    assert rrn_partial('900101-1******')
    assert not rrn_partial('2024-1***')


def test_iso_timestamp_and_money_are_handled():
    df = pd.DataFrame({'START': ['2016-12-28T22:25:54Z'], 'COST': ['201301.12']})
    found = {(f.rule, f.column) for f in check_table(df, CFG)}
    assert ('연-월-일 날짜', 'START') in found
    assert not any(c == 'COST' for _, c in found)


def test_uuid_column_is_not_phone():
    df = pd.DataFrame({'Id': ['0a1b2c3d-0101-2345-6789-0123456789ab', '1a1b2c3d-0213-4567-8901-0123456789ab']})
    assert not any(f.level == BLOCK for f in check_table(df, CFG))
