"""공개 데이터로 검증.

    python benchmark_public.py            # 전부 (Synthea 샘플 약 9MB 를 data/ 에 한 번 내려받음)
    python benchmark_public.py --skip-ocr # OCR 시험 빼고 빠르게

쓰는 공개 데이터 (저장소에는 넣지 않고 실행할 때 받는다)
  A. Synthea 합성 환자 CSV 샘플 (Apache-2.0)                  열 단위 식별자 탐지
  B. pydicom 내장 시험 DICOM 파일 (MIT)                       헤더 식별자 탐지 / 비식별 사본 오탐
  C. scikit-image 공개 이미지 + pydicom CT 픽셀 (BSD/MIT)      화면 속 글자에 가짜 식별자를 심어 OCR 탐지
  D. scikit-image 공개 이미지                                 얼굴 탐지
  E. 합성 영상                                                 프레임 속 글자 탐지
심는 값은 전부 무작위로 만든 가짜다.
"""
import argparse
import io
import random
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import export_check as ec
import media

CFG = ec.load_config(None)
DATA = Path(__file__).parent / 'data'
SYNTHEA_URL = 'https://synthetichealth.github.io/synthea-sample-data/downloads/latest/synthea_sample_data_csv_latest.zip'
FONT = None


def table(title, header, rows):
    print(f'\n#### {title}\n')
    print('| ' + ' | '.join(header) + ' |')
    print('|' + '---|' * len(header))
    for r in rows:
        print('| ' + ' | '.join(map(str, r)) + ' |')


# ---------------------------------------------------------------- A. Synthea
# HIPAA Safe Harbor 기준 정답: 환자 식별에 쓰일 수 있는 열
SYNTHEA_TRUTH = {
    'patients.csv': {'BIRTHDATE', 'DEATHDATE', 'SSN', 'DRIVERS', 'PASSPORT', 'FIRST', 'MIDDLE', 'LAST', 'MAIDEN',
                     'BIRTHPLACE', 'ADDRESS', 'CITY', 'COUNTY', 'FIPS', 'ZIP', 'LAT', 'LON'},
    'encounters.csv': {'START', 'STOP'},
    'providers.csv': {'NAME', 'ADDRESS', 'CITY', 'ZIP', 'LAT', 'LON', 'NPI'},
}


def bench_synthea():
    DATA.mkdir(exist_ok=True)
    z = DATA / 'synthea_csv.zip'
    if not z.exists():
        urllib.request.urlretrieve(SYNTHEA_URL, z)
    rows = []
    with zipfile.ZipFile(z) as zf:
        names = {Path(n).name: n for n in zf.namelist()}
        for fname, truth in SYNTHEA_TRUTH.items():
            df = pd.read_csv(zf.open(names[fname]), dtype=str)
            flagged = {f.column for f in ec.check_table(df, CFG) if f.level in (ec.BLOCK, ec.WARN)}
            flagged = {c for c in flagged if c in df.columns}
            tp, fn, fp = truth & flagged, truth - flagged, flagged - truth
            rows.append((fname, f'{len(df):,}행 {len(df.columns)}열', f'{len(tp)}/{len(truth)}',
                         ', '.join(sorted(fn)) or '없음', ', '.join(sorted(fp)) or '없음'))
    table('A. Synthea 합성 환자 데이터: 식별자 열 탐지', ['파일', '규모', '잡은 식별자 열', '놓친 열', '잘못 잡은 열'], rows)


# ---------------------------------------------------------------- B. DICOM
def bench_dicom():
    import warnings
    import pydicom
    warnings.filterwarnings('ignore')
    files = sorted((Path(pydicom.__file__).parent / 'data' / 'test_files').rglob('*.dcm'))  # 패키지에 들어 있는 것만
    has_name = caught = clean_fp = total = 0
    for f in files:
        try:
            ds = pydicom.dcmread(f, force=True)
        except Exception:
            continue
        if 'PixelData' not in ds:
            continue
        total += 1
        name = str(ds.get('PatientName', '')).strip()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'a.dcm'
            try:
                ds.save_as(p)
            except Exception:  # 일부 시험 파일은 일부러 깨진 형식이라 저장이 안 된다
                total -= 1
                continue
            cfg = {**CFG, 'ocr': False, 'face': False}
            found = media.check(p, cfg)
            if name:
                has_name += 1
                caught += any(fd.rule == 'DICOM 환자 이름' for _, fd in found)
            # 비식별 사본: PS3.15 대상 속성을 지우고 다시 점검
            for kw, _, _ in media.DICOM_ID:
                if kw in ds:
                    delattr(ds, kw)
            ds.remove_private_tags()
            ds.PatientIdentityRemoved = 'YES'
            try:
                ds.save_as(p)
            except Exception:
                continue
            found = media.check(p, cfg)
            clean_fp += any(fd.level == ec.BLOCK and fd.rule.startswith('DICOM') for _, fd in found)
    table('B. pydicom 공개 시험 DICOM: 헤더 점검',
          ['대상', '결과'],
          [('픽셀이 있는 시험 파일', f'{total}개'),
           ('환자 이름이 들어 있는 파일 중 탐지', f'{caught}/{has_name}'),
           ('식별 속성을 지운 사본에서 헤더 위반 오탐', f'{clean_fp}/{total}')])


# ---------------------------------------------------------------- C. OCR 심기
def base_images():
    from pydicom.data import get_testdata_file
    import pydicom
    from skimage import data
    imgs = []
    for n in ['coffee', 'rocket', 'coins', 'brick', 'grass', 'gravel', 'hubble_deep_field', 'retina', 'colorwheel', 'page']:
        im = getattr(data, n)()
        im = im if im.ndim == 3 else np.stack([im] * 3, -1)
        imgs.append((n, im[..., :3].astype('uint8')))
    for n in ['CT_small.dcm', 'MR_small.dcm']:
        a = media.to_uint8(pydicom.dcmread(get_testdata_file(n)).pixel_array)
        a = np.kron(a, np.ones((4, 4), dtype='uint8'))  # 작은 시험 영상은 키워서 글자를 올린다
        imgs.append((n, np.stack([a] * 3, -1)))
    return imgs


def fake_values(r):
    y, m, d = r.randint(40, 99), r.randint(1, 12), r.randint(1, 28)
    return {
        '주민등록번호 형식': f'주민 {y:02d}{m:02d}{d:02d}-{r.choice("12")}{r.randint(0, 999999):06d}',
        '전화번호 형식': f'TEL 010-{r.randint(1000, 9999)}-{r.randint(1000, 9999)}',
        '연-월-일 날짜': f'Study {r.randint(2015, 2025)}-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}',
        '문장 속 성명': f'성명: {r.choice("김이박최정강조윤")}{r.choice("민서지현준우")}{r.choice("호연아수")}',
    }


DECOYS = ['HR 0.82 (0.61-1.10)', 'p < 0.001', 'Value 0.9001011234567', 'EDI 052400010', 'v1.2.3 build 2024',
          'Series 3  Image 12', 'WL 40 WW 400', 'Slice 2.50 mm', 'kVp 120  mAs 250', 'AUC 0.914']


def draw(img, text, r):
    from PIL import Image, ImageDraw, ImageFont
    global FONT
    pil = Image.fromarray(img)
    size = r.randint(18, 34)
    font = ImageFont.truetype(FONT, size) if FONT else ImageFont.load_default()
    d = ImageDraw.Draw(pil)
    w, h = pil.size

    def box(fnt):
        b = d.textbbox((0, 0), text, font=fnt, stroke_width=2)
        return b[2] - b[0], b[3] - b[1]
    tw, th = box(font)
    while tw > w - 10 and size > 12:  # 글자가 화면 밖으로 잘리지 않게 줄인다
        size -= 2
        font = ImageFont.truetype(FONT, size) if FONT else ImageFont.load_default()
        tw, th = box(font)
    x, y = r.randint(5, max(5, w - tw - 5)), r.randint(5, max(5, h - th - 10))
    d.text((x, y), text, font=font, fill='white', stroke_width=2, stroke_fill='black')
    return np.array(pil)


def bench_ocr(seed=11, per_image=2):
    global FONT
    for f in ('C:/Windows/Fonts/malgun.ttf', '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
              '/System/Library/Fonts/AppleSDGothicNeo.ttc'):
        if Path(f).exists():
            FONT = f
            break
    r = random.Random(seed)
    cfg = {**CFG, 'face': False}
    hits, tries = {}, {}
    fp, n_decoy = 0, 0
    for name, base in base_images():
        for _ in range(per_image):
            for rule, text in fake_values(r).items():
                found, _ = media.scan_pixels(draw(base.copy(), text, r), cfg, name)
                tries[rule] = tries.get(rule, 0) + 1
                hits[rule] = hits.get(rule, 0) + any(f.rule == rule for f in found)
            decoy = draw(base.copy(), r.choice(DECOYS), r)
            found, _ = media.scan_pixels(decoy, cfg, name)
            n_decoy += 1
            fp += any(f.level == ec.BLOCK for f in found)
    rows = [(k, f'{hits[k]}/{tries[k]}', f'{hits[k] / tries[k]:.0%}') for k in tries]
    rows.append(('(오탐) 통계값·코드·장비값만 있는 화면', f'{fp}/{n_decoy}', f'{fp / n_decoy:.0%}'))
    table('C. 공개 이미지·CT/MR 픽셀에 가짜 식별자를 그려 넣은 OCR 탐지', ['심은 것', '탐지', '비율'], rows)


# ---------------------------------------------------------------- D. 얼굴
def bench_faces():
    from skimage import data
    people = ['astronaut', 'camera']
    others = ['coffee', 'chelsea', 'rocket', 'coins', 'brick', 'grass', 'gravel', 'horse', 'page', 'hubble_deep_field',
              'immunohistochemistry', 'retina', 'cell', 'logo', 'colorwheel', 'moon', 'text', 'clock']
    def faces(n):
        im = getattr(data, n)()
        im = (im * 255).astype('uint8') if im.dtype == bool else im
        return media.count_faces(im if im.dtype == np.uint8 else media.to_uint8(im)) or 0
    tp = sum(faces(n) > 0 for n in people)
    fp = sum(faces(n) > 0 for n in others)
    table('D. scikit-image 공개 이미지: 얼굴 탐지 (YuNet)', ['대상', '결과'],
          [('사람 얼굴이 있는 사진', f'{tp}/{len(people)} 탐지'), ('얼굴 없는 사진 (고양이·동전·조직 슬라이드 등)', f'{fp}/{len(others)} 오탐')])


# ---------------------------------------------------------------- E. 영상
def bench_video():
    import cv2
    r = random.Random(5)
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for label, plant in (('3~5초에 전화번호 자막', True), ('통계값 자막만 (깨끗)', False)):
            p = Path(td) / 'v.mp4'
            vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*'mp4v'), 10, (640, 360))
            for i in range(100):
                frame = np.full((360, 640, 3), 40, 'uint8')
                frame = draw(frame, f'HR 0.82 p=0.0{i % 9 + 1}', r)
                if plant and 30 <= i < 50:
                    frame = draw(frame, f'보호자 010-{r.randint(1000, 9999)}-{r.randint(1000, 9999)}', r)
                vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            vw.release()
            found = media.check(p, {**CFG, 'face': False})
            blocks = [fd for _, fd in found if fd.level == ec.BLOCK]
            rows.append((label, ', '.join(f'{fd.rule} {fd.rows}초' for fd in blocks) or '위반 없음'))
    table('E. 합성 영상 (1초에 한 장 점검)', ['영상', '결과'], rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-ocr', action='store_true')
    a = ap.parse_args()
    bench_synthea()
    bench_dicom()
    bench_faces()
    if not a.skip_ocr:
        bench_ocr()
        bench_video()


if __name__ == '__main__':
    main()
