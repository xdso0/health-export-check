"""이미지·영상·DICOM 점검.

세 갈래로 본다.
  1. 메타데이터   EXIF·PNG 텍스트·DICOM 헤더에 남은 식별자
  2. 화면 속 글자 OCR로 읽은 뒤 표·문서와 같은 값 형식 규칙을 적용 (번인 텍스트)
  3. 얼굴         OpenCV 정면 얼굴 탐지

근거:
  - 보건의료데이터 활용 가이드라인(2025.12) 영상정보 가명처리: 헤더 식별자, 번인 텍스트, 안면 재구성
  - DICOM PS3.15 Annex E (Basic Application Level Confidentiality Profile) 의 제거 대상 속성
OCR 결과 글자는 출력하지 않는다. 규칙 이름, 위치(줄·초), 건수만 출력한다.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from export_check import BLOCK, INFO, WARN, Finding, check_lines

IMAGE = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif', '.webp'}
VIDEO = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v'}
DICOM = {'.dcm', '.dicom'}
SUFFIXES = IMAGE | VIDEO | DICOM


def kind(path: Path) -> str | None:
    suf = path.suffix.lower()
    if suf in IMAGE:
        return 'image'
    if suf in VIDEO:
        return 'video'
    if suf in DICOM:
        return 'dicom'
    if suf == '' and path.is_file():  # 확장자 없는 DICOM: 128바이트 뒤 'DICM'
        with open(path, 'rb') as f:
            head = f.read(132)
        if head[128:132] == b'DICM':
            return 'dicom'
    return None


# ---------------------------------------------------------------- OCR · 얼굴
@lru_cache(maxsize=1)
def ocr_engine():
    try:
        from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR
    except ImportError:
        return None
    # 각도 분류기는 끈다: 똑바른 숫자 줄을 뒤집힌 글자로 오인하는 경우가 있었다
    return RapidOCR(params={'Rec.lang_type': LangRec.KOREAN, 'Rec.ocr_version': OCRVersion.PPOCRV5,
                            'Rec.model_type': ModelType.MOBILE, 'Global.use_cls': False,
                            'Global.log_level': 'error'})


def ocr_lines(img: np.ndarray) -> list[str] | None:
    eng = ocr_engine()
    if eng is None:
        return None
    res = eng(img)
    return list(res.txts or []) if res is not None else []


YUNET_URL = ('https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/'
             'face_detection_yunet_2023mar.onnx')
YUNET_PATH = Path.home() / '.cache' / 'health-export-check' / 'face_detection_yunet_2023mar.onnx'


@lru_cache(maxsize=1)
def face_detector():
    """OpenCV YuNet 얼굴 탐지기 (opencv_zoo 공개 모델, 첫 실행 때 한 번 내려받음)."""
    try:
        import cv2
    except ImportError:
        return None
    if not YUNET_PATH.exists():
        import urllib.request
        YUNET_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
        except OSError:
            return None
    return cv2.FaceDetectorYN.create(str(YUNET_PATH), '', (320, 320), 0.8)


def count_faces(img: np.ndarray, threshold: float = 0.8) -> int | None:
    det = face_detector()
    if det is None:
        return None
    import cv2
    rgb = np.stack([img] * 3, -1) if img.ndim == 2 else img[..., :3]
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    scale = 640 / max(h, w) if max(h, w) > 640 else 1.0  # 큰 사진은 줄여서 속도 확보
    if scale != 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))
    det.setInputSize((bgr.shape[1], bgr.shape[0]))
    det.setScoreThreshold(threshold)
    _, faces = det.detect(bgr)
    return 0 if faces is None else len(faces)


def to_uint8(arr: np.ndarray) -> np.ndarray:
    a = arr.astype('float32')
    lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
    a = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * 255
    return a.astype('uint8')


def scan_pixels(img: np.ndarray, cfg: dict, where: str) -> tuple[list[Finding], dict]:
    """한 장의 픽셀을 OCR·얼굴 점검. 반환: (발견, 상태)"""
    out, state = [], {}
    if cfg.get('ocr', True):
        lines = ocr_lines(img)
        state['ocr'] = lines is not None
        if lines:
            for f in check_lines(lines, cfg, '화면 속 글자'):
                f.column, f.unit = f'{where} 화면 속 글자', '줄'
                out.append(f)
    n = 0
    if cfg.get('face', True):
        n = count_faces(img)
        state['face'] = n is not None
        if n:
            out.append(Finding('얼굴 의심', BLOCK, '영상정보 가명처리: 얼굴은 가리거나 잘라낸 뒤 반출', where, n))
    ratio = skin_ratio(img)
    if ratio >= cfg.get('skin_ratio', 0.25):
        # 얼굴이 함께 있으면 사람 사진이 확실하므로 주의, 아니면 피부색 비슷한 물체일 수 있어 참고
        out.append(Finding('피부 노출 많음', WARN if state.get('face') and n else INFO,
                           f'피부색 영역 {ratio:.0%}: 문신·흉터·점 같은 신체 특징은 사람이 확인 (문신 자동 탐지 아님)', where))
    return out, state


def skin_ratio(img: np.ndarray) -> float:
    """YCrCb 피부색 범위에 드는 픽셀 비율. 흑백(CT·X-ray 등)은 0.

    문신 자체를 찾는 공개 모델이 마땅치 않아, 피부가 넓게 찍힌 사진을 사람 확인 대상으로 걸러내는 용도다.
    나무·가죽·고양이 털처럼 피부색과 비슷한 물체도 걸린다.
    """
    if img.ndim != 3 or img.shape[2] < 3:
        return 0.0
    rgb = img[..., :3]
    if np.abs(rgb[..., 0].astype(int) - rgb[..., 2]).mean() < 3:  # 사실상 흑백
        return 0.0
    import cv2
    ycc = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2YCrCb)
    cr, cb = ycc[..., 1], ycc[..., 2]
    mask = (cr >= 135) & (cr <= 173) & (cb >= 85) & (cb <= 127)
    return float(mask.mean())


def engine_notes(states: list[dict], cfg: dict) -> list[Finding]:
    notes = []
    if cfg.get('ocr', True) and states and not any(s.get('ocr') for s in states):
        notes.append(Finding('OCR 미설치', INFO, '화면 속 글자는 점검하지 못함 (pip install rapidocr onnxruntime)'))
    if cfg.get('face', True) and states and not any(s.get('face') for s in states):
        notes.append(Finding('얼굴 탐지 미설치', INFO, '얼굴은 점검하지 못함 (pip install opencv-python-headless)'))
    return notes


# ---------------------------------------------------------------- 이미지
EXIF_RISK = {  # 태그 번호: (이름, 등급)
    34853: ('GPS 위치', BLOCK), 315: ('작성자(Artist)', WARN), 42032: ('카메라 소유자', WARN),
    42033: ('기기 일련번호', WARN), 36867: ('촬영 일시', WARN), 306: ('파일 일시', INFO),
    270: ('이미지 설명', INFO), 37510: ('사용자 주석', INFO), 40093: ('작성자(Windows)', WARN),
}


def check_image(path: Path, cfg: dict) -> list[tuple[str, Finding]]:
    from PIL import Image, ImageSequence
    out: list[tuple[str, Finding]] = []
    with Image.open(path) as im:
        exif = im.getexif()
        texts = []
        for tag, (name, lvl) in EXIF_RISK.items():
            if tag in exif and exif[tag] not in (None, '', b''):
                out.append(('메타데이터', Finding(f'EXIF {name}', lvl, '메타데이터 제거 후 반출', 'EXIF')))
                if isinstance(exif[tag], (str, bytes)):
                    texts.append(exif[tag].decode('utf-8', 'ignore') if isinstance(exif[tag], bytes) else exif[tag])
        texts += [str(v) for v in im.info.values() if isinstance(v, str)]  # PNG tEXt 등
        out += [('메타데이터', f) for f in check_lines(texts, cfg, '메타데이터 글자')] if texts else []
        frames = [np.array(fr.convert('RGB')) for fr in ImageSequence.Iterator(im)][:20]  # 여러 장 TIFF·GIF
    states = []
    for i, img in enumerate(frames, 1):
        found, st = scan_pixels(img, cfg, f'{i}번째 장' if len(frames) > 1 else '이미지')
        states.append(st)
        out += [('픽셀', f) for f in found]
    out += [('안내', f) for f in engine_notes(states, cfg)]
    return out


# ---------------------------------------------------------------- DICOM (PS3.15 Annex E 일부)
DICOM_ID = [  # (키워드, 이름, 등급)
    ('PatientName', '환자 이름', BLOCK), ('PatientID', '환자 ID', BLOCK), ('OtherPatientIDs', '다른 환자 ID', BLOCK),
    ('OtherPatientIDsSequence', '다른 환자 ID', BLOCK), ('PatientBirthDate', '생년월일', BLOCK),
    ('PatientAddress', '환자 주소', BLOCK), ('PatientTelephoneNumbers', '환자 전화', BLOCK),
    ('MedicalRecordLocator', '의무기록 위치', BLOCK), ('PatientMotherBirthName', '어머니 성명', BLOCK),
    ('ReferringPhysicianName', '의뢰의 이름', WARN), ('PerformingPhysicianName', '시행의 이름', WARN),
    ('OperatorsName', '촬영자 이름', WARN), ('PhysiciansOfRecord', '담당의 이름', WARN),
    ('InstitutionName', '기관명', WARN), ('InstitutionAddress', '기관 주소', WARN), ('StationName', '장비 이름', WARN),
    ('DeviceSerialNumber', '장비 일련번호', WARN), ('AccessionNumber', '접수번호', WARN), ('StudyID', '검사 ID', WARN),
    ('StudyDate', '검사일', WARN), ('SeriesDate', '시리즈일', WARN), ('AcquisitionDate', '촬영일', WARN),
    ('ContentDate', '생성일', WARN), ('PatientAge', '나이', INFO), ('PatientSex', '성별', INFO),
]


def check_dicom(path: Path, cfg: dict) -> list[tuple[str, Finding]]:
    import pydicom
    out: list[tuple[str, Finding]] = []
    ds = pydicom.dcmread(path, force=True)
    basis = 'DICOM PS3.15 Annex E 제거·대체 대상'
    for kw, name, lvl in DICOM_ID:
        v = ds.get(kw)
        if v is not None and str(v.value if hasattr(v, 'value') else v).strip() not in ('', '[]'):
            out.append(('헤더', Finding(f'DICOM {name}', lvl, basis, kw)))
    if str(ds.get('PatientIdentityRemoved', '')).upper() != 'YES':
        out.append(('헤더', Finding('비식별 표시 없음', INFO, 'PatientIdentityRemoved(0012,0062) ≠ YES', 'PatientIdentityRemoved')))
    if str(ds.get('BurnedInAnnotation', '')).upper() == 'YES':
        out.append(('헤더', Finding('번인 주석 있음', BLOCK, 'BurnedInAnnotation(0028,0301) = YES, 화면 속 글자 확인', 'BurnedInAnnotation')))
    private = sum(1 for el in ds if el.tag.is_private)
    if private:
        out.append(('헤더', Finding('사설(private) 태그', WARN, '제조사 태그에 식별자가 들어갈 수 있음', count=private)))
    texts = [str(el.value) for el in ds if el.VR in ('LO', 'LT', 'ST', 'UT', 'SH', 'PN')]
    out += [('헤더', f) for f in check_lines(texts, cfg, '헤더 글자')]
    body = str(ds.get('BodyPartExamined', '')).upper()
    if ds.get('Modality') in ('CT', 'MR') and any(b in body for b in ('HEAD', 'BRAIN', 'SKULL', 'FACE')):
        out.append(('헤더', Finding('두부 CT/MR', WARN, '표면 렌더링으로 얼굴 재구성 가능, 안면 제거(defacing) 검토')))
    states = []
    try:
        arr = ds.pixel_array
    except Exception:
        arr = None
        out.append(('픽셀', Finding('픽셀 읽기 실패', INFO, '압축 형식 디코더가 없으면 화면 속 글자 점검 불가')))
    if arr is not None:
        if arr.ndim == 4 or (arr.ndim == 3 and arr.shape[-1] not in (3, 4)):
            arr = arr[len(arr) // 2]  # 여러 장이면 가운데 한 장
        found, st = scan_pixels(to_uint8(arr) if arr.dtype != np.uint8 else arr, cfg, '픽셀')
        states.append(st)
        out += [('픽셀', f) for f in found]
    out += [('안내', f) for f in engine_notes(states, cfg)]
    return out


# ---------------------------------------------------------------- 영상
def check_video(path: Path, cfg: dict) -> list[tuple[str, Finding]]:
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError('영상을 열 수 없음 (코덱 확인)')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, int(round(fps * cfg.get('video_interval_sec', 1.0))))
    hits: dict[tuple, list[float]] = defaultdict(list)
    proto: dict[tuple, Finding] = {}
    states, seen, idx = [], 0, 0
    while seen < cfg.get('video_max_frames', 300):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        t = round(idx / fps, 1)
        found, st = scan_pixels(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), cfg, f'{t}초')
        states.append(st)
        for f in found:
            k = (f.rule, f.level)
            hits[k].append(t)
            proto[k] = f
        seen += 1
        idx += step
    cap.release()
    out = []
    for k, times in hits.items():
        f = proto[k]
        out.append(('프레임', Finding(f.rule, f.level, f.basis, '영상', len(times), times, unit='초')))
    if total and seen * step < total:
        out.append(('안내', Finding('일부 구간만 점검', INFO, f'{seen}장 점검 (설정 video_max_frames)')))
    out.append(('안내', Finding('영상 메타데이터·음성 미점검', INFO, '촬영기기·위치 메타데이터와 음성은 별도 확인')))
    out += [('안내', f) for f in engine_notes(states, cfg)]
    return out


def check(path: Path, cfg: dict) -> list[tuple[str, Finding]]:
    return {'image': check_image, 'dicom': check_dicom, 'video': check_video}[kind(path)](path, cfg)
