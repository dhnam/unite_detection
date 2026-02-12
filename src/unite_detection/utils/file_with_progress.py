import os
import zipfile
from pathlib import Path

from tqdm.auto import tqdm


def copy_with_progress(src: str|Path, dst:str|Path):
    # 파일 크기 확인
    file_size = os.path.getsize(src)

    # 파일을 바이너리 모드로 열기
    with open(src, "rb") as fsrc:
        with open(dst, "wb") as fdst:
            # tqdm 설정 (단위는 바이트, 총 크기 지정)
            with tqdm(
                total=file_size, unit="B", unit_scale=True, desc=os.path.basename(src)
            ) as pbar:
                while True:
                    # 1MB씩 읽어서 복사
                    chunk = fsrc.read(1024 * 1024)
                    if not chunk:
                        break
                    fdst.write(chunk)
                    pbar.update(len(chunk))  # 프로그래스바 업데이트


def unzip_by_size(zip_path: str|Path, extract_to: str|Path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        # 전체 파일의 압축 해제 후 총 용량 계산
        total_size = sum(file.file_size for file in zip_ref.infolist())

        with tqdm(total=total_size, unit="B", unit_scale=True, desc="해제 중") as pbar:
            for file_info in zip_ref.infolist():
                zip_ref.extract(file_info, extract_to)
                pbar.update(file_info.file_size)  # 각 파일 용량만큼 게이지 증가
