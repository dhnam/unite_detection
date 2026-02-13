import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
from pathlib import Path
from shutil import copy2

import cv2
from tqdm.auto import tqdm


def resize_single_video(item):
    src_path, dst_path, size = item
    # 폴더가 없으면 생성
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # FFmpeg 명령어 구성
    # -y: 기존 파일 덮어쓰기
    # -i: 입력 파일
    # -vf: 비디오 필터 (리사이즈)
    # -c:v libx264: H.264 코덱 사용
    # -crf 23: 일반적인 화질 설정 (낮을수록 고화질)
    # -preset veryfast: 속도 우선 인코딩
    # -an: 오디오 제거 (학습에 필요 없음, 용량 절감)
    cmd = [
        "ffmpeg",
        "-y",
        "-hwaccel",
        "cuda",  # 디코딩 가속 유지
        "-i",
        str(src_path),
        # 성공했던 코드의 필터 형식을 그대로 사용 (flags=lanczos 등)
        "-vf",
        f"scale={size[0]}:{size[1]}:flags=lanczos",
        "-c:v",
        "libx264",  # NVENC 대신 검증된 libx264 사용
        "-preset",
        "fast",  # 속도 조절
        "-crf",
        "23",  # 화질 설정
        "-pix_fmt",
        "yuv420p",  # 재생 호환성을 위해 추가
        "-an",  # 오디오 제거
        str(dst_path),
    ]

    # 실행 (로그는 숨김)
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # ... (cmd 설정 부분)


def preprocess_celebdf(src_root, dst_root, size=(384, 384), max_workers=8):
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    if dst_root.exists():
        print(f"Target directory {dst_root} already exists. Skipping preprocessing.")
        return

    # 모든 mp4 파일 찾기
    video_files = list(src_root.glob("*/*.mp4"))
    tasks = []

    for src_path in video_files:
        rel_path = src_path.relative_to(src_root)
        dst_path = dst_root / rel_path
        if not dst_path.exists():
            tasks.append((src_path, dst_path, size))

    print(f"Starting preprocessing: {len(tasks)} videos...")

    # 멀티프로세싱으로 병렬 처리
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(
            tqdm(
                executor.map(resize_single_video, tasks, chunksize=4), total=len(tasks)
            )
        )
    copy2(
        src_root / "List_of_testing_videos.txt", dst_root / "List_of_testing_videos.txt"
    )

    print("Preprocessing completed.")


def extract_frames_single_video(item):
    src_path, dst_dir, size = item

    if dst_dir.exists() and any(dst_dir.iterdir()):
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = str(dst_dir / "frame_%06d.jpg")

    # FFmpeg 명령어 변경
    # -vf "select='not(mod(n\,2))',scale=...":
    #   1. select: 짝수 프레임(0, 2, 4...)만 선택
    #   2. scale: 리사이즈
    # -vsync vfr: 선택된 프레임만 출력 (타임스탬프 유지를 위해 빈 프레임 생성 방지)
    cmd = [
        "ffmpeg",
        "-y",
        "-hwaccel",
        "cuda",
        "-i",
        str(src_path),
        "-vf",
        rf"select='not(mod(n\,2))',scale={size[0]}:{size[1]}:flags=lanczos",
        "-vsync",
        "vfr",  # Variable Frame Rate: 선택된 프레임만 씀
        "-q:v",
        "2",
        output_pattern,
        "-threads",
        "1",
        "-hide_banner",
        "-loglevel",
        "error",
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def preprocess_celebdf_frames(src_root, dst_root, size=(384, 384), max_workers=8):
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    video_files = list(src_root.glob("*/*.mp4"))
    tasks = []

    print("Scanning files...")
    for src_path in video_files:
        rel_path = src_path.relative_to(src_root)
        dst_dir = dst_root / rel_path.with_suffix("")

        if not dst_dir.exists() or not any(dst_dir.iterdir()):
            tasks.append((src_path, dst_dir, size))

    print(f"Starting frame extraction (Even frames only): {len(tasks)} videos...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(
            tqdm(
                executor.map(extract_frames_single_video, tasks, chunksize=1),
                total=len(tasks),
            )
        )

    txt_src = src_root / "List_of_testing_videos.txt"
    if txt_src.exists():
        copy2(txt_src, dst_root / "List_of_testing_videos.txt")

    print("Preprocessing completed.")


def process_single_image(args):
    """
    하나의 이미지를 처리하는 워커 함수입니다.
    multiprocessing에서 호출됩니다.

    Args:
        args (tuple): (source_path, target_path, target_size)
    """
    src_path, dst_path, target_size = args

    try:
        # 이미지 읽기
        img = cv2.imread(src_path)
        if img is None:
            return f"Error reading: {src_path}"

        # 리사이징 (너비, 높이)
        # 보간법은 cv2.INTER_LINEAR (기본값) 혹은 축소 시 cv2.INTER_AREA 추천
        resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

        # 저장할 폴더가 없으면 생성 (Race condition 방지를 위해 exist_ok=True)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # 이미지 저장
        cv2.imwrite(dst_path, resized_img)
        return None  # 성공 시 None 반환

    except Exception as e:
        return f"Exception at {src_path}: {str(e)}"


def preprocess_gta_v(source_dir: str | Path, output_dir: str | Path, target_size, num_workers=8):
    """
    메인 처리 함수
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # 1. 파일 목록 스캔
    # 구조: source_dir/Images/video_seq/000000.bmp
    # 패턴: 모든 하위 폴더의 images 폴더 안의 bmp 파일들
    print(f"Scanning files in {source_dir}...")

    # glob 패턴: source_dir 아래 모든 폴더(*) 아래 images 폴더 아래 *.bmp
    # 혹은 재귀적으로 찾으려면 rglob 사용 가능. 여기서는 데이터셋 구조에 맞춰 명시적으로 찾습니다.
    # SAIL-VOS 구조: root -> Images -> vid_seq -> *.bmp
    all_files = list(source_path.glob("Images/*/*.bmp"))

    if not all_files:
        print("No files found. Please check the source directory structure.")
        return

    print(f"Found {len(all_files)} images. Preparing tasks...")

    # 2. 작업 리스트 생성
    tasks = []

    for src_file in all_files:
        src_p = Path(src_file)

        # 경로 파싱
        # src_p: .../ah_1_mcs_1/000000.bmp
        # file_name: 000000.bmp
        # video_seq_name: ah_1_mcs_1 (부모의 부모 폴더 이름)

        file_name = src_p.with_suffix(".png").name
        video_seq_name = src_p.parent.name

        # 목표 경로 생성: output_dir/ah_1_mcs_1/000000.bmp
        # (images 폴더 depth를 제거함)
        dst_p = output_path / video_seq_name / file_name

        tasks.append((str(src_p), str(dst_p), target_size))

    # 3. 멀티프로세싱 실행
    print(f"Starting processing with {num_workers} workers...")
    print(f"Target Size: {target_size}")

    with Pool(processes=num_workers) as pool:
        # tqdm을 사용하여 진행률 표시
        # imap_unordered가 리스트를 미리 만들지 않아 메모리 효율적이며 순서 상관없이 처리됨
        results = list(
            tqdm(pool.imap_unordered(process_single_image, tasks), total=len(tasks))
        )

    # 4. 결과 리포트
    errors = [res for res in results if res is not None]

    print("\nProcessing Complete.")
    if errors:
        print(f"{len(errors)} errors occurred:")
        for err in errors[:10]:  # 처음 10개만 출력
            print(err)
        if len(errors) > 10:
            print("...")
    else:
        print("Successfully processed all images without errors.")
