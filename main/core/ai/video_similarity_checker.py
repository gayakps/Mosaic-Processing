import asyncio
import datetime

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2


async def preprocess_frame(frame, use_clahe=True):
    # 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # CLAHE 적용
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    else:
        # 히스토그램 평활화
        gray = cv2.equalizeHist(gray)

    return gray


async def resize_images(image1, image2):
    # 이미지 크기 일치
    height, width = image2.shape[:2]
    resized_image1 = cv2.resize(image1, (width, height))
    return resized_image1, image2


async def compare_frames(frame1, frame2):
    # 유효성 검사
    if frame1 is None or frame2 is None:
        return None

    frame1, frame2 = await resize_images(frame1, frame2)
        # 프레임 전처리
    frame1_processed = await preprocess_frame(frame1, use_clahe=True)
    frame2_processed = await preprocess_frame(frame2, use_clahe=True)
        # 두 프레임 사이의 SSIM 계산
    ssim_index, _ = await asyncio.to_thread(ssim, frame1_processed, frame2_processed, full=True)

    return ssim_index





async def add_text_and_line(frame1, frame2, line_color=(0, 0, 255), text_color=(255, 255, 255)):
    # 두 프레임의 높이를 확인하고, 높이가 다를 경우 더 작은 높이에 맞춰 조정합니다.
    height1, width1 = frame1.shape[:2]
    height2, width2 = frame2.shape[:2]
    new_height = min(height1, height2)

    # 두 이미지를 같은 높이로 조정
    frame1_resized = cv2.resize(frame1, (width1, new_height))
    frame2_resized = cv2.resize(frame2, (width2, new_height))

    # 각 프레임에 텍스트 추가
    cv2.putText(frame1_resized, 'Before Frame', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
    cv2.putText(frame2_resized, 'After Frame', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)

    # 두 이미지를 옆으로 결합
    combined_frame = cv2.hconcat([frame1_resized, frame2_resized])

    # 두 이미지 사이에 빨간 선 추가
    cv2.line(combined_frame, (width1, 0), (width1, new_height), line_color, 5)  # 선의 두께를 5로 설정

    return combined_frame
