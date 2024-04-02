import asyncio
import os

import cv2

from core.config import option
from core.util.ffmpeg_video import get_video_properties

before_frame = None
frame_lock = asyncio.Lock()  # 프레임 접근을 위한 비동기 락
cap = None
width = None
height = None
out = None
result_video = None
video_props = None
duration = None

class VideoSource:

    # 전역 변수로 최신 프레임 저장

    def start(self):
        global cap
        global width
        global height
        global out
        global result_video
        global video_props
        global duration
        # 디렉터리 경로 설정
        directory = option.source_directory

        # 디렉터리 내의 파일 목록 가져오기
        files = os.listdir(directory)

        # 디렉터리 내의 첫 번째 파일 이름 얻기 (파일이 하나만 있다고 가정)
        source_file_name = files[0] if files else None

        source_video = f'{directory}/{source_file_name}'

        cap = cv2.VideoCapture(source_video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(width), int(height))

        video_props = get_video_properties(source_video)  # 데이터 추출
        print(video_props)

        print(f"Original video FPS: {fps}")
        print(f"Original video resolution: {width}x{height}")

        # VideoWriter 설정
        fourcc = cv2.VideoWriter_fourcc(*video_props['codec'])  # ffmpeg로부터 추출한 코덱에 따라 변경할 수 있습니다.
        # fourcc = cv2.VideoWriter.fourcc(*'DIVX')

        # 결과 영상 파일 이름
        result_video = f'{directory}/Result_{source_file_name}'

        # out = cv2.VideoWriter(result_video, fourcc, video_props['fps'], (video_props['width'], video_props['height']))
        out = cv2.VideoWriter(result_video, fourcc, video_props['fps'], size)

        duration = int(video_props['duration'])



