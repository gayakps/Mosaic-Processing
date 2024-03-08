import asyncio

import cv2

from main.core.util.ffmpeg_video import get_video_properties

source_file_name = '공항.mov'
source_video = f'/Users/seonwoo/Desktop/{source_file_name}'

cap = cv2.VideoCapture(source_video)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

video_props = get_video_properties(source_video) # 데이터 추출
print(video_props)

print(f"Original video FPS: {fps}")
print(f"Original video resolution: {width}x{height}")

# VideoWriter 설정
fourcc = cv2.VideoWriter_fourcc(*video_props['codec'])  # ffmpeg로부터 추출한 코덱에 따라 변경할 수 있습니다.

result_file_name = f'Result_{source_file_name}.mp4'
# 결과 영상 파일 이름
result_video = f'/Users/seonwoo/Desktop/Mosaic_Result/{result_file_name}'

out = cv2.VideoWriter(result_video, fourcc, video_props['fps'], (video_props['width'], video_props['height']))
duration = int(video_props['duration'])

# 전역 변수로 최신 프레임 저장
current_frame = None
frame_lock = asyncio.Lock()  # 프레임 접근을 위한 비동기 락
