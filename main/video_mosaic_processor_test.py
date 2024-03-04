import datetime

import cv2
import numpy as np
from ultralytics import YOLO

from main.ffmpeg_video import get_video_properties, merge_audio
from deep_sort_realtime.deepsort_tracker import DeepSort


model = YOLO('yolov8n-face.pt')  # pretrained YOLOv8n model
tracker = DeepSort(max_age=50)
# 원본 영상 파일 이름
source_file_name = '롯데택배_얼굴_3명.mov'
source_video = f'/Users/seonwoo/Desktop/{source_file_name}'


# 원본 영상 파일 열기
# Open video file
cap = cv2.VideoCapture(source_video)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

video_props = get_video_properties(source_video) # 데이터 추출
print(video_props)

print(f"Original video FPS: {fps}")
print(f"Original video resolution: {width}x{height}")

CONFIDENCE_THRESHOLD = 0.6
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

# VideoWriter 설정
fourcc = cv2.VideoWriter_fourcc(*video_props['codec'])  # ffmpeg로부터 추출한 코덱에 따라 변경할 수 있습니다.

result_file_name = f'Result_{source_file_name}'
# 결과 영상 파일 이름
result_video = f'/Users/seonwoo/Desktop/Mosaic_Result/{result_file_name}'

out = cv2.VideoWriter(result_video, fourcc, video_props['fps'], (video_props['width'], video_props['height']))

# 결과 영상을 위한 VideoWriter 객체 생성

while cap.isOpened():

    start = datetime.datetime.now()

    ret, frame = cap.read()

    if not ret:
        break  # 프레임을 다 읽으면 종료

    detection = model.predict(source=[frame], save=False)

    results = []

    for result in detection:
        boxes = result.boxes  # Boxes object for bounding box outputs

        for data in result.boxes.data.tolist():
            confidence = float(data[4])
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            label = int(data[5])

            w, h = xmax - xmin, ymax - ymin  # Calculate width and height of the box

            # 감지된 얼굴 영역에 대한 모자이크 처리
            face = frame[ymin:ymax, xmin:xmax]  # Use ymin, ymax, xmin, xmax instead of y:y+h, x:x+w
            # 모자이크 처리를 위해 축소할 크기를 변경합니다. (더 작은 값으로)
            face = cv2.resize(face, (w // 10, h // 10))  # 여기서 모자이크 강도를 조절합니다.
            # 다시 원래 크기로 확대합니다. 모자이크 블록이 더 크게 보입니다.
            face = cv2.resize(face, (w, h), interpolation=cv2.INTER_AREA)
            frame[ymin:ymax, xmin:xmax] = face  # Replace the face area with the mosaic

            # cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
            # text = f'[Mosaic] X: {xmin} Y: {ymin} W: {xmax} H: {ymax}'
            # cv2.putText(frame, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, label])

        tracks = tracker.update_tracks(results, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()

            xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
            cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
            cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)


    # 모자이크 처리된 프레임을 결과 영상 파일에 쓰기
    out.write(frame)

    # 결과 프레임 표시

    end = datetime.datetime.now()

    total = (end - start).total_seconds()
    print(f'Time to process 1 frame: {total * 1000:.0f} milliseconds')

    fps = f'FPS: {1 / total:.2f}'
    cv2.putText(frame, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('Mosaic Face Detection', frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 작업 완료 후 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()

final_video = f'Include_Sound_{result_video}'

merge_audio(result_video, source_video, final_video)

