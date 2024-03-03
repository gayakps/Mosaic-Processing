import cv2
import numpy as np
from ultralytics import YOLO

from main.ffmpeg_video import get_video_properties

model = YOLO('yolov8n-face.pt')  # pretrained YOLOv8n model
# 원본 영상 파일 이름
source_video = '/Users/seonwoo/Desktop/롯데택배_얼굴_3명.mov'


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

# VideoWriter 설정
fourcc = cv2.VideoWriter_fourcc(*video_props['codec'])  # ffmpeg로부터 추출한 코덱에 따라 변경할 수 있습니다.

# 결과 영상 파일 이름
result_video = '/Users/seonwoo/Desktop/Mosaic_Result/롯데택배_얼굴_3명_Resoult.mp4'

out = cv2.VideoWriter(result_video, fourcc, video_props['fps'], (video_props['width'], video_props['height']))

# 결과 영상을 위한 VideoWriter 객체 생성

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 프레임을 다 읽으면 종료

    results = model(frame)

    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs

        for data in result.boxes.data.tolist():
            confidence = float(data[4])
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            label = int(data[5])

            w, h = xmax - xmin, ymax - ymin  # Calculate width and height of the box

            # 감지된 얼굴 영역에 대한 모자이크 처리
            face = frame[ymin:ymax, xmin:xmax]  # Use ymin, ymax, xmin, xmax instead of y:y+h, x:x+w
            # 모자이크 처리를 위해 축소할 크기를 변경합니다. (더 작은 값으로)
            face = cv2.resize(face, (w // 20, h // 20))  # 여기서 모자이크 강도를 조절합니다.
            # 다시 원래 크기로 확대합니다. 모자이크 블록이 더 크게 보입니다.
            face = cv2.resize(face, (w, h), interpolation=cv2.INTER_AREA)
            frame[ymin:ymax, xmin:xmax] = face  # Replace the face area with the mosaic

            # cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
            text = f'[Mosaic] X: {xmin} Y: {ymin} W: {xmax} H: {ymax}'
            cv2.putText(frame, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


    # 모자이크 처리된 프레임을 결과 영상 파일에 쓰기
    out.write(frame)

    # 결과 프레임 표시
    cv2.imshow('Mosaic Face Detection', frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 작업 완료 후 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()
