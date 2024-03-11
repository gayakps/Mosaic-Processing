import asyncio
import datetime
import cv2
import numpy as np
from deep_sort_realtime.deep_sort import tracker
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

from main.core.ai import Tracker
from main.core.ai.Tracker import GREEN, WHITE
from main.core.ai.video_similarity_checker import compare_frames, add_text_and_line
from main.core.infra import VideoSource
from main.core.web.mosaic_process_to_web import app

# 모델과 비디오 소스 설정
model = YOLO('../yolo/yolov8n-face.pt')

before_number_of_face = 0  # 이전 프레임에서 추출된 얼굴 개수


async def process_video():
    global before_number_of_face  # 얼굴 개수 전역 변수 설정
    frame_size = 0
    scale_factor = 0.7  # 해상도를 70%로 설정 (즉, 30% 감소)
    start = datetime.datetime.now()

    total_frames = int(VideoSource.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    try:
        while True:

            now_frame_start = datetime.datetime.now()

            tracking_target_results = []

            ret, frame = await asyncio.to_thread(VideoSource.cap.read)

            frame_size += 1

            if not ret or frame is None:
                print('루프가 종료 되었습니다')
                break  # 프레임을 다 읽으면 종료

            if frame is not None and VideoSource.width > 0 and VideoSource.height > 0:
                frame = cv2.resize(frame, (int(VideoSource.width), int(VideoSource.height)))
            else:
                print('Invalid frame or target dimensions')
                continue  # 다음 프레임으로 건너뛰기

            frame = cv2.resize(frame, (int(VideoSource.width), int(VideoSource.height)))
            # 모델을 프레임에 적용하여 결과를 얻습니다.
            results = await asyncio.to_thread(model, frame)

            mosaic_users = 0

            # 감지된 결과에 대한 모자이크 처리
            for result in results:
                boxes = result.boxes  # Boxes object for bounding box output# s

                for data in result.boxes.data.tolist():
                    mosaic_users = mosaic_users + 1
                    confidence = float(data[4])
                    xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
                    label = int(data[5])

                    if Tracker.advanced_tracker_enable is True:  # 이전 얼굴보다 현재 얼굴이 작거나 현재 추출된 얼굴 개수가 0 이라면
                        tracking_target_results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, label])

                    if Tracker.enable is True:
                        tracking_target_results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, label])
                    else:
                        w, h = xmax - xmin, ymax - ymin  # Calculate width and height of the box
                        # 감지된 얼굴 영역에 대한 모자이크 처리
                        face = frame[ymin:ymax, xmin:xmax]
                        face = cv2.resize(face, (w // 10, h // 10))
                        face = cv2.resize(face, (w, h), interpolation=cv2.INTER_AREA)
                        frame[ymin:ymax, xmin:xmax] = face
                        text = f'Mosaic Target - {mosaic_users}'
                        cv2.putText(frame, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                                    cv2.LINE_AA)

            now_number_of_face = mosaic_users

            enable_try_advanced_tracker = now_number_of_face < before_number_of_face or now_number_of_face == 0 and Tracker.enable is False  ## Tracker 가 False 일 때 만 사용가능

            print(f'Before Face Amount : {before_number_of_face} Now : {now_number_of_face} 강화된 Tracking : {enable_try_advanced_tracker}')

            similarity = await compare_frames(VideoSource.current_frame, frame, use_gray=True)

            if similarity is not None:
                print(f'[두 이미지간의 유사도 {similarity:.2f}]')

                if similarity < 0.55:
                    # 두 이미지를 옆으로 결합
                    combined_frame = await add_text_and_line(VideoSource.current_frame, frame)
                    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S_%f")[:-3]  # 마지막 3개 문자를 제거하여 밀리초까지 포함
                    # 파일 경로와 이름을 설정할 때 현재 날짜와 시간을 포함시킵니다.
                    filename = f'/Users/seonwoo/Desktop/유사도_불일치/유사도_{similarity:.2f}_{current_datetime}.jpg'
                    cv2.imwrite(filename, combined_frame)

            before_number_of_face = now_number_of_face
            # 모자이크 처리된 프레임을 결과 영상 파일에 쓰기

            if Tracker.enable is True or enable_try_advanced_tracker is True:

                print('Enabling Try Advanced 작업을 시작합니다')

                tracking_results = Tracker.tracker.update_tracks(tracking_target_results, frame=frame)

                for result in tracking_results:

                    if not result.is_confirmed():
                        continue

                    track_id = result.track_id
                    ltrb = result.to_ltrb()

                    xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

                    if ymin < 0:
                        ymin = 0

                    if xmin < 0:
                        xmin = 0

                    # 모자이크를 적용할 영역 (xmin, ymin, xmax, ymax)
                    roi = frame[ymin:ymax, xmin:xmax]

                    # 모자이크 처리할 때, 각 구역의 크기를 정합니다. 더 크게 설정할수록 더 크게 픽셀화됩니다.
                    mosaic_size = 15  # 처리 속도를 높이기 위해 모자이크 크기를 증가

                    # 벡터화된 연산을 사용한 모자이크 처리
                    # roi를 축소하고 다시 확대하여 모자이크 효과를 적용합니다.
                    h, w = roi.shape[:2]

                    w_mosaic_after = w // mosaic_size
                    h_mosaic_after = h // mosaic_size

                    if w_mosaic_after <= 0:
                        w_mosaic_after = 1
                    if h_mosaic_after <= 0:
                        h_mosaic_after = 1

                    print(
                        f'Xmin : {xmin}, Ymin : {ymin} Xmax : {xmax}, Ymax : {ymax} h : {h} w : {w} Result : {w_mosaic_after} , {h_mosaic_after}')

                    roi_small = cv2.resize(roi, (w_mosaic_after, h_mosaic_after), interpolation=cv2.INTER_LINEAR)
                    roi_mosaic = cv2.resize(roi_small, (w, h), interpolation=cv2.INTER_NEAREST)

                    # 원래 이미지에 모자이크 처리된 영역을 다시 삽입합니다.
                    frame[ymin:ymax, xmin:xmax] = roi_mosaic

                    # 사각형과 텍스트를 그립니다.
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(frame, str(f'Tracking ID : {track_id}'), (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 255), 2)

            # 결과 프레임 표시
            end = datetime.datetime.now()
            total = (end - start).total_seconds()

            now_time = int(total)

            progress = (frame_size / total_frames) * 100

            text = f'Processing {frame_size}/{total_frames} frames ({progress:.2f}%) | Time {now_time} sec Video Length : {VideoSource.duration}'
            cv2.putText(frame, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            VideoSource.out.write(frame)

            now_frame_end = datetime.datetime.now()
            now_frame_total = (now_frame_end - now_frame_start).microseconds

            print(f'#### 1 Frame Process: {now_frame_total / 1000:.0f} ms #### Total : {total:.1f} sec')

            async with VideoSource.frame_lock:
                VideoSource.current_frame = frame


    except asyncio.CancelledError as e:
        print(f'Error: {e}')
    finally:
        # 자원 정리 코드
        VideoSource.cap.release()
        VideoSource.out.release()  # 비디오 파일 작성에 사용되는 경우
        print('destroyed Resource')

    print('메서드가 종료 되었습니다')


loop = asyncio.get_event_loop()


async def main():
    server = asyncio.create_task(app.run_task(host='0.0.0.0', port=6840))
    video = asyncio.create_task(process_video())

    await video

    # process_video()가 완료되면 서버를 종료합니다.
    server.cancel()
    try:
        await server
    except asyncio.CancelledError:
        # 서버 종료 시 발생하는 CancelledError 예외를 처리합니다.
        print('Server closed')  # 서버가 종료되면 메시지를 출력합니다.

    # 비디오 표시 스레드를 종료합니다.


if __name__ == '__main__':
    asyncio.run(main())
