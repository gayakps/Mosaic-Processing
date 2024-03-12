import asyncio
import datetime
import cv2
from ultralytics import YOLO

from main.core.ai import face_tracker
from main.core.ai.video_similarity_checker import compare_frames, add_text_and_line
from main.core.infra import VideoSource
from main.core.web.mosaic_process_to_web import app

# 모델과 비디오 소스 설정

model = YOLO('/Users/seonwoo/Desktop/작업/Mosaic-System-Folder/Mosaic-Processing/main/yolo/yolov8n-face.pt')

before_number_of_face = 0  # 이전 프레임에서 추출된 얼굴 개수
before_face_coordinate = []
now_frame_face_coordinate = []


async def process_video():
    global before_number_of_face  # 얼굴 개수 전역 변수 설정
    global before_face_coordinate
    global now_frame_face_coordinate

    frame_size = 0
    start = datetime.datetime.now()

    total_frames = int(VideoSource.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    try:
        while True:

            now_frame_start = datetime.datetime.now()

            # print(f'Before/Now Face Coordinate Size : [ {len(before_face_coordinate)} / {len(now_frame_face_coordinate)} ]')

            if len(now_frame_face_coordinate) > 0:
                before_face_coordinate = now_frame_face_coordinate.copy()
                now_frame_face_coordinate.clear()
                # print(f'(Copy After) Before/Now Face Coordinate Size : [ {len(before_face_coordinate)} / {len(now_frame_face_coordinate)} ]')

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

                    coordinate = [[xmin, ymin, xmax - xmin, ymax - ymin], confidence, label]

                    now_frame_face_coordinate.append(coordinate)

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

            enable_try_advanced_tracker = now_number_of_face < before_number_of_face ## Tracker 가 False 일 때 만 사용가능

            if enable_try_advanced_tracker == True:
                print(f'[Frame : {frame_size}] Before Face Amount : {before_number_of_face} Now : {now_number_of_face} 강화된 Tracking : {enable_try_advanced_tracker}')

            similarity = await compare_frames(VideoSource.before_frame, frame)

            if similarity is not None:

                print(f'[두 이미지간의 유사도 {similarity:.2f}]')

                if similarity <= 0.45:
                    # 두 이미지를 옆으로 결합
                    combined_frame = await add_text_and_line(VideoSource.before_frame, frame)
                    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S_%f")[:-3]  # 마지막 3개 문자를 제거하여 밀리초까지 포함
                    # 파일 경로와 이름을 설정할 때 현재 날짜와 시간을 포함시킵니다.r
                    filename = f'/Users/seonwoo/Desktop/유사도_불일치/유사도_{similarity:.2f}_{current_datetime}.jpg'
                    cv2.imwrite(filename, combined_frame)
                else: # 유사도가 높다는건 비슷한 화면임을 의미함

                    if enable_try_advanced_tracker is True and face_tracker.advanced_tracker_enable: # 강화된 Tracking 이 활성화 되었을 때에

                        ## 이전 프레임과 비교하여 최신 프레임에서 Trakcing Processing 을 해야함

                        print('------- Enabling Try Advanced 작업을 시작합니다 ------')

                        tracking_results = face_tracker.tracker.update_tracks(before_face_coordinate, frame=VideoSource.before_frame)

                        for result in tracking_results:

                            print('Tracking 작업을 시작합니다')

                            if not result.is_confirmed():
                                print('확정되지 않았습니다')
                                continue

                            track_id = result.track_id
                            ltrb = result.to_ltrb()

                            xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

                            if ymin < 0:
                                ymin = 0

                            if xmin < 0:
                                xmin = 0

                            # 현재 처리하려는 얼굴의 좌표
                            current_face = [xmin, ymin, xmax - xmin, ymax - ymin]

                            # 현재 처리하려는 얼굴이 이미 처리되었는지 확인
                            already_processed = any(
                                all(abs(current_face[i] - existing_face[i]) < 10 for i in range(4))  # 임계값 설정
                                for existing_face, _, _ in now_frame_face_coordinate  # existing_face는 좌표만 사용
                            )

                            if already_processed:
                                # 이미 처리된 얼굴은 건너뜁니다.
                                print(f'Face at {current_face} is skipped as it is already processed.')
                                continue
                            else:
                                # 모자이크 처리를 수행합니다.
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

                                roi_small = cv2.resize(roi, (w_mosaic_after, h_mosaic_after),
                                                       interpolation=cv2.INTER_LINEAR)
                                roi_mosaic = cv2.resize(roi_small, (w, h), interpolation=cv2.INTER_NEAREST)

                                # 원래 이미지에 모자이크 처리된 영역을 다시 삽입합니다.
                                frame[ymin:ymax, xmin:xmax] = roi_mosaic

                                # 사각형과 텍스트를 그립니다.
                                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                                cv2.putText(frame, str(f'Tracking ID : {track_id}'), (xmin + 5, ymin - 8),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (0, 0, 255), 2)
                                # 새 얼굴 좌표 추가
                                now_frame_face_coordinate.append([current_face, result.confidence, track_id])
                                # 처리되지 않은 얼굴에 대한 정보를 출력합니다.
                                print(f'Mosaic is applied at {current_face}.')

                        print('----------------------------------')


            before_number_of_face = now_number_of_face

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

            before_face_coordinate.clear() # 값 초기화

            print(f'#### 1 Frame Process: {now_frame_total / 1000:.0f} ms #### Total : {total:.1f} sec')

            async with VideoSource.frame_lock:
                VideoSource.before_frame = frame


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
