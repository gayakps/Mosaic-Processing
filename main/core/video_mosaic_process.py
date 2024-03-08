import asyncio
import datetime
import cv2
from ultralytics import YOLO

from main.core.infra import VideoSource
from main.core.web.mosaic_process_to_web import app

# 모델과 비디오 소스 설정
model = YOLO('../yolo/yolov8n-face.pt')


async def process_video():

    frame_size = 0
    scale_factor = 0.7  # 해상도를 70%로 설정 (즉, 30% 감소)
    start = datetime.datetime.now()

    total_frames = int(VideoSource.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    try:
        while True:

            ret, frame = await asyncio.to_thread(VideoSource.cap.read)

            frame_size += 1
            print(f'NOW FRAME : {frame_size}')
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

                    w, h = xmax - xmin, ymax - ymin  # Calculate width and height of the box

                    print(f'W {w} H {h}')

                    # 감지된 얼굴 영역에 대한 모자이크 처리
                    face = frame[ymin:ymax, xmin:xmax]
                    face = cv2.resize(face, (w // 10, h // 10))
                    face = cv2.resize(face, (w, h), interpolation=cv2.INTER_AREA)
                    frame[ymin:ymax, xmin:xmax] = face
                    text = f'Mosaic Target - {mosaic_users}'
                    cv2.putText(frame, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                                cv2.LINE_AA)
                    # Optional: Add bounding box or text overlay on frame

            # 모자이크 처리된 프레임을 결과 영상 파일에 쓰기

            #http://116.46.219.186:6840/video
            #http://127.0.0.1:6840/video

            # 결과 프레임 표시
            end = datetime.datetime.now()
            total = (end - start).total_seconds()
            print(f'Process 1 frame: {total * 1000:.0f} milliseconds')

            now_time = int(total)

            progress = (frame_size / total_frames) * 100

            text = f'Processing {frame_size}/{total_frames} frames ({progress:.2f}%) | Time {now_time} sec Video Length : {VideoSource.duration}'
            cv2.putText(frame, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            VideoSource.out.write(frame)

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
