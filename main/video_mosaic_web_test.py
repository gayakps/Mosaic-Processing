import asyncio
import datetime
import queue
import threading
import time

from quart import Quart, Response
import cv2
from quart.wrappers import request
from ultralytics import YOLO

from main.ffmpeg_video import get_video_properties

app = Quart(__name__)

# 모델과 비디오 소스 설정
model = YOLO('yolov8n-face.pt')

source_file_name = 'Test2.mp4'
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
global_frame_lock = threading.Lock()  # global_frame에 대한 접근을 동기화하기 위한 Lock

global_frame = None


async def process_video():
    global current_frame, global_frame
    frame_size = 0
    scale_factor = 0.7  # 해상도를 70%로 설정 (즉, 30% 감소)
    start = datetime.datetime.now()

    try:
        while True:

            ret, frame = await asyncio.to_thread(cap.read)

            frame_size += 1
            print(f'NOW FRAME : {frame_size}')
            if not ret or frame is None:
                print('루프가 종료 되었습니다')
                break  # 프레임을 다 읽으면 종료

            frame = cv2.resize(frame, (int(width), int(height)))

            # 모델을 프레임에 적용하여 결과를 얻습니다.
            results = await asyncio.to_thread(model, frame)

            # 감지된 결과에 대한 모자이크 처리
            for result in results:
                boxes = result.boxes  # Boxes object for bounding box outputs
                for data in result.boxes.data.tolist():
                    confidence = float(data[4])
                    xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
                    label = int(data[5])

                    w, h = xmax - xmin, ymax - ymin  # Calculate width and height of the box

                    # 감지된 얼굴 영역에 대한 모자이크 처리
                    face = frame[ymin:ymax, xmin:xmax]
                    face = cv2.resize(face, (w // 20, h // 20))
                    face = cv2.resize(face, (w, h), interpolation=cv2.INTER_AREA)
                    frame[ymin:ymax, xmin:xmax] = face
                    text = f'[Mosaic] X: {xmin} Y: {ymin} W: {xmax} H: {ymax}'
                    cv2.putText(frame, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    # Optional: Add bounding box or text overlay on frame

            # 모자이크 처리된 프레임을 결과 영상 파일에 쓰기
            out.write(frame)

            # 결과 프레임 표시
            end = datetime.datetime.now()
            total = (end - start).total_seconds()
            print(f'Process 1 frame: {total * 1000:.0f} milliseconds')

            text = f'현재 작업중인 시간 {total*1000}/{duration} 남은 시간 {duration-(total*1000)}'
            cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            with global_frame_lock:
                global_frame = frame.copy()

            async with frame_lock:
                current_frame = frame

    except asyncio.CancelledError as e:
        print(f'Error: {e}')
    finally:
        # 자원 정리 코드
        cap.release()
        out.release()  # 비디오 파일 작성에 사용되는 경우
        print('destroyed Resource')

    print('메서드가 종료 되었습니다')




async def generate_frames():
    global current_frame
    while True:
        async with frame_lock:
            if current_frame is not None:
                ret, buffer = cv2.imencode('.jpg', current_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        await asyncio.sleep(0)  # 다른 작업으로 제어를 넘김

@app.route('/video')
async def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def display_video_frames():
    global global_frame
    try:
        while True:
            with global_frame_lock:
                frame_to_show = global_frame.copy() if global_frame is not None else None

            if frame_to_show is not None:
                cv2.imshow('Mosaic Face Detection', frame_to_show)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q'를 누르면 루프 종료
                    break
            else:
                print('NONE 입니다')
                time.sleep(0.1)
    except Exception as e:
        print(f'Error: {e}')
    cv2.destroyAllWindows()

async def shutdown_app():
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]
    await asyncio.gather(*tasks, return_exceptions=True)
    await app.shutdown()

loop = asyncio.get_event_loop()
async def main():

    server = asyncio.create_task(app.run_task(port=6840))
    video = asyncio.create_task(process_video())

    # display_thread = threading.Thread(target=display_video_frames)
    # display_thread.start() 동작안됨

    # process_video()가 완료될 때까지 기다립니다.
    await video

    # process_video()가 완료되면 서버를 종료합니다.
    server.cancel()
    try:
        await server
    except asyncio.CancelledError:
        # 서버 종료 시 발생하는 CancelledError 예외를 처리합니다.
        print('Server closed')  # 서버가 종료되면 메시지를 출력합니다.

    # 비디오 표시 스레드를 종료합니다.
    display_thread.join()

if __name__ == '__main__':
    asyncio.run(main())

