import asyncio
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

# 전역 변수로 최신 프레임 저장
current_frame = None
frame_lock = asyncio.Lock()  # 프레임 접근을 위한 비동기 락

async def process_video():
    global current_frame
    while True:
        ret, frame = await asyncio.to_thread(cap.read)
        if not ret:
            async with frame_lock:
                current_frame = None  # No more frames to process

            print('모든 서버가 종료 되었습니다 종료 대기')


            # Clean up resources
            cap.release()
            cv2.destroyAllWindows()




            # 모든 작업이 완료되었으므로 이벤트 루프를 종료합니다.
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            list(map(lambda task: task.cancel(), tasks))  # 모든 태스크를 취소합니다.
            await asyncio.gather(*tasks, return_exceptions=True)  # 모든 태스크가 취소될 때까지 기다립니다.
            loop.stop()  # 이벤트 루프를 중지합니다.

            print('루프가 종료 되었습니다')


            # Stop the web server
            await request.app.shutdown()  # 서버 종료를 위해 await 사용

            print('모든 서버가 종료 되었습니다')


            break  # 프레임을 다 읽으면 종료

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

                # Optional: Add bounding box or text overlay on frame

        async with frame_lock:
            current_frame = frame



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

loop = asyncio.get_event_loop()
def run_app():
    task = loop.create_task(process_video())  # Create a background task
    loop.run_until_complete(app.run_task(port=6840))  # Run Quart app as a task
    try:
        loop.run_forever()
    finally:
        loop.close()

if __name__ == '__main__':
    run_app()
