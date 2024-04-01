import asyncio

from quart import Response, Quart
import cv2

from core.infra import video_source

app = Quart(__name__)

async def generate_frames():
    while True:
        async with video_source.frame_lock:  # 수정된 라인
            if video_source.before_frame is not None:
                ret, buffer = cv2.imencode('.jpg', video_source.before_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                print('No frame received')
        await asyncio.sleep(0.1)  # 다른 작업으로 제어를 넘김




@app.route('/video')
async def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')