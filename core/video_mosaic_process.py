#!/usr/bin/env python3

import asyncio
import datetime
import json
import os
import sys
import traceback

import cv2

from ultralytics import YOLO

from core.ai import face_tracker
from core.ai.video_similarity_checker import compare_frames, add_text_and_line
from core.amqp.rabbit_mq import rabbit_mq_data_producer_to_java_server
from core.aws import aws_s3
from core.config import option
from core.infra import video_source
from core.util.ffmpeg_video import get_video_properties
from core.web.mosaic_process_to_web import app


# 모델과 비디오 소스 설정

model = YOLO(option.face_model_path)

before_number_of_face = 0  # 이전 프레임에서 추출된 얼굴 개수
before_face_coordinate = []
now_frame_face_coordinate = []

user_id = None # user_id 전달
video_id = None # video_id 전달
video_file_name = None  # video_file_name 전달

async def process_video(user_id, video_id, video_file_name):

    global before_number_of_face, now_time
    global before_face_coordinate
    global now_frame_face_coordinate

    aws_s3.download_file(user_id=user_id, video_id=video_id, file_name=video_file_name) # Blocking until download all files

    # 디렉터리 경로 설정
    directory = option.source_directory

    # 디렉터리 내의 파일 목록 가져오기, mov와 mp4 확장자를 가진 파일만 필터링
    files = [file for file in os.listdir(directory) if file.endswith(('.mov', '.mp4', '.MOV', '.MP4', '.avi', '.AVI', '.MKV', '.mkv'))]

    # 디렉터리 내의 첫 번째 영상 파일 이름 얻기
    # 파일이 없을 경우 None을 반환
    source_file_name = files[0] if files else None

    if source_file_name is None:
        print('No source video Source File Name video')
        return

    print(f'source_file_name: {source_file_name}')

    source_video = f'{directory}/{source_file_name}'

    video_cap = cv2.VideoCapture(source_video)
    video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    size = (int(video_width), int(video_height))

    video_props = get_video_properties(source_video)  # 데이터 추출
    print(video_props)

    print(f"Original video FPS: {fps}")
    print(f"Original video resolution: {video_width}x{video_height}")

    # VideoWriter 설정
    fourcc = cv2.VideoWriter_fourcc(*video_props['codec'])  # ffmpeg로부터 추출한 코덱에 따라 변경할 수 있습니다.
    # fourcc = cv2.VideoWriter.fourcc(*'DIVX')

    # 결과 영상 파일 이름
    result_video = f'{directory}/Result_{source_file_name}'

    # out = cv2.VideoWriter(result_video, fourcc, video_props['fps'], (video_props['width'], video_props['height']))
    out = cv2.VideoWriter(result_video, fourcc, video_props['fps'], size)

    duration = int(video_props['duration'])

    frame_size = 0
    start = datetime.datetime.now()
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    message = json.dumps({'date': str(start), 'type': "VIDEO_INFO", 'data': str(video_props) })
    rabbit_mq_data_producer_to_java_server.sendToJavaServer(message)

    success = True
    state = "Success"

    try:
        while True:

            now_frame_start = datetime.datetime.now()

            print(f'Before/Now Face Coordinate Size : [ {len(before_face_coordinate)} / {len(now_frame_face_coordinate)} ]')

            if len(now_frame_face_coordinate) > 0:
                before_face_coordinate = now_frame_face_coordinate.copy()
                now_frame_face_coordinate.clear()
                print(f'(Copy After) Before/Now Face Coordinate Size : [ {len(before_face_coordinate)} / {len(now_frame_face_coordinate)} ]')

            ret, frame = await asyncio.to_thread(video_cap.read)

            if not ret or frame is None:
                print('루프가 종료 되었습니다')
                break  # 프레임을 다 읽으면 종료

            if frame is not None and video_width > 0 and video_height > 0:
                frame = cv2.resize(frame, (int(video_width), int(video_height)))
            else:
                print('Invalid frame or target dimensions')
                continue  # 다음 프레임으로 건너뛰기

            frame = cv2.resize(frame, (int(video_width), int(video_height)))
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
                    # text = f'Mosaic Target - {mosaic_users}'
                    # cv2.putText(frame, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    #             cv2.LINE_AA)

            now_number_of_face = mosaic_users

            if face_tracker.advanced_tracker_enable is True: # Advanced Tracker 가 활성화 되었을 때

                enable_try_advanced_tracker = now_number_of_face < before_number_of_face  ## Tracker 가 False 일 때 만 사용가능

                print(f'[Frame : {frame_size}] Before Face Amount : {before_number_of_face} Now : {now_number_of_face} 강화된 Tracking : {enable_try_advanced_tracker}')

                similarity = await compare_frames(video_source.before_frame, frame)

                if similarity is not None:

                    print(f'[두 이미지간의 유사도 {similarity:.2f}]')

                    if similarity <= 0.45:
                        # 두 이미지를 옆으로 결합
                        combined_frame = await add_text_and_line(video_source.before_frame, frame)
                        current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S_%f")[
                                           :-3]  # 마지막 3개 문자를 제거하여 밀리초까지 포함
                        # 파일 경로와 이름을 설정할 때 현재 날짜와 시간을 포함시킵니다.r
                        filename = f'/Users/seonwoo/Desktop/유사도_불일치/유사도_{similarity:.2f}_{current_datetime}.jpg'
                        cv2.imwrite(filename, combined_frame)
                    else:  # 높은 유사도를 보이고 있을 때
                        if enable_try_advanced_tracker is True and face_tracker.advanced_tracker_enable is True and len(
                                before_face_coordinate) > 0:  # 강화된 Tracking 이 활성화 되었을 때에
                            print('------- Enabling Try Advanced 작업을 시작합니다 ------')
                            message = json.dumps({'date': str(now_frame_start), 'type': "SIMILARITY_INFO", 'frame': str(frame_size), 'similarity': str(similarity), 'data': '유사도가 높습니다, Tracking System 이 가동됩니다'})
                            rabbit_mq_data_producer_to_java_server.sendToJavaServer(message)

                            face_tracker.track_object(cv2, before_face_coordinate, now_frame_face_coordinate, frame)

            out.write(frame)

            frame_size += 1
            before_number_of_face = now_number_of_face

            # 결과 프레임 표시
            end = datetime.datetime.now()
            total = (end - start).total_seconds()

            now_time = int(total)

            progress = (frame_size / total_frames) * 100

            text = f'Processing {frame_size}/{total_frames} frames ({progress:.2f}%) | Time {now_time} / {duration} Tracking Option : {face_tracker.advanced_tracker_enable}'
            cv2.putText(frame, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

            now_frame_end = datetime.datetime.now()
            now_frame_total = (now_frame_end - now_frame_start).microseconds

            before_face_coordinate.clear() # 값 초기화

            print(f'#### 1 Frame Process: {now_frame_total / 1000:.0f} ms #### Total : {total:.1f} sec')

            async with video_source.frame_lock:
                video_source.before_frame = frame


    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f'Error: {e}\n{error_traceback}')
        success = False
        state = f'{e}\n{error_traceback}'
    finally:
        # 자원 정리 코드

        if success is True:
            video_cap.release()
            out.release()  # 비디오 파일 작성에 사용되는 경우
            aws_s3.upload_file(file_name=result_video, video_id=video_id, user_id=user_id)
            print('destroyed Resource')
        else:
            print('실패했습니다 모든것을 재시도합니다')
            # await main()

        now = datetime.datetime.now()
        message = json.dumps(
            {'date': str(now),
             'type': "UPLOAD_STATE",
             'now_frame': frame_size,
             'elapse_time_sec': now_time,
             'total_frame': str(total_frames),
             'status': success,
             'msg': state}
        )
        rabbit_mq_data_producer_to_java_server.sendToJavaServer(message)

    print('메서드가 종료 되었습니다')


loop = asyncio.get_event_loop()


async def main(user_id, video_id, video_file_name):
    server = asyncio.create_task(app.run_task(host='0.0.0.0', port=6840))
    video = asyncio.create_task(process_video(user_id, video_id, video_file_name))

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
    print('Start System')
    try:
        os.chdir(sys._MEIPASS)
        print(sys._MEIPASS)
    except:
        os.chdir(os.getcwd())

    user_id = sys.argv[1] # user_id 전달
    video_id = sys.argv[2] # video_id 전달 ( 사용자에게 전달받아 제작됨 )
    video_file_name = sys.argv[3] # video_file_name 전달
    print(f'user_id: {user_id}, video_id: {video_id} video_file_name: {video_file_name}')

    asyncio.run(main(user_id, video_id, video_file_name))
