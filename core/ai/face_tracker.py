from deep_sort_realtime.deepsort_tracker import DeepSort

advanced_tracker_enable = True # 강화된 Tracker 만일 YOLO 에서 감지된 얼굴이 없거나 이전 감지된 개수보다 '작을때' 해당 기능이 True 라면 Tracking 을 진행합니다


# 얼굴 추적을 위한 DeepSort 인스턴스 생성
tracker = DeepSort(
    max_iou_distance=0.3,   # 얼굴의 크기가 작기 때문에 IOU 거리를 줄입니다.
    max_age=1,             # 추적 대상이 오래 보이지 않아도 추적을 유지합니다.
    n_init=1,               # 초기화 프레임 수; 얼굴이 잠깐 보여도 빠르게 추적을 시작합니다.
    nms_max_overlap=0.5,    # 비최대 억제; 얼굴 감지에서 중복을 줄입니다.
    max_cosine_distance=0.3, # 코사인 거리; 얼굴의 외모 변화에 좀 더 유연하게 대응합니다.
    nn_budget=None,         # 네트워크 메모리; 얼굴 추적에서는 기본 설정을 유지합니다.
    gating_only_position=True, # 위치와 모양을 모두 고려하여 추적 결정을 합니다.
    embedder="mobilenet",   # 얼굴 추적을 위해 사용할 수 있는 경량 모델 선택.
    half=True,              # 계산 속도를 위해 반정밀도 사용.
    bgr=True,               # OpenCV 기반의 이미지 포맷을 사용합니다.
    embedder_gpu=True       # 임베딩 계산에 GPU를 사용합니다.
)



CONFIDENCE_THRESHOLD = 0.6
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)


def calculate_iou(box1, box2):
    """
    두 사각형(얼굴)의 IOU를 계산합니다.
    :param box1: 첫 번째 사각형의 좌표 [xmin, ymin, xmax, ymax]
    :param box2: 두 번째 사각형의 좌표 [xmin, ymin, xmax, ymax]
    :return: 두 사각형의 IOU (0 ~ 1 사이의 값)
    """
    # 각 사각형의 (x, y) 좌표
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # 교차 영역의 (x, y) 좌표
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)

    # 교차 영역의 넓이
    inter_area = max(xi_max - xi_min, 0) * max(yi_max - yi_min, 0)

    # 각 사각형의 넓이
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # 합집합 영역의 넓이
    union_area = box1_area + box2_area - inter_area

    # IOU 계산
    iou = inter_area / union_area
    return iou

def track_object(cv2, before_frame_face_coordinate, now_frame_face_coordinate, frame):
    # 추적을 업데이트합니다.
    # tracker.predict()  # 추적기의 상태를 예측(업데이트)

    tracking_results = tracker.update_tracks(before_frame_face_coordinate, frame=frame) # 이전 프레임으로 인식

    print(f'{tracking_results} tracking results')

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

        # 현재 처리하려는 얼굴의 좌표
        current_face = [xmin, ymin, xmax - xmin, ymax - ymin]

        # 현재 처리하려는 얼굴이 이미 처리되었는지 확인
        already_processed = any(
            all(abs(current_face[i] - existing_face[i]) < 30 for i in range(4))  # 임계값 설정
            for existing_face, _, _ in now_frame_face_coordinate  # existing_face는 좌표만 사용
        )

        if already_processed:
            # 이미 처리된 얼굴은 건너뜁니다.
            cv2.putText(frame, str(f'Already Tracking ID : {track_id}'), (xmin + 5, ymin - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)
            print(f'Face at {current_face} is skipped as it is already processed.')
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

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
            # now_frame_face_coordinate.append([current_face, result.confidence, track_id])
            # 처리되지 않은 얼굴에 대한 정보를 출력합니다.
            print(f'Mosaic is applied at {current_face}.')