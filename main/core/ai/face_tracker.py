from deep_sort_realtime.deepsort_tracker import DeepSort

enable = False

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


advanced_tracker_enable = True # 강화된 Tracker 만일 YOLO 에서 감지된 얼굴이 없거나 이전 감지된 개수보다 '작을때' 해당 기능이 True 라면 Tracking 을 진행합니다

CONFIDENCE_THRESHOLD = 0.6
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
