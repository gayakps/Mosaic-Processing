import asyncio
import os

import cv2

from core.config import option
from core.util.ffmpeg_video import get_video_properties

before_frame = None
frame_lock = asyncio.Lock()  # 프레임 접근을 위한 비동기 락




