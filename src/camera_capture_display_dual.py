import cv2
import numpy as np
import os
import time
from picamera2 import Picamera2
from libcamera import Transform

# ===== 설정 =====
BUFFER_BASE = "./capture_buffer"
NUM_BUFFERS = 4
FRAME_WIDTH, FRAME_HEIGHT = 800, 480
CAPTURE_FPS = 60
VIDEO_DURATION = 0.5  # 0.5초 동안 저장
FRAMES_PER_VIDEO = int(CAPTURE_FPS * VIDEO_DURATION)

CURRENT_BUFFER_FILE = "current_buffer.txt"  # 현재 버퍼 인덱스 기록용 파일

# ===== 버퍼 디렉토리 생성 =====
for i in range(NUM_BUFFERS):
    os.makedirs(os.path.join(BUFFER_BASE, f"buffer_{i}"), exist_ok=True)

# ===== PiCamera2 설정 =====
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
    controls={"FrameRate": CAPTURE_FPS},
    transform=Transform(hflip=0, vflip=0)
)
picam2.configure(config)
picam2.start()

print("[INFO] 4-버퍼 이미지 캡처 시작... (Ctrl+C로 중단)")

current_buffer = 0

try:
    while True:
        buffer_dir = os.path.join(BUFFER_BASE, f"buffer_{current_buffer}")
        
        # 기존 이미지 삭제
        for f in os.listdir(buffer_dir):
            os.remove(os.path.join(buffer_dir, f))

        # 현재 버퍼 기록
        with open(CURRENT_BUFFER_FILE, "w") as f:
            f.write(str(current_buffer))

        start_time = time.time()
        frame_count = 0

        for frame_idx in range(FRAMES_PER_VIDEO):
            
            frame_bgr = picam2.capture_array()

            # 파일 저장
            filename = os.path.join(buffer_dir, f"{frame_idx:04d}.jpg")
            cv2.imwrite(filename, frame_bgr)

            frame_count += 1

        elapsed = time.time() - start_time
        fps = frame_count / elapsed
        print(f"[DEBUG] buffer_{current_buffer} 저장 FPS: {fps:.2f} ({frame_count}장)")

        # 다음 버퍼로 순환
        current_buffer = (current_buffer + 1) % NUM_BUFFERS

except KeyboardInterrupt:
    print("\n[INFO] 이미지 캡처 종료됨")
