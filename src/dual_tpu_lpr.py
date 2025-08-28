import cv2
import numpy as np
import time
import os
import glob
from PIL import Image, ImageDraw, ImageFont
import tflite_runtime.interpreter as tflite

# ===== Dual TPU 모델 로딩 =====
delegate_det = tflite.load_delegate("/usr/lib/aarch64-linux-gnu/libedgetpu.so.1", {"device": "pci:0"})
delegate_ocr = tflite.load_delegate("/usr/lib/aarch64-linux-gnu/libedgetpu.so.1", {"device": "pci:0"})

det_interpreter = tflite.Interpreter(
    model_path="models/EfficientDet_L0_numplate_full_int8_edgetpu.tflite",
    experimental_delegates=[delegate_det]
)
det_interpreter.allocate_tensors()

lpr_interpreter = tflite.Interpreter(
    model_path="models/kr_LPRnet_mix_nstn_int8_edgetpu.tflite",
    experimental_delegates=[delegate_ocr]
)
lpr_interpreter.allocate_tensors()

# ===== 폰트 설정 =====
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
font_small = ImageFont.truetype(font_path, 32)
font_big = ImageFont.truetype(font_path, 128)

# ===== 문자 리스트 =====
char_list = [
    "0","1","2","3","4","5","6","7","8","9",
    "가","나","다","라","마","거","너","더","러","머",
    "버","서","어","저","고","노","도","로","모","보",
    "소","오","조","구","누","두","루","무","부","수",
    "우","주","하","-"," "
]

# ===== 프레임 설정 =====
skip_frames = 10
frame_interval = 1.0 / 60.0

# ===== 버퍼 설정 =====
buffer_index = 0
buffer_dir = f"capture_buffer/buffer_{buffer_index}"
image_files = sorted(glob.glob(os.path.join(buffer_dir, "*.jpg")))
image_index = 0

# ===== 상태 변수 =====
frame_count = 0
cached_frame = None
cached_detections = None
overlay_image = None
overlay_start_time = None
overlay_duration = 10
last_ocr_text = ""

# ===== FPS 계산용 =====
frame_counter = 0
start_time = time.time()
displayed_fps = 0

# ===== 창 설정 =====
cv2.namedWindow("License Plate Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("License Plate Recognition", 800, 800)
cv2.moveWindow("License Plate Recognition", 100, 100)

# ===== 추론 함수들 =====
def run_detection(frame):
    h, w, _ = frame.shape
    img = cv2.resize(frame, (320, 320))[None, ...].astype(np.uint8)
    det_interpreter.set_tensor(det_interpreter.get_input_details()[0]['index'], img)
    det_interpreter.invoke()
    scores = det_interpreter.get_tensor(det_interpreter.get_output_details()[0]['index'])[0]
    boxes  = det_interpreter.get_tensor(det_interpreter.get_output_details()[1]['index'])[0]
    results = []
    for i, score in enumerate(scores):
        if score > 0.9:
            ymin, xmin, ymax, xmax = boxes[i]
            x1, y1 = int(xmin * w), int(ymin * h)
            x2, y2 = int(xmax * w), int(ymax * h)
            results.append((x1, y1, x2, y2))
    return results

def run_lprnet(plate):
    img = cv2.resize(plate, (94, 24))[None, ...].astype(np.uint8)
    lpr_interpreter.set_tensor(lpr_interpreter.get_input_details()[0]['index'], img)
    lpr_interpreter.invoke()
    preds = lpr_interpreter.get_tensor(lpr_interpreter.get_output_details()[0]['index'])
    return np.squeeze(preds, axis=0)

# ===== 메인 루프 =====
while True:
    loop_start = time.time()

    if image_index >= len(image_files):
        buffer_index = (buffer_index + 1) % 4
        buffer_dir = f"capture_buffer/buffer_{buffer_index}"
        image_files = sorted(glob.glob(os.path.join(buffer_dir, "*.jpg")))
        image_index = 0
        if not image_files:
            time.sleep(0.01)
            continue

    frame_path = image_files[image_index]
    frame = cv2.imread(frame_path)
    image_index += 1

    if frame is None or frame.size == 0:
        print(f"[ERROR] 잘못된 frame 읽기: {frame_path}")
        continue

    frame_count += 1
    frame_counter += 1
    now = time.time()

    if now - start_time >= 1.0:
        displayed_fps = frame_counter / (now - start_time)
        frame_counter = 0
        start_time = now

    # Detection
    if frame_count % skip_frames == 0:
        dets = run_detection(frame)
        if dets:
            cached_frame = frame.copy()
            cached_detections = dets
        else:
            cached_detections = None

    display_frame = frame.copy()

    # 박스 표시
    if cached_detections:
        for x1, y1, x2, y2 in cached_detections:
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 오버레이 표시
    if overlay_image is not None and overlay_start_time is not None:
        if now - overlay_start_time < overlay_duration:
            overlay_small = cv2.resize(overlay_image, (320, 320))
            h, w, _ = display_frame.shape
            display_frame[h - 320:h, w - 320:w] = overlay_small

            pil = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            if last_ocr_text:
                try:
                    bbox = font_big.getbbox(last_ocr_text)
                    text_width = bbox[2] - bbox[0]
                except:
                    text_width = draw.textsize(last_ocr_text, font=font_big)[0]
                cx = (pil.width - text_width) // 2
                draw.text((cx, 30), last_ocr_text, font=font_big, fill=(0, 0, 0))
            display_frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        else:
            overlay_image = None
            overlay_start_time = None
            last_ocr_text = ""

    # FPS 표시
    pil = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text((20, 20), f"FPS: {displayed_fps:.2f}", font=font_small, fill=(255, 0, 255))
    display_frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    cv2.imshow("License Plate Recognition", display_frame)
    key = cv2.waitKey(1) & 0xFF

    # OCR 실행
    if key == 32 and cached_frame is not None and cached_detections:
        x1, y1, x2, y2 = cached_detections[0]
        plate = cached_frame[y1:y2, x1:x2]
        if plate.size == 0:
            continue
        preds = run_lprnet(plate)
        indices = np.argmax(preds, axis=1)
        text = ""
        prev = -1
        for idx in indices:
            if idx != prev and idx < len(char_list) and char_list[idx] not in ["-", " "]:
                text += char_list[idx]
            prev = idx

        last_ocr_text = text
        overlay_start_time = time.time()
        overlay_image = cached_frame.copy()
        cv2.rectangle(overlay_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        image_pil = Image.fromarray(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
        draw_overlay = ImageDraw.Draw(image_pil)
        draw_overlay.text((x1, y1 - 20), text, font=font_small, fill=(0, 255, 0))
        overlay_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        cv2.imwrite(f"capture_{int(time.time())}.jpg", cached_frame)

    if key == 27:
        break

    # 60fps 유지
    elapsed = time.time() - loop_start
    if elapsed < frame_interval:
        time.sleep(frame_interval - elapsed)

cv2.destroyAllWindows()
