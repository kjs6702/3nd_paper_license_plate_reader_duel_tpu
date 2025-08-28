# A Study on the Performance Improvement of a Real-Time Potable Vehicle License Plate Recognition System Using Dual Edge TPU

> 실시간 차량 번호판 인식을 위한 Edge AI 시스템  
> Raspberry Pi 5 + Coral **PCIe Dual Edge TPU** 기반 **EfficientDet_Lite0 + LPRNet** 구성

---

##  Description

본 프로젝트는 **Raspberry Pi 5**와 **Coral Dual Edge TPU (PCIe)** 기반으로 동작하는 **실시간 자동차 번호판 인식 시스템**입니다.  
기존 YOLOv8n 기반 시스템을 개선하여, **EfficientDet_Lite0 모델**을 사용해 탐지 성능을 향상시키고,  
전체 파이프라인 속도 및 경량화를 모두 달성하였습니다.  

전체 파이프라인은 네트워크 연결 없이도 작동하는 **완전한 On-Device AI 시스템**입니다.

---

##  System Pipeline

1. **EfficientDet-Lite0** (TPU 실행): 차량 번호판 탐지  
2. 이미지 Crop 및 Resize (94×24×3)  
3. 스페이스바 입력 시 **LPRNet** OCR 실행 (CPU 실행)  
4. 결과 문자열 Overlay → 10초간 이미지 상단에 표시  
![pipeline](./results/pipeline.png)

---

##  Project Structure

```
src/                   # 소스코드
results/               # 결과 이미지 저장
requirements.txt       # 의존 패키지 리스트
README.md              # 프로젝트 설명 문서
```

---

##  Environment

| 항목          | 사양 |
|---------------|------|
| Device        | Raspberry Pi 5 (8GB) |
| OS            | Debian 12 (Bookworm) |
| Python(camera_capture.py)        | 3.11 |
| Python(video_display.py)        | 3.9 |
| Accelerator   | Coral PCIe Dual Edge TPU |
| Camera        | Raspberry Pi Camera Module 3 |
| Display       | 7” DSI Touch Display |
| Power         | USB PD (5V 2.4A) / PD 트리거 5V 5A |

![Environment](./results/result.png)

---

##  Model Files

| 모델         | 파일명                              | 실행 위치  |
|--------------|--------------------------------------|-------------|
| EfficientDet_Lite_0 | `efficientdet_lite0_edgetpu.tflite` | Dual Edge TPU |
| LPRNet       | `lprnet_float32.tflite`             | CPU         |

##EfficientDet_Lite0

![efficientDet_Lite0](./results/efficientDet_Lite0.png)

##LPRNet구조

![LPRNet](./results/LPRNet.png)

---

##  Performance

| 항목                | 기존 시스템 (USB Edge TPU) | 개선 시스템 (Dual Edge TPU) |
|---------------------|-----------------------------|-------------------------------|
| YOLOv8n / EfficientDet-Lite0 mAP@50-95 | 0.715 | **0.876** |
| LPRNet OCR CER |  0.0024 | **0.0029** |
| LPRNet OCR Exact Match Accuracy |  0.9834 | **0.9811** |
| Avg FPS       | 18.53 FPS | **45.60 FPS** |
| 전체 모델 용량 | 13.6 MB | **6.6 MB** |
| Power Usage | 최대 11W | 동일 수준 (최대 11W) |



---

##  Dataset

- 출처: [AI Hub](https://aihub.or.kr)  
- 데이터셋: "자동차 차종/연식/번호판 인식용 영상"  
- 사용량:  
  - **EfficientDet_Lite0**: 1000장 (Train: 700, Val: 200, Test: 100)  
  - **LPRNet**: 56,666장 (Train: 39,666, Val: 11,333, Test: 5,667)  

> YOLOv8n으로 탐지 → 번호판 영역 crop → LPRNet 학습

---

##  Key Features

- 실시간 Dual TPU 기반 객체 탐지 + OCR
- 카메라로 프레임 저장 후 Python 3.9 환경에서 영상 재구성
- 효율적인 멀티버퍼링 + 이중 스레드 파이프라인
- 터치 디스플레이를 통한 결과 실시간 출력

---

##  Conclusion

본 시스템은 실시간 Edge AI 성능을 극대화하기 위해,  
**YOLOv8n → EfficientDet_Lite0**, **USB TPU → PCIe Dual Edge TPU**로 전환하여  
FPS를 **약 2.5배 증가**시키고 전체 모델 용량을 **절반 이하로 감소**시켰습니다.  
이는 경량화 및 고속화를 동시에 달성한 **휴대 가능한 번호판 인식기**로서의 가능성을 보여줍니다.

## 기존 파이프라인 fps

![single_result](./results/single_result.png)

## 변경한 파이프라인 fps

![dual_result](./results/dual_result.png)

---

##  License

MIT License  

---

##  Authors

박시형 (Park Si-Hyeong)  
김지섭 (Kim Ji-Seob)  
강인경 (Kang In-Gyeong)  
정영락 (Jeung Young-Rak)  
> Dept. of Computer Software, Dong-Seoul University
