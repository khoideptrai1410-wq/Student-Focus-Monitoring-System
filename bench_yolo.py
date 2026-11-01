import sys
import time

import numpy as np
from ultralytics import YOLO

model = YOLO(sys.argv[1])
frame = np.zeros((480, 640, 3), dtype=np.uint8)

for _ in range(5):  # chạy nóng máy
    model.predict(frame, imgsz=416, device="cpu", verbose=False)

start = time.perf_counter()
for _ in range(30):
    model.predict(frame, imgsz=416, device="cpu", verbose=False)
print(f"{(time.perf_counter() - start) / 30 * 1000:.1f} ms/khung hình")