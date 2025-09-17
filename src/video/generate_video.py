# generate_video.py
import os
import cv2
import numpy as np
from pathlib import Path
rng = np.random.default_rng(42)

W, H = 960, 540
FPS = 30
SECONDS = 20
N_COOKIES = 7
RADIUS = 28

# Colores en BGR
BKG = (245, 245, 245)
BELT = (210, 210, 210)
OK_GREEN = (0, 180, 0)       # VERDE (OK)
DEFECT_RED = (0, 0, 255)     # ROJO (DEFECTO)
BROWN = (60, 80, 150)        # “masa” marrón (neutro)

out_dir = Path("assets"); out_dir.mkdir(parents=True, exist_ok=True)
video_path = out_dir / "galletitas.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(video_path), fourcc, FPS, (W, H))

# Posiciones y tipos
ys = [H//2 - 70 + 35*(i % 4 - 2) for i in range(N_COOKIES)]
xs = [-(i*140) for i in range(N_COOKIES)]
speeds = rng.integers(4, 7, size=N_COOKIES)

# Asignamos tipos cíclicos: verde/ok y rojo/defecto (al menos 2 rojas siempre)
types = ["green" if i % 3 else "red" for i in range(N_COOKIES)]
if types.count("red") < 2:
    types[0] = types[2] = "red"

for f in range(FPS*SECONDS):
    frame = np.full((H, W, 3), BKG, dtype=np.uint8)

    # Cinta transportadora
    belt_y1, belt_y2 = H//2 - 100, H//2 + 100
    cv2.rectangle(frame, (0, belt_y1), (W, belt_y2), BELT, -1)

    for i in range(N_COOKIES):
        xs[i] += int(speeds[i])
        if xs[i] - RADIUS > W:
            xs[i] = -rng.integers(40, 160)   # reingresa por la izquierda
            types[i] = "green" if rng.random() > 0.35 else "red"  # 35% defectuosas

        # Disco base marrón
        cv2.circle(frame, (xs[i], ys[i]), RADIUS, BROWN, -1)

        # Glaseado según tipo
        color = OK_GREEN if types[i] == "green" else DEFECT_RED
        cv2.circle(frame, (xs[i], ys[i]), int(RADIUS*0.75), color, -1)

    out.write(frame)

out.release()
print(f"Video generado: {video_path}")
