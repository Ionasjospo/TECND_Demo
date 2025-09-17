# generate_video.py
import os
import cv2
import numpy as np
from pathlib import Path
rng = np.random.default_rng(42)

W, H   = 960, 540
FPS    = 30
SECS   = 25

# Cinta y ROI (igual que en el detector)
ROI = (220, 80, 520, 380)   # x, y, w, h

# Movimiento
SPEED   = 2                 # más despacio (px/frame)
SPACING = 190               # separación para que entre una por vez en la ROI
RADIUS  = 28
N_COOKS = 10

# Colores (BGR)
BKG   = (245, 245, 245)
BELT  = (210, 210, 210)
BROWN = (60, 80, 150)       # masa
GREEN = (0, 180, 0)         # OK
RED   = (0, 0, 255)         # defecto

out_dir = Path("assets"); out_dir.mkdir(parents=True, exist_ok=True)
path = out_dir / "galletitas.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(path), fourcc, FPS, (W, H))

# Pistas/lanes suaves para que pasen centradas por la ROI
y0 = ROI[1] + ROI[3]//2
ys = [y0 + (i % 2) * 40 - 20 for i in range(N_COOKS)]

# Posiciones iniciales escalonadas a la izquierda
xs = [-(i * SPACING) for i in range(N_COOKS)]

# Tipos iniciales (mayoría verdes, algunas rojas)
types = ["green" if (i % 3) else "red" for i in range(N_COOKS)]

for f in range(FPS * SECS):
    frame = np.full((H, W, 3), BKG, dtype=np.uint8)

    # Cinta y ROI
    belt_y1, belt_y2 = H//2 - 100, H//2 + 100
    cv2.rectangle(frame, (0, belt_y1), (W, belt_y2), BELT, -1)
    rx, ry, rw, rh = ROI
    cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (255, 255, 0), 2)

    for i in range(N_COOKS):
        xs[i] += SPEED
        # cuando sale por derecha, vuelve a entrar por izquierda con nuevo tipo
        if xs[i] - RADIUS > W:
            xs[i] = -rng.integers(SPACING//2, SPACING)  # reaparece bien atrás
            types[i] = "green" if rng.random() > 0.35 else "red"

        # base
        cv2.circle(frame, (xs[i], ys[i]), RADIUS, BROWN, -1)
        # glaseado según tipo
        color = GREEN if types[i] == "green" else RED
        cv2.circle(frame, (xs[i], ys[i]), int(RADIUS*0.75), color, -1)

    out.write(frame)

out.release()
print(f"Video generado: {path}")
