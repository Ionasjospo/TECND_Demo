import os, time, json
import cv2
import numpy as np
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

WEBHOOK_URL = os.getenv("WEBHOOK_URL_PROD", "").strip()
PRODUCT_ID   = os.getenv("PRODUCT_ID", "demo_product")
MIN_CONF     = float(os.getenv("MIN_CONFIDENCE", "0.6"))

# ---- HSV ranges ----
GREEN_LOWER = np.array([35, 80, 50])
GREEN_UPPER = np.array([85, 255, 255])

RED1_LOWER  = np.array([0, 100, 80])
RED1_UPPER  = np.array([10, 255, 255])
RED2_LOWER  = np.array([170, 100, 80])
RED2_UPPER  = np.array([180, 255, 255])

KERNEL3 = np.ones((3,3), np.uint8)
KERNEL5 = np.ones((5,5), np.uint8)

# MODO PER-FRAME: enviar un evento cada N frames mientras haya defecto
PER_FRAME_INTERVAL = 5  # 5 -> ~6 eventos/seg a 30 FPS

def find_boxes(hsv, lower, upper, min_area=450):
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, KERNEL5)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area >= min_area:
            x,y,w,h = cv2.boundingRect(c)
            boxes.append((x,y,w,h,area))
    return boxes

def confidence(area, denom_area):
    rel = area / float(denom_area)
    return float(min(1.0, rel * 10.0))  # simple para demo

def send_to_n8n(payload: dict):
    if not WEBHOOK_URL:
        print("ERROR: WEBHOOK_URL vacío en .env"); return
    try:
        r = requests.post(WEBHOOK_URL, json=payload, timeout=5)
        print("[n8n]", r.status_code, r.text[:200])
    except Exception as e:
        print("Error enviando a n8n:", e)

def run_video(video_path: str = "assets/galletitas.mp4"):
    if not os.path.isfile(video_path):
        print(f"No existe el archivo de video: {video_path}"); return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("No se pudo abrir el video."); return

    # ROI (estación de control) – ajusta a tu video
    use_roi = True
    ROI = (220, 80, 520, 380)  # x, y, w, h

    def in_roi(box, roi):
        x,y,w,h,_ = box
        rx,ry,rw,rh = roi
        return not (x+w < rx or x > rx+rw or y+h < ry or y > ry+rh)

    frame_no = 0  # <-- contador LOCAL

    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop para demo
            continue

        frame_no += 1

        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Detecciones
        green_boxes = find_boxes(hsv, GREEN_LOWER, GREEN_UPPER)
        red_boxes   = find_boxes(hsv, RED1_LOWER, RED1_UPPER) + \
                      find_boxes(hsv, RED2_LOWER, RED2_UPPER)

        # ROI + área de referencia para confianza
        denom_area = w * h
        if use_roi:
            x0,y0,w0,h0 = ROI
            cv2.rectangle(frame,(x0,y0),(x0+w0,y0+h0),(255,255,0),2)
            red_boxes = [b for b in red_boxes if in_roi(b, ROI)]
            denom_area = w0 * h0

        # Pintar
        for (x,y,bw,bh,_) in green_boxes:
            cv2.rectangle(frame,(x,y),(x+bw,y+bh),(0,255,0),2)
            cv2.putText(frame,"GOOD",(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        max_conf = 0.0
        for (x,y,bw,bh,area) in red_boxes:
            cv2.rectangle(frame,(x,y),(x+bw,y+bh),(0,0,255),2)
            cv2.putText(frame,"DEFECT",(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
            max_conf = max(max_conf, confidence(area, denom_area))

        cv2.putText(frame, f"Conf:{max_conf:.2f}", (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"Good:{len(green_boxes)}  Defects:{len(red_boxes)}",
                    (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # --- envío per-frame ---
        has_defect = (len(red_boxes) > 0) and (max_conf >= MIN_CONF)
        if has_defect and (frame_no % PER_FRAME_INTERVAL == 0):
            payload = {
                "product_id":  PRODUCT_ID,
                "defect_type": "color_red",
                "confidence":  round(max_conf, 3),
                "is_defect":   True,
                "timestamp":   datetime.now(timezone.utc).isoformat()
            }
            print("Enviando a n8n:", json.dumps(payload))
            send_to_n8n(payload)

        cv2.imshow("QC Vision (video) - q=quit", frame)
        if (cv2.waitKey(25) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_video()
