# cam_qc.py
import os, time, json, base64
import cv2
import numpy as np
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").strip()
PRODUCT_ID   = os.getenv("PRODUCT_ID", "demo_product")
MIN_CONF     = float(os.getenv("MIN_CONFIDENCE", "0.6"))

# Control de disparos
COOLDOWN_SECONDS = 5.0     # tiempo mínimo entre eventos
CONFIRM_FRAMES   = 5       # frames consecutivos con defecto para confirmar

# ---- HSV ranges (ajustables según luz/objetos) ----
GREEN_LOWER = np.array([35, 80, 50])
GREEN_UPPER = np.array([85, 255, 255])

RED1_LOWER  = np.array([0, 100, 80])
RED1_UPPER  = np.array([10, 255, 255])
RED2_LOWER  = np.array([170, 100, 80])
RED2_UPPER  = np.array([180, 255, 255])

KERNEL3 = np.ones((3,3), np.uint8)
KERNEL5 = np.ones((5,5), np.uint8)

def find_boxes(hsv, lower, upper, min_area=800):
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

def confidence_from_area(area, frame_w, frame_h):
    rel = area / float(frame_w * frame_h)
    return float(min(1.0, rel * 10.0))

def send_to_n8n(payload: dict):
    if not WEBHOOK_URL:
        print("ERROR: WEBHOOK_URL vacío en .env")
        return
    try:
        r = requests.post(WEBHOOK_URL, json=payload, timeout=5)
        print("[n8n]", r.status_code, r.text[:200])
    except Exception as e:
        print("Error enviando a n8n:", e)

def run_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la webcam.")
        return

    defect_frames = 0
    last_sent_ts  = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        green_boxes = find_boxes(hsv, GREEN_LOWER, GREEN_UPPER)
        red_boxes   = find_boxes(hsv, RED1_LOWER, RED1_UPPER) + find_boxes(hsv, RED2_LOWER, RED2_UPPER)

        # Dibujar
        for (x,y,bw,bh,_) in green_boxes:
            cv2.rectangle(frame,(x,y),(x+bw,y+bh),(0,255,0),2)
            cv2.putText(frame,"GOOD",(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        max_conf = 0.0
        for (x,y,bw,bh,area) in red_boxes:
            cv2.rectangle(frame,(x,y),(x+bw,y+bh),(0,0,255),2)
            cv2.putText(frame,"DEFECT",(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
            max_conf = max(max_conf, confidence_from_area(area, w, h))

        good_count   = len(green_boxes)
        defect_count = len(red_boxes)

        cv2.putText(frame, f"Good:{good_count}  Defects:{defect_count}",
                    (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        has_defect = defect_count > 0 and max_conf >= MIN_CONF
        now = time.time()

        # Confirmación por N frames + cooldown
        defect_frames = defect_frames + 1 if has_defect else 0
        ready     = (now - last_sent_ts) >= COOLDOWN_SECONDS
        confirmed = defect_frames >= CONFIRM_FRAMES

        if confirmed and ready:
            payload = {
                "product_id":  PRODUCT_ID,
                "defect_type": "color_red",
                "confidence":  round(max_conf, 3),
                "is_defect":   True,
                "timestamp":   datetime.now(timezone.utc).isoformat()
            }
            print("Enviando a n8n:", json.dumps(payload))
            send_to_n8n(payload)
            last_sent_ts  = now
            defect_frames = 0

        cv2.imshow("QC Vision (webcam) - q=quit", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam()
