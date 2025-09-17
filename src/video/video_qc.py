import os, json, cv2, numpy as np, requests
from datetime import datetime, timezone
from dotenv import load_dotenv
from queue import Queue, Empty
from threading import Thread, Event

load_dotenv()

# ---- Config desde .env ----
WEBHOOK_URL   = os.getenv("WEBHOOK_URL_PROD", "").strip()
PRODUCT_BASE  = os.getenv("PRODUCT_BASE", "galletita")
MIN_CONF      = float(os.getenv("MIN_CONFIDENCE", "0.30"))         # umbral razonable
CONF_MULT     = float(os.getenv("CONF_AREA_MULTIPLIER", "25.0"))   # escala de área->conf

# ---- HSV ----
GREEN_LOWER = np.array([35, 80, 50]);  GREEN_UPPER = np.array([85, 255, 255])
RED1_LOWER  = np.array([0, 100, 80]);  RED1_UPPER  = np.array([10, 255, 255])
RED2_LOWER  = np.array([170,100, 80]); RED2_UPPER  = np.array([180,255,255])

KERNEL3 = np.ones((3,3), np.uint8)
KERNEL5 = np.ones((5,5), np.uint8)

# enviar cada N frames (evita saturar)
PER_FRAME_INTERVAL = 3

# ---- sender asíncrono (no bloquea el video) ----
SESSION = requests.Session()
send_q: Queue = Queue(maxsize=200)
stop_ev = Event()

def sender_worker():
    while not stop_ev.is_set():
        try:
            payload = send_q.get(timeout=0.1)
        except Empty:
            continue
        try:
            r = SESSION.post(WEBHOOK_URL, json=payload, timeout=3)
            print("[n8n]", r.status_code, str(r.text)[:120])
        except Exception as e:
            print("Error enviando a n8n:", e)
        finally:
            send_q.task_done()

def find_boxes(hsv, lower, upper, min_area=650):
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

def area_conf(area, denom_area):
    # relación de área * multiplicador configurable
    return float(min(1.0, (area / float(denom_area)) * CONF_MULT))

def run_video(video_path="assets/galletitas.mp4"):
    if not os.path.isfile(video_path):
        print(f"No existe el archivo de video: {video_path}"); return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("No se pudo abrir el video."); return

    # ROI (estación)
    RX, RY, RW, RH = 220, 80, 520, 380
    LEFT_EDGE = RX + 8   # borde izquierdo "útil" para armar/disparar

    # 2 carriles (coincide con tu generador)
    LANES = 2
    lane_h = RH // LANES
    lane_armed = [True] * LANES             # listo para disparar
    product_seq = 1
    frame_no = 0

    # hilo del sender
    th = Thread(target=sender_worker, daemon=True); th.start()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame_no += 1

            h, w = frame.shape[:2]
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # detectar
            green_boxes = find_boxes(hsv, GREEN_LOWER, GREEN_UPPER)
            red_boxes   = find_boxes(hsv, RED1_LOWER, RED1_UPPER) + \
                          find_boxes(hsv, RED2_LOWER, RED2_UPPER)

            # dibujar ROI
            cv2.rectangle(frame, (RX,RY), (RX+RW,RY+RH), (255,255,0), 2)

            # filtrar a ROI y seleccionar "mejor" blob por lane (mayor área)
            best = [None]*LANES
            denom_area = RW * RH
            for (x,y,bw,bh,area) in red_boxes:
                cx, cy = x + bw//2, y + bh//2
                if not (RX <= cx <= RX+RW and RY <= cy <= RY+RH):  # fuera de ROI
                    continue
                lane = int((cy - RY) / lane_h)
                lane = max(0, min(LANES-1, lane))
                if best[lane] is None or area > best[lane][-1]:
                    best[lane] = (x,y,bw,bh,area,cx,cy)

            # dibujar verdes
            for (x,y,bw,bh,_) in green_boxes:
                cv2.rectangle(frame,(x,y),(x+bw,y+bh),(0,255,0),2)
                cv2.putText(frame,"GOOD",(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

            # por lane: mostrar y disparar 1 vez cuando entra por la izquierda
            total_def = 0
            for lane in range(LANES):
                if best[lane] is None: 
                    continue
                x,y,bw,bh,area,cx,cy = best[lane]
                conf = area_conf(area, denom_area)
                total_def += 1
                # pintar rojo
                cv2.rectangle(frame,(x,y),(x+bw,y+bh),(0,0,255),2)
                cv2.putText(frame,f"DEFECT {conf:.2f}",(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

                # re-armar cuando aparece algo a la izquierda de la ROI
                if cx < LEFT_EDGE - 12:
                    lane_armed[lane] = True

                # disparo: cruza a la ROI desde la izquierda y está armado
                should_fire = (lane_armed[lane] and cx >= LEFT_EDGE and
                               conf >= MIN_CONF and (frame_no % PER_FRAME_INTERVAL == 0))
                if should_fire:
                    product_id = f"{PRODUCT_BASE}-{product_seq:04d}"
                    payload = {
                        "product_id":  product_id,
                        "defect_type": "color_red",
                        "confidence":  round(conf, 3),
                        "is_defect":   True,
                        "timestamp":   datetime.now(timezone.utc).isoformat()
                    }
                    print("POST →", json.dumps(payload))
                    product_seq += 1
                    lane_armed[lane] = False     # evita múltiples POST por la misma galleta
                    try:
                        send_q.put_nowait(payload)  # asíncrono
                    except:
                        print("send_q llena: descartado para mantener FPS")

            # overlay
            cv2.putText(frame, f"Defects:{total_def}", (10,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            cv2.imshow("QC Vision (video) - q=quit", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
    finally:
        stop_ev.set(); th.join(timeout=1.0)
        cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    run_video()
