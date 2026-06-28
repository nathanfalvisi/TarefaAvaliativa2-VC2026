"""
AR Inspector  Deteccao com YOLOv8n
====================================
Usa a biblioteca oficial Ultralytics, que baixa o yolov8n.pt automaticamente
na primeira execucao. Detecta 80 objetos COCO: garrafa, copo, caderno/livro,
celular, mochila, laptop, tesoura, pessoa, etc.

Requisitos:
    pip install ultralytics opencv-contrib-python numpy

Executar:
    python ar_yolo_corrigido.py
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime
from ultralytics import YOLO

#  Config 
MODEL_NAME = "yolov8n.pt"   # A Ultralytics baixa automaticamente se nao existir
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CONF_THRESH = 0.40
IOU_THRESH  = 0.45
INPUT_SIZE  = 640

#  Classes COCO que queremos mostrar com informacoes extras 
OBJECT_INFO = {
    "bottle":     {"pt": "Garrafa",    "info": "Recipiente liquido", "color": (0,200,255)},
    "cup":        {"pt": "Copo",       "info": "Utensilio para bebida", "color": (0,220,200)},
    "book":       {"pt": "Livro/Caderno","info": "Material de leitura","color": (80,180,255)},
    "cell phone": {"pt": "Celular",    "info": "Dispositivo movel",  "color": (50,230,50)},
    "backpack":   {"pt": "Mochila",    "info": "Bolsa de transporte","color": (200,80,200)},
    "handbag":    {"pt": "Bolsa",      "info": "Acessorio pessoal",  "color": (220,100,180)},
    "wallet":     {"pt": "Carteira",   "info": "Item de valor",      "color": (50,150,255)},
    "suitcase":   {"pt": "Mala",       "info": "Bagagem",            "color": (160,160,50)},
    "laptop":     {"pt": "Notebook",   "info": "Computador portatil","color": (30,200,30)},
    "keyboard":   {"pt": "Teclado",    "info": "Periferico",         "color": (100,200,100)},
    "mouse":      {"pt": "Mouse",      "info": "Periferico",         "color": (120,220,120)},
    "remote":     {"pt": "Controle",   "info": "Controle remoto",    "color": (180,200,50)},
    "scissors":   {"pt": "Tesoura",    "info": "Objeto cortante ",  "color": (0,80,255)},
    "vase":       {"pt": "Vaso",       "info": "Objeto decorativo",  "color": (150,200,255)},
    "clock":      {"pt": "Relogio",    "info": "Medidor de tempo",   "color": (200,200,100)},
    "chair":      {"pt": "Cadeira",    "info": "Mobiliario",         "color": (160,160,160)},
    "dining table":{"pt":"Mesa",       "info": "Mobiliario",         "color": (140,140,140)},
    "tv":         {"pt": "TV",         "info": "Eletronico",         "color": (0,160,255)},
    "person":     {"pt": "Pessoa",     "info": "Ser humano",         "color": (255,200,0)},
    "cat":        {"pt": "Gato",       "info": "Animal domestico",   "color": (0,200,180)},
    "dog":        {"pt": "Cachorro",   "info": "Animal domestico",   "color": (0,180,200)},
}

# Todas as 80 classes COCO (para decodificar saida do modelo)
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush",
]

MODES = ["INFO", "SIMPLES", "DEBUG"]


#  Inferencia YOLOv8 

class YOLOv8:
    def __init__(self, model_name=MODEL_NAME):
        # Na primeira execucao, o arquivo yolov8n.pt e baixado automaticamente.
        self.model = YOLO(model_name)

    def detect(self, frame, conf_thresh=CONF_THRESH):
        """Retorna lista de {label, conf, bbox} usando a API atual da Ultralytics."""
        results = self.model(
            frame,
            imgsz=INPUT_SIZE,
            conf=conf_thresh,
            iou=IOU_THRESH,
            verbose=False
        )

        detections = []
        result = results[0]

        if result.boxes is None:
            return detections

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            label = result.names.get(cls_id, str(cls_id))
            x = int(max(0, x1))
            y = int(max(0, y1))
            w = int(max(0, x2 - x1))
            h = int(max(0, y2 - y1))

            detections.append({
                "label": label,
                "conf": conf,
                "bbox": (x, y, w, h),
            })

        return detections


#  Desenho / Overlay 

def draw_hud(frame, fps, mode, count, paused):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 42), (18, 18, 18), -1)
    cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)
    pause_s = "   PAUSADO" if paused else ""
    txt = f"AR YOLOv8  |  FPS:{fps:4.1f}  |  {mode}  |  Objetos:{count}{pause_s}"
    cv2.putText(frame, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (210, 235, 255), 1, cv2.LINE_AA)
    mc = {"INFO": (80,255,80), "SIMPLES": (255,200,50), "DEBUG": (255,100,255)}
    cv2.circle(frame, (w - 16, 21), 7, mc.get(mode, (200,200,200)), -1)


def draw_detection(frame, det, mode, anim_t):
    label = det["label"]
    conf  = det["conf"]
    x, y, w, h = det["bbox"]

    info = OBJECT_INFO.get(label, {
        "pt": label.capitalize(),
        "info": "Objeto COCO",
        "color": (180, 180, 180),
    })
    col    = info["color"]
    name   = info["pt"]
    detail = info["info"]

    # Borda animada
    pulse = int(160 + 95 * np.sin(anim_t * 3.5))
    glow  = tuple(min(int(c * 0.55 + pulse * 0.45), 255) for c in col)
    cv2.rectangle(frame, (x, y), (x+w, y+h), glow, 1)
    cv2.rectangle(frame, (x, y), (x+w, y+h), col, 2)

    # Cantos em destaque
    sz = max(10, min(20, w//5, h//5))
    for cx2, cy2 in [(x,y),(x+w,y),(x,y+h),(x+w,y+h)]:
        dx = sz if cx2 == x else -sz
        dy = sz if cy2 == y else -sz
        cv2.line(frame, (cx2,cy2), (cx2+dx,cy2), col, 3)
        cv2.line(frame, (cx2,cy2), (cx2,cy2+dy), col, 3)

    if mode == "SIMPLES":
        tag = f"{name}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.64, 2)
        ty = max(y - 8, th + 5)
        ov = frame.copy()
        cv2.rectangle(ov, (x-2, ty-th-6), (x+tw+6, ty+4), (15,15,15), -1)
        cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame, tag, (x+2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.64, col, 2, cv2.LINE_AA)
        return

    # Painel lateral INFO / DEBUG
    bar = "" * int(conf * 14) + "" * (14 - int(conf * 14))
    lines = [
        (f" {name}", col, 0.58, 2),
        (f" {bar} {conf:.0%}", (150,150,150), 0.36, 1),
        (" ", (55,55,55), 0.33, 1),
        (f" {detail}", (200,210,230), 0.40, 1),
    ]
    if mode == "DEBUG":
        lines += [
            (" ", (55,55,55), 0.33, 1),
            (f" COCO: {label}", (180,255,180), 0.36, 1),
            (f" bbox: {x},{y}  {w}x{h}", (180,255,180), 0.36, 1),
            (f" conf: {conf:.3f}", (180,255,180), 0.36, 1),
        ]

    lh = 21
    ph = len(lines) * lh + 14
    pw = 210
    px = x + w + 12
    py = max(y, 45)
    fh, fw = frame.shape[:2]
    if px + pw > fw: px = x - pw - 12
    px = max(0, px)
    py = min(py, fh - ph - 5)

    ov = frame.copy()
    cv2.rectangle(ov, (px,py), (px+pw,py+ph), (12,12,30), -1)
    cv2.addWeighted(ov, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (px,py), (px+pw,py+ph), glow, 2)
    cv2.line(frame, (x+w//2, y+h//2), (px, py+ph//2), col, 1, cv2.LINE_AA)

    for i, (t, c, sc, th) in enumerate(lines):
        cv2.putText(frame, t, (px+4, py+18+i*lh),
                    cv2.FONT_HERSHEY_SIMPLEX, sc, c, th, cv2.LINE_AA)


def draw_help(frame):
    h, w = frame.shape[:2]
    pw, ph = 340, 250
    px, py = w//2-pw//2, h//2-ph//2
    ov = frame.copy()
    cv2.rectangle(ov, (px,py), (px+pw,py+ph), (10,10,40), -1)
    cv2.addWeighted(ov, 0.88, frame, 0.12, 0, frame)
    cv2.rectangle(frame, (px,py), (px+pw,py+ph), (100,180,255), 2)
    lines = [
        ("CONTROLES", (160,220,255), 0.62),
        ("", None, 0.4),
        (" M   Modo: INFO / SIMPLES / DEBUG", (200,200,200), 0.42),
        (" P   Pausar / Continuar", (200,200,200), 0.42),
        (" C   Screenshot", (200,200,200), 0.42),
        (" +/- Ajustar confianca minima", (200,200,200), 0.42),
        (" H   Esta ajuda", (200,200,200), 0.42),
        (" ESC Sair", (200,200,200), 0.42),
        ("", None, 0.4),
        (" Modelo: YOLOv8n (80 classes COCO)", (150,255,150), 0.42),
        (" Garrafa  Copo  Livro  Celular", (150,255,150), 0.40),
        (" Mochila  Bolsa  Laptop  Tesoura...", (150,255,150), 0.40),
    ]
    for i,(t,c,sc) in enumerate(lines):
        if t and c:
            cv2.putText(frame, t, (px+10, py+28+i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, sc, c, 1, cv2.LINE_AA)


#  Main 

def main():
    print("=" * 55)
    print("  AR Inspector  YOLOv8n")
    print("=" * 55)

    print("  Carregando modelo YOLOv8n...")
    print("  Na primeira execucao, o yolov8n.pt pode ser baixado automaticamente.")
    model = YOLOv8()
    print("  Modelo pronto!")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  Erro: camera nao encontrada.")
        print("  Conecte uma webcam e tente novamente.")
        return

    print(f"  Camera aberta  |  conf>={CONF_THRESH}")
    print("  Pressione H para ver os controles")
    print("=" * 55)

    mode_idx  = 0
    paused    = False
    show_help = False
    last_frame = None
    conf_thresh = CONF_THRESH
    anim_t    = 0.0
    prev_time = time.time()
    fps       = 0.0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            last_frame = frame.copy()
        else:
            frame = last_frame.copy()

        now = time.time()
        dt  = max(now - prev_time, 1e-6)
        prev_time = now
        fps = fps * 0.9 + (1.0 / dt) * 0.1
        anim_t += dt

        mode = MODES[mode_idx]

        detections = model.detect(frame, conf_thresh)

        for det in detections:
            draw_detection(frame, det, mode, anim_t)

        draw_hud(frame, fps, mode, len(detections), paused)

        # Mostrar confianca atual no canto
        cv2.putText(frame, f"conf>={conf_thresh:.2f}  [+/-]",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160,160,160), 1)

        if show_help:
            draw_help(frame)

        cv2.imshow("AR YOLOv8", frame)

        key = cv2.waitKey(1) & 0xFF
        if   key == 27:             break
        elif key in (ord('m'),ord('M')): mode_idx = (mode_idx+1) % len(MODES)
        elif key in (ord('p'),ord('P')): paused = not paused
        elif key in (ord('h'),ord('H')): show_help = not show_help
        elif key == ord('+'):       conf_thresh = min(0.95, conf_thresh + 0.05)
        elif key == ord('-'):       conf_thresh = max(0.10, conf_thresh - 0.05)
        elif key in (ord('c'),ord('C')):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            p  = os.path.join(OUTPUT_DIR, f"ar_{ts}.png")
            cv2.imwrite(p, frame)
            print(f"  Screenshot: {p}")

    cap.release()
    cv2.destroyAllWindows()
    print("Encerrado.")


if __name__ == "__main__":
    main()
