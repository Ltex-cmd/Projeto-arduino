import cv2
import glob
import os
import serial
import time

import torch
from glasses_detector import GlassesClassifier

# ---------- Ajustes (mexa aqui pra calibrar) ----------
KIND            = "anyglasses" # "anyglasses" (qualquer oculos) | "eyeglasses" (so de grau)
PROBA_THRESHOLD = 0.70         # confianca minima (0-1). Subir = menos falso positivo.
CROP_MARGIN     = 1.0          # margem em volta do rosto (1.0 calibrado pra caixa do YuNet)
CLASSIFY_EVERY  = 3            # classifica 1 a cada N frames
CONFIRM_HITS    = 2            # N positivas seguidas pra disparar
REARM_MISSES    = 3            # N negativas seguidas rearmam o gatilho
COOLDOWN_S      = 4            # segundos minimos entre disparos
FACE_SCORE      = 0.6          # confianca minima do detector de rosto (YuNet)
CAM_W, CAM_H    = 640, 480     # resolucao da captura (menor = mais rapido)
SERIAL_BAUD     = 9600         # tem que bater com Serial.begin() do .ino
# ------------------------------------------------------


def find_serial_port():
    """Acha a porta do Nano sozinho. Mac: tty.wchusbserial/usbserial. Linux: ttyUSB/ttyACM."""
    for pattern in ("/dev/tty.wchusbserial*", "/dev/tty.usbserial*",
                    "/dev/tty.usbmodem*", "/dev/cu.wchusbserial*",
                    "/dev/ttyUSB*", "/dev/ttyACM*"):
        hits = glob.glob(pattern)
        if hits:
            return hits[0]
    return None


# GPU se existir (Mac: mps / Linux com NVIDIA: cuda); senao CPU.
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(f"Carregando modelo em {DEVICE} (1a vez baixa os pesos)...")
classifier = GlassesClassifier(kind=KIND, device=DEVICE)

# Detector de rosto YuNet (neural, robusto a franja/angulo/oclusao).
YUNET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "face_detection_yunet_2023mar.onnx")
face_detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 320), FACE_SCORE)
print("Modelo pronto.")

# Serial OPCIONAL: se o Arduino nao estiver ligado, roda so a deteccao.
port = find_serial_port()
arduino = None
if port:
    try:
        arduino = serial.Serial(port, SERIAL_BAUD, timeout=2)
        time.sleep(2)   # abrir a serial RESETA o Arduino; espera ele voltar
        arduino.reset_input_buffer()
        print(f"Arduino conectado em {port}")
    except Exception as e:
        print(f"Falha ao abrir {port} ({e}). Rodando so deteccao.")
else:
    print("Nenhuma porta de Arduino achada. Rodando so deteccao.")


def fire():
    """Manda '1' e espera o ACK 'FIRED' do firmware."""
    global arduino
    if not arduino:
        return
    try:
        arduino.reset_input_buffer()
        arduino.write(b'1')
        resp = arduino.readline().decode(errors="ignore").strip()
        if resp == "FIRED":
            print("ACK: FIRED")
        elif resp == "COOLDOWN":
            print("firmware em cooldown, jato segurado")
        else:
            print(f"sem ACK (resp: '{resp}') - confira cabo/firmware")
    except serial.SerialException:
        print("Serial caiu (Arduino desplugado?) - seguindo so com deteccao.")
        arduino = None


cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # sem fila de frames velhos = sem lag

last_trigger = 0
frame_n = 0
hits = 0            # positivas seguidas
misses = 0          # negativas seguidas
armed = True        # gatilho por borda: 1 disparo por aparicao
last_boxes = []     # rostos do ultimo frame classificado: ((x0,y0,x1,y1), proba, usando)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame_n += 1
    if frame_n % CLASSIFY_EVERY == 0:
        fh, fw = frame.shape[:2]
        face_detector.setInputSize((fw, fh))
        _, faces = face_detector.detect(frame)

        wearing = False
        last_boxes = []

        if faces is not None:
            for f in faces:                      # todos os rostos do quadro
                x, y, w, h = f[:4].astype(int)
                mx, my = int(CROP_MARGIN * w), int(CROP_MARGIN * h)
                x0, y0 = max(0, x - mx), max(0, y - my)
                x1, y1 = min(fw, x + w + mx), min(fh, y + h + my)
                crop = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)

                proba = float(classifier.predict(crop, format="proba"))
                has = proba >= PROBA_THRESHOLD
                wearing = wearing or has         # qualquer rosto de oculos conta
                last_boxes.append(((x0, y0, x1, y1), proba, has))

            print("  ".join(f"rosto {p:.2f}{'*' if u else ''}"
                            for _, p, u in last_boxes)
                  + f"  (seguidas: {hits}, armado: {armed})")
        else:
            print(f"sem rosto  (armado: {armed})")

        if wearing:
            hits += 1
            misses = 0
        else:
            hits = 0
            misses += 1
            if misses >= REARM_MISSES and not armed:
                armed = True          # alvo sumiu -> rearma pro proximo
                print("rearmado")

        if armed and hits >= CONFIRM_HITS:
            now = time.time()
            if now - last_trigger > COOLDOWN_S:
                last_trigger = now
                armed = False         # 1 jato por aparicao
                print("OCULOS! -> disparar")
                fire()

    # Desenha os resultados na tela
    for (x0, y0, x1, y1), proba, has in last_boxes:
        color = (0, 255, 0) if has else (180, 180, 180)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        label = f"oculos {proba:.0%}" if has else f"sem oculos {proba:.0%}"
        cv2.putText(frame, label, (x0, max(20, y0 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == 27:   # ESC sai
        break

cam.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
