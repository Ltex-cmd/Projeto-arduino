import cv2
import glob
import numpy as np
import serial
import time
import os

# ---------- Ajustes (mexa aqui pra calibrar) ----------
ORB_FEATURES    = 2000         # quantos pontos o ORB procura (mais = detecta mais longe)
RATIO_TEST      = 0.75         # ratio de Lowe: menor = mais rigido (menos falso positivo)
MIN_MATCHES     = 10           # minimo de matches pra tentar a homografia
INLIER_THRESHOLD = 12          # inliers RANSAC pra confirmar "achou a foto". Calibrar.
COOLDOWN_S      = 4            # segundos entre disparos (evita metralhar o servo)
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

# Detector ORB
orb = cv2.ORB_create(ORB_FEATURES)

# Imagem de referencia (a que voce vai imprimir/mostrar)
# Caminho relativo ao script: funciona rodando de qualquer pasta.
TARGET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nerd.jpeg")
target = cv2.imread(TARGET_PATH, 0)
if target is None:
    print("Erro: nao carregou nerd.jpeg")
    exit()

kp1, des1 = orb.detectAndCompute(target, None)

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
bf = cv2.BFMatcher(cv2.NORM_HAMMING)   # sem crossCheck: knnMatch faz o ratio test
last_trigger = 0
misses = 0        # frames seguidos sem a foto
armed = True      # gatilho por borda: 1 disparo por aparicao da foto
REARM_MISSES = 10 # frames sem a foto pra rearmar

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp2, des2 = orb.detectAndCompute(gray, None)

    # Sem descritores suficientes nesse frame: so mostra a camera e segue
    if des1 is None or des2 is None or len(des2) < 2:
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    # Ratio test de Lowe: pra cada ponto, compara o melhor com o 2o melhor.
    # So aceita se o melhor for bem melhor que o 2o -> match confiavel.
    good = []
    for pair in bf.knnMatch(des1, des2, k=2):
        if len(pair) == 2:
            m, n = pair
            if m.distance < RATIO_TEST * n.distance:
                good.append(m)

    # Homografia + RANSAC: os matches precisam formar a geometria de um
    # plano (a foto). Matches espalhados aleatorios nao formam -> corta
    # falso positivo e permite threshold baixo com seguranca.
    inliers = 0
    corners = None
    if len(good) >= MIN_MATCHES:
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is not None:
            inliers = int(mask.sum())
            h, w = target.shape
            box = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            corners = cv2.perspectiveTransform(box, H)

    print(f"Matches: {len(good)}  inliers: {inliers}  armado: {armed}")

    if inliers >= INLIER_THRESHOLD:
        misses = 0
        cv2.putText(frame, "FOUND!", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if corners is not None:
            cv2.polylines(frame, [np.int32(corners)], True, (0, 255, 0), 3)

        now = time.time()
        if armed and now - last_trigger > COOLDOWN_S:
            last_trigger = now
            armed = False     # 1 jato por aparicao; rearma quando a foto sair do quadro
            print("FOUND! -> disparar")
            fire()
    else:
        misses += 1
        if misses >= REARM_MISSES and not armed:
            armed = True      # foto sumiu -> pronto pro proximo
            print("rearmado")

    # Janela de debug: mostra os matches
    match_img = cv2.drawMatches(
        target, kp1, gray, kp2, good[:30], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imshow("Matches", match_img)
    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == 27:   # ESC sai
        break

cam.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
