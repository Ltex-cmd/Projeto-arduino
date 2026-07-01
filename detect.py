import cv2
import serial
import time
import os

# ---------- Ajustes (mexa aqui pra calibrar) ----------
ORB_FEATURES   = 1500          # quantos pontos o ORB procura
MATCH_DISTANCE = 45            # dist. maxima p/ match "bom" (menor = mais rigido)
MATCH_THRESHOLD = 50           # quantos matches bons = "achou a foto"
COOLDOWN_S     = 4             # segundos entre disparos (evita metralhar o servo)
SERIAL_PORT    = "/dev/tty.usbserial-XXXX"  # AJUSTAR no Mac: rode  ls /dev/tty.*
SERIAL_BAUD    = 9600          # tem que bater com Serial.begin() do .ino
# ------------------------------------------------------

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
try:
    arduino = serial.Serial(SERIAL_PORT, SERIAL_BAUD)
    time.sleep(2)   # abrir a serial RESETA o Arduino; espera ele voltar
    print(f"Arduino conectado em {SERIAL_PORT}")
except Exception as e:
    arduino = None
    print(f"Sem Arduino ({e}). Rodando so deteccao (sem disparar servo).")

cam = cv2.VideoCapture(0)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
last_trigger = 0

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp2, des2 = orb.detectAndCompute(gray, None)

    # Sem descritores nesse frame: so mostra a camera e segue
    if des1 is None or des2 is None:
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good = [m for m in matches if m.distance < MATCH_DISTANCE]

    print(f"Matches bons: {len(good)}")

    if len(good) > MATCH_THRESHOLD:
        cv2.putText(frame, "FOUND!", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        now = time.time()
        if now - last_trigger > COOLDOWN_S:   # cooldown: so dispara 1x a cada COOLDOWN_S
            last_trigger = now
            print("FOUND! -> disparar")
            if arduino:
                try:
                    arduino.write(b'1')
                except serial.SerialException:
                    print("Serial caiu (Arduino desplugado?) - seguindo so com deteccao.")
                    arduino = None

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
