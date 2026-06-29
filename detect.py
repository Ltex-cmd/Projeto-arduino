import cv2
import serial
import time

# Create the ORB detector
orb = cv2.ORB_create(1500)

# Load the reference image (the one you'll print)
target = cv2.imread("nerd.jpeg", 0)

# Check that the image loaded correctly
if target is None:
    print("Error: Could not load jpeg image")
    exit()

# Detect features in the reference image
kp1, des1 = orb.detectAndCompute(target, None)

# Open the webcam
cam = cv2.VideoCapture(0)

# VER SE O CÓDIGO É ESSE MESMO !!!!!!!!!!!!
# arduino = serial.Serial("/dev/ttyACM0", 9600)

while True:
    ret, frame = cam.read()

    if not ret:
        break

    # Convert the webcam frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect features in the current frame
    kp2, des2 = orb.detectAndCompute(gray, None)

    if des1 is None or des2 is None:
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    # Skip this frame if no descriptors were found
    if des2 is None:
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    # Create a brute-force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the descriptors
    matches = bf.match(des1, des2)

    # Sort matches by quality (smaller distance is better)
    matches = sorted(matches, key=lambda x: x.distance)

    # Keep only good matches
    good = [m for m in matches if m.distance < 45]

    print(f"Good matches: {len(good)}")

    # If enough good matches, we found the picture
    if len(good) > 50:
        cv2.putText(frame, "FOUND!", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        print("FOUND!")
        # arduino.write(b'1')
        last_trigger = time.time()  
    # Draw the first 30 good matches
    match_img = cv2.drawMatches(
        target,
        kp1,
        gray,
        kp2,
        good[:30],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imshow("Matches", match_img)

    # For now, just show the webcam
    cv2.imshow("Camera", frame)

    # Press ESC to quit
    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()