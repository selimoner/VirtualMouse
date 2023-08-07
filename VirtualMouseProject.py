import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy
import mediapipe as mp

# Değişkenlerimizi tanımlıyoruz
camWidth, camHeight = 640, 480
previousTime = 0
previousLocationX, previousLocationY = 0, 0
currentLocationX, currentLocationY = 0,0

detector = htm.handDetector(maxHands=2) # Ekranda sadece 1 el olacağını deklare ediyoruz

cap = cv2.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)

screenWidth, screenHeight = autopy.screen.size()
frameReduction = 100 # Frame Reduction
smoothening = 5
# 1536.0 864.0
# print(screenWidth, screenHeight)

cv2.namedWindow("Frame", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)

while True:
    # 1.Adım : Eldeki noktaları al
    success, img = cap.read()
    img = detector.findHands(img)
    landmarksList, boundingBox = detector.findPosition(img)

    # 2.Adım : işaret ve orta parmağın uçlarının indekslerini al
    if len(landmarksList) != 0:
        x1, y1 = landmarksList[8][1:]
        x2, y2 = landmarksList[12][1:]

        print(x1, y1, x2, y2)

        # 3.Adım : Hangi parmakların kalkık olduğunu kontrol et
        fingers = detector.fingersUp()
        # print(fingers)

        cv2.rectangle(img, (frameReduction, frameReduction), (camWidth - frameReduction, camHeight - frameReduction),
                      (255, 0, 255), 2)

        # 4.Adım : Sadece işaret parmağı aktifse imleci hareket ettir
        if fingers[1] == 1 and fingers[2] == 0:

            # 5.Adım : Koordinatları dönüştür

            x3 = np.interp(x1, (frameReduction, camWidth-frameReduction), (0, screenWidth))
            y3 = np.interp(y1, (frameReduction, camHeight-frameReduction), (0, screenHeight))
            # 6.Adım : Değerleri düzelt

            currentLocationX = previousLocationX + (x3 - previousLocationX) / smoothening
            currentLocationY = previousLocationY + (y3 - previousLocationY) / smoothening

            # 7.Adım : Imleci Hareket ettir
            autopy.mouse.move(screenWidth - currentLocationX, currentLocationY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            previousLocationX, previousLocationY = currentLocationX, currentLocationY

        # 8.Adım : hem işaret hemde orta parmak havadaysa tıkla
        if fingers[1] == 1 and fingers[2] == 1:
            # 9.Adım : parmaklar arasındaki mesafeyi bul
            length, img, lineInfo = detector.findDistance(8,12, img)
            print(length)
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                # 10.Adım : eğer aradaki mesafe kısaysa tıkla
                autopy.mouse.click()

    # 11.Adım : FPS i ölç
    currentTime = time.time()
    fps = 1/(currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)

    # 12.Adım : Ekranda göster

    cv2.imshow("Frame", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


