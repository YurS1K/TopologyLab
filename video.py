import cv2
import numpy as np

y1 = 640
y2 = 1080
cap = cv2.VideoCapture("13.mp4")
backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=10, history=200, detectShadows=False) # Пример настройки
while True:
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fg_mask = backSub.apply(gray, learningRate=0.009)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))  # Больший размер ядра
    # fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
    # fg_mask = cv2.dilate(fg_mask, kernel, iterations=3)
    mask_eroded = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("mask", mask_eroded)


    contours, hierarchy = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_contour_area = 600
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Расчёт центроидов, рисование рамки
    cur_centroids = []

    for cnt in large_contours:
        x, y, w, h = cv2.boundingRect(cnt)

        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


    cv2.imshow('video feed', frame)
cv2.destroyAllWindows()