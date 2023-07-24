import cv2
import numpy as np
from collections import deque

# nesne merkezini depolayacak veri tipi
buffer_size = 16  # deque nin boyutu
pts = deque(maxlen=buffer_size)  # nesnenin merkez pointleri ddequeinin içinde max size ı yukarıda belirlediğimiz

# mavi renk aralığı HSV
blueLower = (84, 98, 0)
blueUpper = (179, 255, 240)

# capture
cap = cv2.VideoCapture(0) # video yakalama nesnesini başlat
cap.set(3, 960)
cap.set(4, 480)

while True:

    success, imgOriginal = cap.read() #videoçekiminden bir kare okur

    if success:

        # blur
        blurred = cv2.GaussianBlur(imgOriginal, (11, 11), 0) #gürültü azaltmak için gauss kullan

        # hsv
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV Image", hsv)

        mask = cv2.inRange(hsv, blueLower, blueUpper)# mavi için alt ve üst sınırlarını temel alarak maske oluştur
        mask = cv2.erode(mask, None, iterations=2)# maskenin etrafında kalan gürültüleri sil
        mask = cv2.dilate(mask, None, iterations=2)#ve genişlet
        cv2.imshow("Mask + erozyon + genisleme", mask)

        # kontur
        (contours, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#conturleri bulmak için findcontours

        for c in contours:
            # tespit edilen her bir kontür için kontürü çevreleyen minimum dkdörtgeni hesapla
            rect = cv2.minAreaRect(c)
            ((x, y), (width, height), rotation) = rect

            s = "x: {}, y: {}, width: {}, height: {}, rotation: {}".format(np.round(x), np.round(y), np.round(width),
                                                                           np.round(height), np.round(rotation))
            print(s)

            # Bounding box
            box = cv2.boxPoints(rect)
            box = np.int64(box)

            # Moment
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # Draw the contour and center
            cv2.drawContours(imgOriginal, [box], 0, (0, 255, 255), 2)
            cv2.circle(imgOriginal, center, 5, (255, 0, 255), -1)
            cv2.putText(imgOriginal, s, (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)
            
            ###nesnenin hareket yolu için
            # pts.appendleft(center)

            # for i in range(1, len(pts)):
            #     if pts[i - 1] is None or pts[i] is None:
            #         continue
            #     cv2.line(imgOriginal, pts[i - 1], pts[i], (0, 255, 0), 3)
            ###
            
        cv2.imshow("Orijinal Tespit", imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


