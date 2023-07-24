import cv2
import numpy as np 

# resmi içe aktar 
img = cv2.imread("C:\\Users\\Hazal\\2_OpenCV_ile_Goruntu_isleme\\6_joining_images\\resimler\\lenna.png")
cv2.imshow("Orijinal", img)

# yatay
hor = np.hstack((img,img))
cv2.imshow("Yatay",hor)

# dikey
ver = np.vstack((img,img))
cv2.imshow("Dikey",ver)

k = cv2.waitKey(0) &0xFF

if k == 27: # esc
    cv2.destroyAllWindows()
