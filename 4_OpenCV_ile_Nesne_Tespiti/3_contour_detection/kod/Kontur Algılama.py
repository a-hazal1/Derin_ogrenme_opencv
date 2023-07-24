import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("C:\\Users\\Hazal\\Derin_ogrenme_ile_goruntu_isleme\\OpenCV_ile_Nesne_Tespiti\\3_contour_detection\\resim\\contour.jpg",0)# resmi içe aktar
plt.figure(), plt.imshow(img, cmap = "gray"), plt.axis("off") # görselleştirme,img göster,eksenleri gösterme

###kodu iyileştirmek için eklenmiş ekstra alan
gb = cv2.GaussianBlur(img, ksize = (3,3), sigmaX = 7)
plt.figure(), plt.imshow(gb, cmap="gray"), plt.axis("off"), plt.title("Gauss Blur")

###

contours, hierarch = cv2.findContours(gb, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)#resim, cv2.RETR_CCOMP=iç ve dış kontürleri almak için, cv2.CHAIN_APPROX_SIMPLE= yatay dikey ve çapraz bölümleri sıkışturur ve uç noktalarını alır

external_contour = np.zeros(img.shape)#dış kontürler
internal_contour = np.zeros(img.shape)#iç kontürler

for i in range(len(contours)):
    
    # external
    if hierarch[0][i][3] == -1:
        cv2.drawContours(external_contour,contours, i, 255)
    else: # internal
        cv2.drawContours(internal_contour,contours, i, 255,-1)

plt.figure(), plt.imshow(external_contour, cmap = "gray"),plt.axis("off") # görselleştirme,img göster,eksenleri gösterme
plt.figure(), plt.imshow(internal_contour, cmap = "gray"),plt.axis("off") # görselleştirme,img göster,eksenleri gösterme
plt.show()