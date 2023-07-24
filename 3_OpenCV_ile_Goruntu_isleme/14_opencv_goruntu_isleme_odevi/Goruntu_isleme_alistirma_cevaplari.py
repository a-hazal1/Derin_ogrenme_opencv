import cv2
import matplotlib.pyplot as plt

img = cv2.imread('C:\\Users\\Hazal\\2_OpenCV_ile_Goruntu_isleme\\14_opencv_goruntu_isleme_odevi\\resimler\\odev1.jpg', 0)

cv2.imshow('Odev-1',img)
print(img.shape)# resmin boyutu

# resmi 4/5 oranında yeniden boyutlandır ve resmi çizdir
imgResized = cv2.resize(img,(int(img.shape[1]*4/5),int(img.shape[0]*4/5)))
cv2.imshow("Yeniden Boyutlandiralim", imgResized)

# orijinal resme bir yazı ekle 
cv2.putText(img,"kedi ",(600,100),cv2.FONT_HERSHEY_COMPLEX, 1 ,(0,0,0))
cv2.imshow('Kedi Text', img)

#  orijinal resme 50 threshold değeri üzerindekileri beyaz yap altındakileri siyah yap, 
# binary threshold yöntemi kullan
_, thresh_img = cv2.threshold(img, thresh = 50, maxval = 255, type = cv2.THRESH_BINARY)
cv2.imshow('Threshold', thresh_img)

# orijinal resme gaussian bulanıklaştırma uygula
gb = cv2.GaussianBlur(img, ksize = (3,3), sigmaX = 7)
cv2.imshow('Gaussian Bulanik', gb)

# orijinal resme Laplacian  gradyan uygulay
laplacian = cv2.Laplacian(img, ddepth = cv2.CV_64F)
cv2.imshow('Laplacian', laplacian)

# orijinal resmin histogramını çizdir
img_hist = cv2.calcHist([img], channels = [0], mask = None, histSize = [256], ranges = [0,256])
plt.figure()
plt.plot(img_hist)

k = cv2.waitKey(0) &0xFF

if k == 27: # esc
    cv2.destroyAllWindows()










