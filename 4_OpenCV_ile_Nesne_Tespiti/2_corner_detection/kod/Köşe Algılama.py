import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("C:\\Users\\Hazal\\Derin_ogrenme_ile_goruntu_isleme\\OpenCV_ile_Nesne_Tespiti\\2_corner_detection\\resimler\\sudoku.jpg", 0)# resmi içe aktar
img = np.float32(img)#resmi ondalıklı sayılara çeviriyoruz(sanırım hassaslık için)
print(img.shape)
plt.figure(), plt.imshow(img, cmap = "gray"), plt.axis("off")# görselleştirme,img göster,eksenleri gösterme

# harris corner detection
dst = cv2.cornerHarris(img, blockSize = 2, ksize = 3, k = 0.04)#blocksize=komşuluk boyutu;ne kadar komşusuna bakılacağını söyler, kernel size kutucuğun boyutu, k free parametre
plt.figure(), plt.imshow(dst, cmap = "gray"), plt.axis("off")# görselleştirme,img göster,eksenleri gösterme

dst = cv2.dilate(dst, None)#dilate fonk tespit edilen noktaları genişletmeye yarar
img[dst>0.2*dst.max()] = 1 #1 yaparak renkleri değişti
plt.figure(), plt.imshow(dst, cmap = "gray"), plt.axis("off")# görselleştirme,img göster,eksenleri gösterme


# shi tomasi detection(harris in gelişmiş bir algoritmasıdır)
img = cv2.imread("C:\\Users\\Hazal\\Derin_ogrenme_ile_goruntu_isleme\\OpenCV_ile_Nesne_Tespiti\\2_corner_detection\\resimler\\sudoku.jpg", 0)#resmi içeri aktar
img = np.float32(img)#resmi ondalıklı sayılara çeviriyoruz
corners = cv2.goodFeaturesToTrack(img, 120, 0.01, 10)#resim,max tespit , kalite , minimum mesafe #bu fonk görüntüdeki en güçlü N köşeyi algılar
corners = np.int64(corners)#cornerları int tipine çeviriyoruz

for i in corners:
    x,y = i.ravel()#.ravel() girişin ögelerini içeren 1 boyutlu bir dizi döndürür
    cv2.circle(img, (x,y),3,(125,125,125),cv2.FILLED)#x,y dairenin merkezi,renk, filled ise içini doldurur
    
plt.imshow(img),plt.axis("off")# img göster,eksenleri gösterme
plt.show()


