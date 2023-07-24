
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("C:\\Users\\Hazal\\Derin_ogrenme_ile_goruntu_isleme\\OpenCV_ile_Nesne_Tespiti\\1_edge_detection\\resim\\london.jpg", 0)# resmi içe aktar
plt.figure(), plt.imshow(img, cmap = "gray"), plt.axis("off") #görselleştirme,img göster,eksenleri gösterme
#en iyi sonucu alabilmek için siyah beyaz kullanıyoruz
edges = cv2.Canny(image = img, threshold1 = 0, threshold2 = 255)#Canny kenar dedektörü için alt eşik değeri,yüksek eşik değeri;bu değerler(0,255) kendi hesaplansın diye konmuştur
plt.figure(), plt.imshow(edges, cmap = "gray"), plt.axis("off") # görselleştirme,img göster,eksenleri gösterme

med_val = np.median(img)# pikselleri sıralayıp sıranın ortasındaki değeri alır
print(med_val)

low = int(max(0, (1 - 0.33)*med_val)) #0 ile medyan değerinin yüzde 67'si arasındaki max medyan değerini alt eşik değeri olarak aldık
high = int(min(255, (1 + 0.33)*med_val))#255 ile medyan değerinin yüzde 133'ü arasındaki min medyan değeriini üst eşik değeri olarak aldık

print(low)
print(high)

edges = cv2.Canny(image = img, threshold1 = low, threshold2 = high)# elde edilen lw ve high değerlerine göre threshold değerleri değiştirilerek kenar saptama
plt.figure(), plt.imshow(edges, cmap = "gray"), plt.axis("off") # görselleştirme,img göster,eksenleri gösterme


blurred_img = cv2.blur(img, ksize = (5,5))#resmi bulanıklaştırma, ksize=kernel size
plt.figure(), plt.imshow(blurred_img, cmap = "gray"), plt.axis("off") # görselleştirme,img göster,eksenleri gösterme

med_val = np.median(blurred_img)#blurlanan resmin piksellerini sıralayıp sıranın ortasındaki değeri alır
print(med_val)

low = int(max(0, (1 - 0.33)*med_val))#0 ile medyan değerinin yüzde 67'si arasındaki max medyan değerini alt eşik değeri olarak aldık
high = int(min(255, (1 + 0.33)*med_val))#255 ile medyan değerinin yüzde 133'ü arasındaki min medyan değeriini üst eşik değeri olarak aldık

print(low)
print(high)

edges = cv2.Canny(image = blurred_img, threshold1 = low, threshold2 = high)#yenilenen blurlu resimden tespit edilen threshold değerleri kullanılarak kenar saptama 
plt.figure(), plt.imshow(edges, cmap = "gray"), plt.axis("off")# görselleştirme,img göster,eksenleri gösterme

plt.show()





















# %%
