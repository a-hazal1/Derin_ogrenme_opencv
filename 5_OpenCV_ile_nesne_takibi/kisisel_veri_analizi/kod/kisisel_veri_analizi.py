"""
veri seti
veri seti indir
resimleri videoya çevir
keşifsel veri analizi yap
takip edilen nesneye ait gt değerini çikar

"""
import cv2
import os
import os.path 
import isfile 
import join
import matplotlib.pyplot as plt

pathIn= r"C:\\Users\\Hazal\\Derin_ogrenme_ile_goruntu_isleme\\5_OpenCV_ile_nesne_takibi\\kisisel_veri_analizi\\kaynak\\img1"#başındaki r read anlamına geliyor

pathout="deneme.mp4"

files = [  for f in os.listdir(path) if isfile(join(path,f)) ]

#dosyam içerisindeki herhangibi bir frame aciyorum
img = cv2.imread(pathIn + "\\"+files[44])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#bgr formatından rgb formatına ceviriyorum
plt.imshow(img)#resmi gosteriyorum

#fps : olusturulan kare hizi
fps = 25

#videonun genislik ve yuksekligi
size = (1920,1080)

# hangi isimle kaydedecegim, video formati, True ise videonun renkli oldugu anlamina gelir
out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*"MP4V"), fps, size, True)

#bir for dongusu actim ve resimleri her bir frame ile sıra sıra birlestirdim
for i in files:
    print(i)
    
    filename = pathIn + "\\" + i
    
    img = cv2.imread(filename)
    
    out.write(img)

out.release()