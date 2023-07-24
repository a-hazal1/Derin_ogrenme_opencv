#%%
import cv2
import matplotlib.pyplot as plt

# içe aktar 
einstein = cv2.imread("C:\\Users\\Hazal\\Derin_ogrenme_ile_goruntu_isleme\\OpenCV_ile_Nesne_Tespiti\\8_face_detection\\resimler\\einstein.jpg", 0)
plt.figure(), plt.imshow(einstein, cmap = "gray"), plt.axis("off")# görselleştirme,img göster,eksenleri gösterme

# sınıflandırıcı
face_cascade = cv2.CascadeClassifier("C:\\Users\\Hazal\\Derin_ogrenme_ile_goruntu_isleme\\OpenCV_ile_Nesne_Tespiti\\8_face_detection\\haarcascade\\haarcascade_frontalface_default.xml")

face_rect = face_cascade.detectMultiScale(einstein)#Bu yöntem, Haar Cascade sınıflandırıcı kullanılarak girdi görüntüsündeki nesneleri (bu durumda yüzler) algılamak için kullanılır. 
#Her dikdörtgenin görüntüde algılanan bir yüzü temsil ettiği bir dikdörtgenler listesi döndürür.

for (x,y,w,h) in face_rect:
    cv2.rectangle(einstein, (x,y),(x+w, y+h),(255,255,255),10)
plt.figure(), plt.imshow(einstein, cmap = "gray"), plt.axis("off")# görselleştirme,img göster,eksenleri gösterme


# içe aktar 
barce = cv2.imread("C:\\Users\\Hazal\\Derin_ogrenme_ile_goruntu_isleme\\OpenCV_ile_Nesne_Tespiti\\8_face_detection\\resimler\\barcelona.jpg", 0)
plt.figure(), plt.imshow(barce, cmap = "gray"), plt.axis("off")# görselleştirme,img göster,eksenleri gösterme

face_rect = face_cascade.detectMultiScale(barce, minNeighbors = 7)


for (x,y,w,h) in face_rect:
    cv2.rectangle(barce, (x,y),(x+w, y+h),(255,255,255),10)
plt.figure(), plt.imshow(barce, cmap = "gray"), plt.axis("off")# görselleştirme,img göster,eksenleri gösterme
plt.show()#bütün img leri show et
#%%
# video
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)#kamerayı başlat

while True:
    
    ret, frame = cap.read()#kareleri oku
    
    if ret:
        
        face_rect = face_cascade.detectMultiScale(frame, minNeighbors = 7)#Haar Cascade sınıflandırıcı kullanılarak giriş çerçevesindeki nesneleri (bu durumda yüzler) algılamak için kullanılır.
            #minneighbors: Algılanan bir yüzü korumak için gereken minimum komşu dikdörtgen sayısını temsil eder.
        for (x,y,w,h) in face_rect:
            cv2.rectangle(frame, (x,y),(x+w, y+h),(255,255,255),10)
        cv2.imshow("face detect", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()





























