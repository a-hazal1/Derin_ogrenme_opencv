import cv2
import os

# files = os.listdir("C:\\Users\\Hazal\\Derin_ogrenme_ile_goruntu_isleme\\OpenCV_ile_Nesne_Tespiti\\9_cat_face_detection\\resimler")
# print(files)
img_path_list = ["C:\\Users\\Hazal\\Derin_ogrenme_ile_goruntu_isleme\\4_OpenCV_ile_Nesne_Tespiti\\9_cat_face_detection\\resimler\\cat_img1.jpg",
                 "C:\\Users\\Hazal\\Derin_ogrenme_ile_goruntu_isleme\\4_OpenCV_ile_Nesne_Tespiti\\9_cat_face_detection\\resimler\\cat_img2.jpg",
                 "C:\\Users\\Hazal\\Derin_ogrenme_ile_goruntu_isleme\\4_OpenCV_ile_Nesne_Tespiti\\9_cat_face_detection\\resimler\\cat_img3.jpg"]
# for f in files:
#     if f.endswith(".jpg"): #eğer sonu .jpg ile bitiyorsa
#         img_path_list.append(f)#listeye ekle
print(img_path_list)

for j in img_path_list:
    print(j)
    image = cv2.imread(j)#resmi oku
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#resmi gri formata dönüştür
    
    detector = cv2.CascadeClassifier("C:\\Users\\Hazal\\Derin_ogrenme_ile_goruntu_isleme\\OpenCV_ile_Nesne_Tespiti\\9_cat_face_detection\\haarcascade\\haarcascade_frontalcatface.xml")
    rects = detector.detectMultiScale(gray, scaleFactor = 1.045, minNeighbors = 2)
    
    for (i, (x,y,w,h)) in enumerate(rects):
        cv2.rectangle(image, (x,y), (x+w, y+h),(0,255,255),2)
        cv2.putText(image, "Kedi {}".format(i+1), (x,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255),2)
        
    cv2.imshow(j, image)
    if cv2.waitKey(0) & 0xFF == ord("q"): continue#q ya bastıkça resimleri sırayla göster
