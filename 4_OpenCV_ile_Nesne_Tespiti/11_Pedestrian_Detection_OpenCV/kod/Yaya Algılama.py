import cv2
import os

img_path_list = ["C:\\Users\\Hazal\\Derin_ogrenme_ile_goruntu_isleme\\4_OpenCV_ile_Nesne_Tespiti\\11_Pedestrian_Detection_OpenCV\\resimler\\img1.jpg",
                 "C:\\Users\\Hazal\\Derin_ogrenme_ile_goruntu_isleme\\4_OpenCV_ile_Nesne_Tespiti\\11_Pedestrian_Detection_OpenCV\\resimler\\img2.jpg",
                 "C:\\Users\\Hazal\\Derin_ogrenme_ile_goruntu_isleme\\4_OpenCV_ile_Nesne_Tespiti\\11_Pedestrian_Detection_OpenCV\\resimler\\img3.jpg"]

    
print(img_path_list)

# hog tanımlayıcısı
hog = cv2.HOGDescriptor()
# tanımlayıcıa SVM ekle
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

for imagePath in img_path_list:
    print(imagePath)
    
    image = cv2.imread(imagePath)
    
    (rects, weights) = hog.detectMultiScale(image, padding = (8,8), scale = 1.05)
    
    for (x,y,w,h) in rects:
        cv2.rectangle(image, (x,y),(x+w,y+h),(0,0,255),2)
         
    cv2.imshow("Yaya: ",image)
    
    if cv2.waitKey(0) & 0xFF == ord("q"): continue
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    