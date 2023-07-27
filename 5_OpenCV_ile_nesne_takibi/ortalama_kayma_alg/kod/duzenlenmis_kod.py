#mod: en yüksek veri noktası yoğunluğu olarak tanımlanabilir
import cv2
import numpy as np

cap= cv2.VideoCapture(0)#kamerayı aç

ret,frame=cap.read()#bir frame okuma

if ret==False:
    print("uyari")
    
#detection
face_cascade= cv2.CascadeClassifier("C:\\Users\\Hazal\\Derin_ogrenme_ile_goruntu_isleme\\5_OpenCV_ile_nesne_takibi\\haarcascade\\haarcascade_frontalface_default.xml")
face_rects=face_cascade.detectMultiScale(frame)#bu değişken içerisinde x ve y değerleri bulunur
if len(face_rects) == 0:
    print("Warning: No face detected in the current frame.")
else:
    (face_x,face_y,w,h)=tuple(face_rects[0])#ilk değeri alınır 
    track_window= (face_x,face_y,w,h)#meanshift algoritması girdisi

    ##region of interest(tespitb ettiğimiz okutucu içerisi)

    roi=frame[face_y : face_y+h , face_x : face_x+w]

    hsv_roi= cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    roi_hist= cv2.calcHist([hsv_roi],[0],None,[180],[0,180])#histogram hesaplama range 0 ile 180 arasında#takip için histogram gerekli
    cv2.normalize(roi_hist,roi_hist, 0, 255, cv2.NORM_MINMAX)#0 ile255 arasında sıkıştıracak şekilde normalize etmek

    # takip için gerekli durdurma kriterleri
    # count =hesaplanacak maksimum üye sayısı
    # epsilon=yinelemeli alg isteden doğruluk veya !değişiklik!

    term_crit=(cv2.TermCriteria_EPS | cv2.TermCriteria_COUNT, 5, 1 )#5 yineleme veye 1 adet epsilon demek(aşağıdan durma kriteri)

    #takibe başlama:
    while True:
        ret,frame=cap.read()
        if ret:
            hsv= cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            # histogramı bir görüntüde bulmak için:
            #belirli bir görüntünün piksellerinin bir histogram modelinin piksel dağılımına ne kadar uyduğumu kaydetmek:
            dst=cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

            ret,track_window=cv2.meanShift(dst,track_window,term_crit)
            x,y,w,h = track_window #nesnenin konumu,yükseklik ve genişlik değerlerini döndürecek
            img2=cv2.rectangle(frame,(x,y),(x+h,y+w),(0,0,255),5)

            cv2.imshow("takip",img2)
            if cv2.waitKey(1) & 0xFF == ord("q"): 
                break
        
cap.release()
cv2.destroyAllWindows()

# tuple sorunu çözüldü:
# Bu sorunu çözmek için, ilk elemanına erişmeden önce boş olmadığından emin olmak için bir kontrol ekledim. Boşsa face_rects, hiçbir yüzün algılanmadığı anlamına gelir, bu nedenle ya bu çerçeveyi işlemeyi atladım ya da yüzün bulunmadığını belirten bir mesaj gösterttim.
