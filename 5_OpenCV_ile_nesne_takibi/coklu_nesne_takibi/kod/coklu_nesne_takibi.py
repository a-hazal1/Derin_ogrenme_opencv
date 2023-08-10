import cv2

OPENCV_OBJECT_TRACKERS = {"csrt"      : cv2.TrackerCSRT_create,
		                  "kcf"       : cv2.TrackerKCF_create,
		                  "boosting"  : cv2.TrackerBoosting_create,
		                  "mil"       : cv2.TrackerMIL_create,
		                  "tld"       : cv2.TrackerTLD_create,
		                  "medianflow": cv2.TrackerMedianFlow_create,
		                  "mosse"     : cv2.TrackerMOSSE_create,
                          "multiTracker": cv2.MultiTracker_create }#Bu sözlük izleyici adlarını (anahtarları) karşılık gelen OpenCV izleyici oluşturma işlevlerine (değerler) eşler.

tracker_name = "medianflow"#Bu, kullanmak istediğiniz izleyici algoritmasını ayarlar. Bu durumda, nesne takibi için bir algoritma olan "MedianFlow" izleyiciyi kullanıyor.

trackers = cv2.MultiTracker_create()#Birden çok nesne izleyici örneğini yönetecek bir nesne oluşturur .

video_path = "5_OpenCV_ile_nesne_takibi/coklu_nesne_takibi/video/MOT17-04-DPM.mp4"
cap = cv2.VideoCapture(video_path)

fps = 30     
f = 0
while True:
    
    ret, frame = cap.read()#videodan kareleri okur
    (H, W) = frame.shape[:2]#gyükseklik ve genişliğini alır
    frame = cv2.resize(frame, dsize = (960, 540))#framin boyutlarını düzenler
    
    (success , boxes) = trackers.update(frame)#Bu satır success(izlemenin başarılı olup olmadığını gösteren bir boole) ve boxes(izlenen sınırlayıcı kutuların bir listesini) döndürür .
    
    info = [("Tracker", tracker_name),
        	("Success", "Yes" if success else "No")]#Kullanılan izleyici ve izlemenin başarı durumu hakkında bilgi içeren bir demet listesi oluşturur .
    
    string_text = ""
    
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        string_text = string_text + text + " "
    
    cv2.putText(frame, string_text, (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)#bilgi metnini çerçeve üzerine çizer
    
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)#izlenen nesnelerin çevresine dikdörtgenler çizer.
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("t"):
        #"t" tuşuna basılırsa, komut dosyası ile nesne seçim moduna girerek 
        box = cv2.selectROI("Frame", frame, fromCenter=False)
    
        tracker = OPENCV_OBJECT_TRACKERS[tracker_name]()
        trackers.add(tracker, frame, box)
    elif key == ord("q"):break#"q" tuşuna basılırsa döngüden çıkılır.

    f = f + 1#İşlenen çerçevelerin kaydını tutar.
    
cap.release()
cv2.destroyAllWindows() 