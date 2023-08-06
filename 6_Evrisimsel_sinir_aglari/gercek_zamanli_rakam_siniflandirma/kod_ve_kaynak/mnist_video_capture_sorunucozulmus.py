import cv2
import pickle#modeli kaydetmek için
import numpy as np


# preprocess(ön işleme)
def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)#resmi gri formata çevirmek
    img = cv2.equalizeHist(img)#gri tonlamalı resim üzerinde histogram eşitleme uygulandı
    img = img /255#normalize etmek için 
    
    return img

cap = cv2.VideoCapture(0)#kamerayı başlat
cap.set(3,480)
cap.set(4,480)
#genişliği ve yüksekliği ayarladık

pickle_in = open("C:\\Users\\Hazal\\Derin_ogrenme_ile_goruntu_isleme\\model_trained_new.p","rb")#eğitmiş olduğum modeli içeri aldım,read edeceğimiz için rb
model = pickle.load(pickle_in)#modeli yükle


while True:
    
    success, frame = cap.read()# Video kaynağından bir kare oku
    
    img = np.asarray(frame)#framei arraye çevirdik
    img = cv2.resize(img, (32,32))#nöral networkümüzü inputumuzu 32,32 almiştık tekrar resize ediyoruz
    img = preProcess(img) #önişlemeye tekrar sok
    
    img = img.reshape(1,32,32,1)#1tane resim old;32,32 boyutu;1 ise gri formatta olduğunu
    
    # predict
    pre=model.predict(img)
    classIndex = np.argmax(pre,axis=1)
    
    predictions = model.predict(img)
    probVal = np.amax(predictions)#dizideki max değeri bulur
    print(classIndex, probVal)
    
    if probVal > 0.7:#olasılığımız0,70 ten büyükse
        cv2.putText(frame, str(classIndex)+ "   "+ str(probVal), (50,50),cv2.FONT_HERSHEY_DUPLEX, 1,(0,255,0),1)

    cv2.imshow("Rakam Siniflandirma",frame)

    if cv2.waitKey(1) & 0xFF == ord("q"): break    
