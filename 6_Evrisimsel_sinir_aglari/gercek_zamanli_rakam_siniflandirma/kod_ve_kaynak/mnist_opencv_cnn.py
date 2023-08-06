import numpy as np
import cv2
import os 
from sklearn.model_selection import train_test_split#veriyi train ve test olarak ikiye ayırıyoruz
from sklearn.metrics import confusion_matrix#değerlendirme metriklerini karışıklık matrisi
import seaborn as sns#görselleştirmek için seaborn kullanacağız
import matplotlib.pyplot as plt#görselleştirmek için
from keras.models import Sequential#taban içine layer eklemek için
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
#dense: tam bağlantıdaki katmanlar
#dropout: seyreltme
#flatten: düzleştirme
#Conv2D: evrişim ağı
#MaxPooling2D: piksel ekleme
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import pickle#modeli yüklemek kaydetmek için

path = "C:\\Users\\Hazal\\Derin_ogrenme_ile_goruntu_isleme\\6_Evrisimsel_sinir_aglari\\gercek_zamanli_rakam_siniflandirma\\myData"

myList = os.listdir(path)#pathin içerisinde bulunan klasörlerin hepsini listeye atar
noOfClasses = len(myList)#mylistin uzunluğu #10

print("Label(sinif) sayisi: ",noOfClasses)


images = []
classNo = []#etiket için boş liste

for i in range(noOfClasses):
    myImageList = os.listdir(path + "\\"+str(i))#/mydata/0,1,2,3,4,5,6,7,8,9#herhangi bir klasörün içine girebileceğiz
    for j in myImageList:
        img = cv2.imread(path + "\\" + str(i) + "\\" + j)#image listemizin içinde bulunan resimlere ulaştık;j:isimleri
        img = cv2.resize(img, (32,32))#bizim eğitimiz nöral networkün girdisi 32 ye 32 olacağı için
        images.append(img)#resimleri resimler listesine
        classNo.append(i)#etiketleri classno listesine
        
print(len(images))#kaçresim olduğu
print(len(classNo))#üst satırla birbirine eşit

images = np.array(images)#images listesini array yaptık
classNo = np.array(classNo)#classNo listesini array yaptık

print(images.shape)#boyutlarına baktık(10160,32,32,3)
print(classNo.shape)#(10160,)

# veriyi ayırma
x_train, x_test, y_train, y_test = train_test_split(images, classNo, test_size = 0.5, random_state = 42)#test verisi yüzde 50, random state :bölünme
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size = 0.2, random_state = 42)
#veriyi test ve train olarak ayırdık
#oluşturulan verilerin boyutları:
print(images.shape)# 10160,32,32,3
print(x_train.shape)#4064,32,32,3
print(x_test.shape)#5080,32,32,3
print(x_validation.shape)#1016,32,32,3 doğrulama çiin

# # vis #görselleştirme
# fig, axes = plt.subplots(3,1,figsize=(7,7))# 3 satır ve 1 sütun alt nokta (eksen) içeren bir şekil oluşturur ve genel şekil boyutunu 7 x 7 olarak ayarlar. plt.subplots()figure nesnesini ve bir eksen nesneleri dizisini içeren bir demet döndürür.
# fig.subplots_adjust(hspace = 0.5)#Bu çizgi, şekildeki alt grafikler arasındaki boşluğu ayarlar.hspace: alt grafikler arasındaki yükseklik boşluğunu kontrol eder
# sns.countplot(y_train, ax = axes[0])# verilerdeki her benzersiz değerin sayısını gösteren bir çubuk grafik oluşturur
# axes[0].set_title("y_train")#Bu satır, ilk alt çizim için başlığı ayarlar (eksen[0]).

# sns.countplot(y_test, ax = axes[1])
# axes[1].set_title("y_test")

# sns.countplot(y_validation, ax = axes[2])
# axes[2].set_title("y_validation")

# preprocess(ön işleme)
def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#resmi gri formata çevirmek
    img = cv2.equalizeHist(img)#gri tonlamalı resim üzerinde histogram eşitleme uygulandı
    img = img /255#normalize etmek için 
    
    return img


# idx = 311 #311.index
# img = preProcess(x_train[idx])#önişlemeye gönder
# img = cv2.resize(img,(300,300))#güzel görünebilmesi için 300*300 olarak düzenle
# cv2.imshow("Preprocess ",img)
    
    #preprocess işlemini tüm veriye uygulamak için  map kullanacağız
    #map(method,ululanacağı parametre)
x_train = np.array(list(map(preProcess, x_train)))#map xtrainin hepsine preprocess uygular,listeye çevrilir ve array yapılır
x_test = np.array(list(map(preProcess, x_test)))#map xtextin hepsine preprocess uygular,listeye çevrilir ve array yapılır
x_validation = np.array(list(map(preProcess, x_validation)))#map xvalidationun hepsine preprocess uygular,listeye çevrilir ve array yapılır

x_train = x_train.reshape(-1,32,32,1)#-1 o parametreyi random atamasını sağlar
print(x_train.shape)#4064,32,32,1
x_test = x_test.reshape(-1,32,32,1)
x_validation = x_validation.reshape(-1,32,32,1)

# data generate
dataGen = ImageDataGenerator(width_shift_range = 0.2,# Bu parametre, veri büyütme sırasında görüntülerin yatay olarak rastgele kaydırılabileceği toplam genişliğin maksimum kısmı. görüntülerin toplam genişliklerinin %10'una kadar yatay olarak kaydırılabileceği anlamına gelir
                             height_shift_range = 0.2,#veri büyütme sırasında görüntülerin dikey olarak rastgele kaydırılabileceği toplam yüksekliğin maksimum kısmı. görüntülerin toplam yüksekliklerinin %10'una kadar dikey olarak kaydırılmasına olanak tanır
                             zoom_range = 0.2,#Bu parametre, büyütme sırasında görüntülerin rasgele yakınlaştırma aralığını belirtir. görüntülerin %10'a kadar yakınlaştırılabileceği veya uzaklaştırılabileceği anlamına gelir
                             rotation_range = 10)#Bu parametre, büyütme sırasında görüntülerin rastgele döndürülebileceği maksimum açıyı derece cinsinden ayarlar. görüntülerin hem saat yönünde hem de saat yönünün tersine 10 dereceye kadar döndürülmesine olanak tanır

dataGen.fit(x_train)# belirtilen veri büyütme param kullanılarak veri büyütmeye gerçekleştirmeye hazır hale getirir

#verileri kategorik hale getir[onehotencoder gibi]
y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)

model = Sequential()#temel layer oluştur
model.add(Conv2D(input_shape = (32,32,1), filters = 8, kernel_size = (5,5), activation = "relu", padding = "same"))#evrişim ağı ekle#input shape=genişlik yükseklik ve 1; padding piksel ekleme -> "same"=1sıra piksel ekleme
model.add(MaxPooling2D(pool_size = (2,2)))#piksel ekleme 2ye 2lik

model.add(Conv2D(filters = 16, kernel_size = (3,3), activation = "relu", padding = "same"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Dropout(0.2))#overfittingi engellemek için
model.add(Flatten())#düzleştirme
model.add(Dense(units=256, activation = "relu" ))#256 hücreli tam bağlantı katmanı
model.add(Dropout(0.2))#overfittingi engellemek için
model.add(Dense(units=noOfClasses, activation = "softmax" ))#çıktı(output layer)

#modeli compile at
model.compile(loss = "categorical_crossentropy", optimizer=("Adam"), metrics = ["accuracy"])
#loss fonk en son hataları hesaplamayı sağlar, başlangıçta hata çok çıkar geriye doğru türev algılama yapar ve bu değişime göre parametreleri düzenleriz
#optimizer: parametrelerimizi bulmamızı sağlayan
#metrics="accuracy" modelin sonuçlarını yüzde olarak değerlendirmemizi sağlar

batch_size = 250#batch size: resimlerimizin kaçlı grup halinde bir iterasyona sokulacağıdır bu tüm resimlerin dolaşması 1 epochs anlamına gelir

#eğitim aşaması hist;history
hist = model.fit_generator(dataGen.flow(x_train, y_train, batch_size = batch_size), 
                                        validation_data = (x_validation, y_validation),
                                        epochs = 35,steps_per_epoch = x_train.shape[0]//batch_size, shuffle = 1)#shuffle veriyi karıştırıyor

pickle_out = open("model_trained_new.p","wb")
pickle.dump(model, pickle_out)#pickle modeli depolayacak
pickle_out.close()

# %% degerlendirme
hist.history.keys()#model çıktılarını içerisinde barındırır hist


plt.figure()#boş grafik
plt.plot(hist.history["loss"], label = "Eğitim Loss")#eğitim kayıpları
plt.plot(hist.history["val_loss"], label = "Val Loss")#val kayıpları
plt.legend()#grafiği açıklayan 
plt.show()#göster

plt.figure()#boş grafik
plt.plot(hist.history["accuracy"], label = "Eğitim accuracy")
plt.plot(hist.history["val_accuracy"], label = "Val accuracy")
plt.legend()#grafiği açıklayan 
plt.show()#göster


score = model.evaluate(x_test, y_test, verbose = 1)#testin sonuçları hesaplama ; verborse=1 bunları görselleştir demek
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])


y_pred = model.predict(x_validation)#modelin eğitim sırasında görmediği girdi verileri için tahminelde eder
y_pred_class = np.argmax(y_pred, axis = 1)#max değerin indexi
Y_true = np.argmax(y_validation, axis = 1)#gerçek y değeri
cm = confusion_matrix(Y_true, y_pred_class)#ikisi arasındaki hangi clası doğru hesaplamışız
f, ax = plt.subplots(figsize=(8,8))#(eksen) içeren bir şekil oluşturur ve genel şekil boyutunu 8 x 8 olarak ayarlar. plt.subplots()figure nesnesini ve bir eksen nesneleri dizisini içeren bir demet döndürür.
sns.heatmap(cm, annot = True, linewidths = 0.01, cmap = "Greens", linecolor = "gray", fmt = ".1f", ax=ax)
#annot = True heatmapin üzerine sayılar yazsın
#fmt = ".1f": virgülden sonra 1 basamak
# ax=ax eksenler yazsın
plt.xlabel("predicted")#x eks
plt.ylabel("true")#y eks
plt.title("cm")
plt.show()