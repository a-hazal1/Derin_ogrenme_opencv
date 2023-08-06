import glob
import os
import numpy as np
from keras.models import Sequential #derin öğrenme alg tasarlanması ve eğitimi gerçekleştirilir
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D 
#dense: tam bağlantıdaki katmanlar
#dropout: seyreltme
#flatten: düzleştirme
#Conv2D: evrişim ağı
#MaxPooling2D: piksel ekleme
from PIL import Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder: 1,2,3,4 diye etiket labellar eklicek 
# OneHotEncoder: kerasta eğitilebilir hale getirecek dönüşümü yapar
from sklearn.model_selection import train_test_split#veri setimizi eğitim ve test olarak ikiye ayıracağız
import seaborn as sns #basit bir görselleme işlemi

import warnings# uyarıları kapatmak için 
warnings.filterwarnings("ignore")#uyarıları kapattık

#resimlerimizi yükleyelim
imgs = glob.glob("C:\\Users\\Hazal\\Derin_ogrenme_ile_goruntu_isleme\\6_Evrisimsel_sinir_aglari\\1_trex_projesi\\img_nihai\\*.png")#ismi ne olursa olsun tüm png uzantıları al

ArithmeticError(
    )
width = 125#genişlik
height = 50#yükseklik
#liste oluştur x ve y değerleri için
X = []
Y = []

for img in imgs:
    
    filename = os.path.basename(img)#resmin ismini al
    label = filename.split("_")[0]#altçizgiye göre ayır ve 0.(ilk)indisi al down up rigt bulunmaya calışcak
    im = np.array(Image.open(img).convert("L").resize((width, height)))#imgleri 125 e 50 boyutuna getiriyoruz
    im = im / 255 #normalize etmek için 255 e böldük
    X.append(im)#resimlerimizi ekledik
    Y.append(label)#resimlere ait labelları ekledik
    
X = np.array(X)# x arraye dönüştürüldü
X = X.reshape(X.shape[0], width, height, 1)#x.shape: kac tane resim olduğu

# sns.countplot(Y)#kaç tane y olduğunu grafik şeklinde gösterir
#keras sayıları binary olarak alır
def onehot_labels(values):
    label_encoder = LabelEncoder()#laberencoderi çalıştır
    integer_encoded = label_encoder.fit_transform(values)#ilk önce ne yapacağını öğreniyor sonra dönüştürüyor
    onehot_encoder = OneHotEncoder(sparse = False)#sparse matris elde etmek istemediğimiz için
    integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)#ilk önce ne yapacağını öğreniyor sonra dönüştürüyor
    return onehot_encoded
#0 için 100
#1 için 010
#2 için 001
Y = onehot_labels(Y)
train_X, test_X, train_y, test_y = train_test_split(X, Y , test_size = 0.25, random_state = 2)  #x y arraylari, test veri seti yüzde 25, random state: bölünme 

# cnn model
model = Sequential() #layerlar üzerine eklenecek temel yapı
model.add(Conv2D(32, kernel_size = (3,3), activation = "relu", input_shape = (width, height, 1)))#evrişim ağı ekle#input shape=genişlik yükseklik ve 1
model.add(Conv2D(64, kernel_size = (3,3), activation = "relu"))#evrişim ağı ekle
model.add(MaxPooling2D(pool_size = (2,2)))# piksel ekleme
model.add(Dropout(0.25))#seyreltme yap(yüzde 25 i kaybolsun)
model.add(Flatten())#düzleştirme sınıflandırma kısmına geçiş yapacağımız için
model.add(Dense(128, activation = "relu"))#tam bağlantı
model.add(Dropout(0.4))#seyreltme
model.add(Dense(3, activation = "softmax"))#output layer

# if os.path.exists("C:\\Users\Hazal\\Derin_ogrenme_ile_goruntu_isleme\\6_Evrisimsel_sinir_aglari\\1_trex_projesi\\trex_weight.h5"):
#     model.load_weights("trex_weight.h5")
#     print("Weights yuklendi")    

model.compile(loss = "categorical_crossentropy", optimizer = "Adam", metrics = ["accuracy"])#modeli compile etmek için
#loss fonk en son hataları hesaplamayı sağlar, başlangıçta hata çok çıkar geriye doğru türev algılama yapar ve bu değişime göre parametreleri düzenleriz
#optimizer: parametrelerimizi bulmamızı sağlayan
#metrics="accuracy" modelin sonuçlarını yüzde olarak değerlendirmemizi sağlar

#training işlemi
model.fit(train_X, train_y, epochs = 35, batch_size = 64)#epochs: resimlerin toplamda kaç kez eğitileceği
#batch size: resimlerimizin kaçlı grup halinde bir iterasyona sokulacağıdır bu tüm resimlerin dolaşması 1 epochs anlamına gelir

score_train = model.evaluate(train_X, train_y)#eğitim sonucu
print("Eğitim doğruluğu: %",score_train[1]*100)#eğitim doğruluğunu yazdır 
    
score_test = model.evaluate(test_X, test_y)#test sonucu
print("Test doğruluğu: %",score_test[1]*100)#test doğruluğunu yazdır 
          

# open("model_new.json","w").write(model.to_json())
# model.save_weights("trex_weight_new.h5")   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    