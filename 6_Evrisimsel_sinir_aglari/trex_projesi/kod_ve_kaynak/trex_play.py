from keras.models import model_from_json
import numpy as np
from PIL import Image
import keyboard
import time
from mss import mss

mon = {"top":300, "left":770, "width":250, "height":100}#sözlük yaratıldı;piksel formatı
sct = mss()#mss kütüphanesi ekrandan ilgili alanı kesip frame haline dönüştürür

width = 125#genişlik
height = 50#yükseklik

# model yükle
model = model_from_json(open("C:\\Users\\Hazal\\Derin_ogrenme_ile_goruntu_isleme\\6_Evrisimsel_sinir_aglari\\1_trex_projesi\\kod_ve_kaynak\\model.json","r").read())
model.load_weights("C:\\Users\\Hazal\\Derin_ogrenme_ile_goruntu_isleme\\6_Evrisimsel_sinir_aglari\\1_trex_projesi\\kod_ve_kaynak\\trex_weight.h5")

# down = 0, right = 1, up = 2
labels = ["Down", "Right", "Up"]

framerate_time = time.time()#zamanı tutacağımız değişken
counter = 0#sayıcı
i = 0
delay = 0.4#komut geldikten hemen sonra başka bir komut vermesi için 0.4 sn beklemesi için
key_down_pressed = False
while True:
    
    img = sct.grab(mon)#ekranı al,mon değişkeni doğrultusunda 
    im = Image.frombytes("RGB", img.size, img.rgb)#rgb formatinda img.size a göre oku
    im2 = np.array(im.convert("L").resize((width, height)))#imgleri 125 e 50 boyutuna getiriyoruz
    im2 = im2 / 255 #normalize etmek için 255 e böldük
    
    X =np.array([im2])#resmi array haline getir
    X = X.reshape(X.shape[0], width, height, 1)#keras formatına göre ;#x.shape: kac tane resim olduğu
    r = model.predict(X)#modeli kullanarak prediction işlemi gerçekleştirir
    #r: toplamı 1 olan 3 sayıdan oluşur
    result = np.argmax(r)#en yüksek sayıyının indexini döndürür
    
    #right çıkıyorsa hiçbir şey yapmama olduğu için sadece down ve up inceleniyor
    
    if result == 0: # down = 0
        
        keyboard.press(keyboard.KEY_DOWN)#aşağı tuşuna basmak
        key_down_pressed = True
        
    elif result == 2:    # up = 2
        
        if key_down_pressed:#eğer öncesinde aşağı tuşuna basılmış ise 
            keyboard.release(keyboard.KEY_DOWN)#aşağı tuşunu bırak
        time.sleep(delay)#bekle
        keyboard.press(keyboard.KEY_UP)#yukarı tuşuna basmak
        
        if i < 1500:
            time.sleep(0.3)
        elif 1500 < i and i < 5000:
            time.sleep(0.2)#hızlandığı için yukarıda daha az zaman geçirmeli
        else:
            time.sleep(0.17)
            
        keyboard.press(keyboard.KEY_DOWN)
        keyboard.release(keyboard.KEY_DOWN)#aşağı tuşunu bırak
    
    counter += 1
    
    if (time.time() - framerate_time) > 1:#eğer şu anki zamanımız ile oyun başladığı zaman farkı 1 den büyükse
        
        counter = 0
        framerate_time = time.time()#zamanı tutacağın değişkeni güncelle
        if i <= 1500:
            delay -= 0.003#beklemeyi azalt
        else:
            delay -= 0.005#beklemeyi azalt
        if delay < 0:#dlay 0dan küçük olamaz
            delay = 0
            
        print("---------------------")
        print("Down: {} \\nRight:{} \\nUp: {} \\n".format(r[0][0],r[0][1],r[0][2]))#0 aşağı 1 sağa 2 yukarı
        i += 1
        

