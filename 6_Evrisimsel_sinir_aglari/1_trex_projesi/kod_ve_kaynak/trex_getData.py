import keyboard
import uuid
import time
from PIL import Image
from mss import mss

"""
http://www.trex-game.skipser.com/
"""

mon = {"top":300, "left":770, "width":250, "height":100}#sözlük yaratıldı;piksel formatı
sct = mss()#mss kütüphanesi ekrandan ilgili alanı kesip frame haline dönüştürür
#mss başlatıldı
i = 0

def record_screen(record_id, key):#key dediğimiz şey klavyede bastığımız şey
    global i#klavyeye kaç kez bastığımız
    
    i += 1#i ye 1 ekle ve i ye eşitle
    print("{}: {}".format(key, i))
    img = sct.grab(mon)#ekranı al,mon değişkeni doğrultusunda 
    im = Image.frombytes("RGB", img.size, img.rgb)#rgb formatinda img.size a göre oku
    im.save("./img/{}_{}_{}.png".format(key, record_id, i))#kaydedeceğimiz resmin ismini verdik
    
is_exit = False#fonksiyondan çıkmayı sağlayacak

def exit():
    global is_exit#içeriden çağırabilmek için
    is_exit = True
    
keyboard.add_hotkey("esc", exit)#çıkmak için esc tuşuna basınca call back olarak exit fonkiyonunu çağıracak

record_id = uuid.uuid4()

while True:
    
    if is_exit==True : break#çıkılıpp çıkılmadığı kontrol edilecek

    try:
        if keyboard.is_pressed(keyboard.KEY_UP):#yukarı tuşuna basıldıysa
            record_screen(record_id, "up")#ekrani kaydetme fonk alıcak
            time.sleep(0.1)#her bir komut sonrasında 0.1 saniye beklemeli yapmadığımız durumlarda çok hızlanıyor
        elif keyboard.is_pressed(keyboard.KEY_DOWN):#aşağı tuşuna basıldıysa
            record_screen(record_id, "down")#ekrani kaydetme fonk alıcak
            time.sleep(0.1)#her bir komut sonrasında 0.1 saniye beklemeli yapmadığımız durumlarda çok hızlanıyor
        elif keyboard.is_pressed("right"):#sağ tuşuna basıldıysa
            record_screen(record_id, "right")#ekrani kaydetme fonk alıcak
            time.sleep(0.1)#her bir komut sonrasında 0.1 saniye beklemeli yapmadığımız durumlarda çok hızlanıyor
    except RuntimeError: continue 
            #try yapıyı dener hata oluşursa kodun çalışmasını devam ettir























