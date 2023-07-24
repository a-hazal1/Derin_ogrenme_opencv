import cv2
import matplotlib.pyplot as plt

# template matching: sablon esleme; kaydırarak şablonu bir seferde bir piksel hareket ettirmektir
# (soldan sağa yukarıdan aşağıya)

img = cv2.imread("C:\\Users\\Hazal\\Derin_ogrenme_ile_goruntu_isleme\\OpenCV_ile_Nesne_Tespiti\\5_template_matching\\resim\\cat.jpg", 0)# resmi içe aktar
print(img.shape)
template = cv2.imread("C:\\Users\\Hazal\\Derin_ogrenme_ile_goruntu_isleme\\OpenCV_ile_Nesne_Tespiti\\5_template_matching\\resim\\cat_face.jpg", 0)# resmi içe aktar
print(template.shape)

h, w = template.shape

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']#bu 6 ana methodun amacı iki resim arasındaki korelasyonu çıkarmaktır.


for meth in methods:
    
    method = eval(meth) #stringleri normal hallerine çeviriyor # 'cv2.TM_CCOEFF' -> cv2.TM_CCOEFF
    
    res = cv2.matchTemplate(img, template, method)
    print(res.shape)#orijinal resimle aynı olmalı
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:#bu iki methodda min loc top lefte karşılık geliyor
        top_left = min_loc
    else:#diğerlerinde ise top left max loc a denk geliyor
        top_left = max_loc
    
    bottom_right = (top_left[0] + w, top_left[1] + h)#kutucuğu belirlemek için gerekli olan w ve h
    
    cv2.rectangle(img, top_left, bottom_right, 255, 2)#kutucuğu çiz
    
    plt.figure()
    plt.subplot(121)#1satır 2 sutün ve 1.sini al
    plt.imshow(res, cmap = "gray")
    plt.title("Eşleşen Sonuç"), plt.axis("off")
    plt.subplot(122)#1satır 2 sutün ve 2.sini al
    plt.imshow(img, cmap = "gray")
    plt.title("Tespit edilen Sonuç"), plt.axis("off")
    plt.suptitle(meth)
    
    plt.show()
    
    
    
 
    
    
    
    