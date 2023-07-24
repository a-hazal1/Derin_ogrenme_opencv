import cv2

# 
img = cv2.imread("C:\\Users\\Hazal\\2_OpenCV_ile_Goruntu_isleme\\4_resize_crop\\resimler\\lenna.png")
print("Resim boyutu: ", img.shape)
cv2.imshow("Orijinal", img)

# resized
imgResized = cv2.resize(img, (800,800))
print("Resized Img Shape: ", imgResized.shape)
cv2.imshow("Img Resized",imgResized)

# kırp
imgCropped = img[:200,:300] # width height -> height width 
cv2.imshow("Kirpilmis Resim",imgCropped)

k = cv2.waitKey(0) &0xFF

if k == 27: # esc
    cv2.destroyAllWindows()
