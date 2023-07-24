import cv2
import numpy as np

# içe aktar resim
img = cv2.imread("C:\\Users\\Hazal\\2_OpenCV_ile_Goruntu_isleme\\7_warp_perspective\\resimler\\kart.png")
cv2.imshow("Orijinal", img)

width = 400
height = 500

pts1 = np.float32([[230,1],[1,472],[540,150],[338,617]])
pts2 = np.float32([[0,0],[0, height],[width,0],[width,height]])
#pts1 kaynak görüntüdeki dört noktayı içerir ve pts2 istenen çıktı görüntüsündeki karşılık gelen dört noktayı içerir.


matrix = cv2.getPerspectiveTransform(pts1, pts2)
print(matrix)

# dönüştürülmüş resim
imgOutput = cv2.warpPerspective(img, matrix, (width,height))
cv2.imshow("cikis resmi", imgOutput)

k = cv2.waitKey(0) &0xFF

if k == 27: # esc
    cv2.destroyAllWindows()