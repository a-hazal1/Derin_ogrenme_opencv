import cv2
import matplotlib.pyplot as plt

# karıştırma
img1 = cv2.imread("C:\\Users\\Hazal\\2_OpenCV_ile_Goruntu_isleme\\8_blending\\resimler\\img1.JPG")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread("C:\\Users\\Hazal\\2_OpenCV_ile_Goruntu_isleme\\8_blending\\resimler\\img2.JPG")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

plt.imshow(img1)
plt.show()

plt.imshow(img2)
plt.show()

print(img1.shape)
print(img2.shape)

img1 = cv2.resize(img1, (600,600))
print(img1.shape)

img2 = cv2.resize(img2, (600,600))
print(img2.shape)


plt.imshow(img1)
plt.show()

plt.imshow(img2)
plt.show()

# karıştırılmış resim = alpha*img1 + beta*img2
blended = cv2.addWeighted(src1 = img1, alpha =0.5, src2= img2, beta = 0.5, gamma = 0)

plt.imshow(blended)
plt.show()















