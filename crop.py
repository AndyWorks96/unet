import cv2

img = cv2.imread("./data/png/754.png")
print(img.shape)
cropped = img[32:224, 32:224]  # 裁剪坐标为[y0:y1, x0:x1]
cv2.imwrite("./data/png/754_crop.png", cropped)
