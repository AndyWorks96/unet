import numpy as np
import cv2
import scipy.io as sio
matfn = './data/754.mat'
data = sio.loadmat(matfn)







x_cor = data['pxy'][0]
y_cor = data['pxy'][1]
im = np.zeros([388,388], dtype="uint8")
cor_xy = np.hstack((x_cor, y_cor))
cv2.polylines(im, np.int32([cor_xy]), 1, 1)
cv2.fillPoly(im, np.int32([cor_xy]), 1)
mask_array = im
print(mask_array)