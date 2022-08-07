import cv2
import numpy as np
from skimage.util import random_noise
import os

i = 1

for filename in os.listdir('C:/Users/pawel/Documents/MAGISTERKA/Data/Set-I/clear'):
    img = cv2.imread('C:/Users/pawel/Documents/MAGISTERKA/Data/Set-I/clear/' + filename)

    noise_img = random_noise(img, mode='localvar')
    noise_img = np.array(255*noise_img, dtype = 'uint8')

    print(i)
    cv2.imwrite('C:/Users/pawel/Documents/MAGISTERKA/Data/Set-I/localvar/' +  filename, noise_img)
    i+=1