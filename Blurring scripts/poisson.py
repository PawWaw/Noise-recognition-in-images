import cv2
import numpy as np
from skimage.util import random_noise
import os

i = 1
PEAK = 0.9

for filename in os.listdir('C:/Users/pawel/Documents/MAGISTERKA/Data/HighBlur/Set-III/clear'):
    img = cv2.imread('C:/Users/pawel/Documents/MAGISTERKA/Data/HighBlur/Set-III/clear/' + filename)

    noisy = np.random.poisson(img / 255.0 * PEAK) / PEAK * 255

    print(i)
    cv2.imwrite('C:/Users/pawel/Documents/MAGISTERKA/Data/HighBlur/Set-III/poisson/' +  filename, noisy)
    i+=1