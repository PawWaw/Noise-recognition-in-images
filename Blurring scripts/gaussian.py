import cv2
import numpy as np
from skimage.util import random_noise
import os

i = 1

for filename in os.listdir('C:/Users/pawel/Documents/MAGISTERKA/Data/HighBlur/Set-III/clear'):
    img = cv2.imread('C:/Users/pawel/Documents/MAGISTERKA/Data/HighBlur/Set-III/clear/' + filename)
    
    noise_img = random_noise(img, mode='gaussian', mean=0, var=0.4)
    noise_img = np.array(255*noise_img, dtype = 'uint8')

    # Display the noise image
    print(i)
    cv2.imwrite('C:/Users/pawel/Documents/MAGISTERKA/Data/HighBlur/Set-III/gaussian/' +  filename, noise_img)
    i+=1