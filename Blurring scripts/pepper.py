import numpy as np
import random
import cv2
import os

def salt_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            # if rdn > thres:
            #     output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

i = 1

for filename in os.listdir('C:/Users/pawel/Documents/MAGISTERKA/Data/HighBlur/Set-III/clear'):
    image = cv2.imread('C:/Users/pawel/Documents/MAGISTERKA/Data/HighBlur/Set-III/clear/'+ filename)
    noise_img = salt_noise(image,0.4)
    print(i)
    cv2.imwrite('C:/Users/pawel/Documents/MAGISTERKA/Data/HighBlur/Set-III/pepper/' +  filename, noise_img)
    i+=1