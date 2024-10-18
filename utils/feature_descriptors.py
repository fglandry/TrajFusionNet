import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import time

from skimage.feature import hog


def get_hu_moments(img_data: np.array) -> np.array:
    _, im = cv2.threshold(img_data.astype(np.uint8), 128, 255, cv2.THRESH_BINARY)
    #cv2.imwrite(f"/home/francois/MASTER/sem_imgs/sem_output_{str(time.time()).replace('.', '_')}.png", im)  
    moments = cv2.moments(im) 
    # Calculate Hu Moments 
    huMoments = cv2.HuMoments(moments)
    # Log scale hu moments 
    for i in range(0,7):
        try:
            huMoments[i] = -1* math.copysign(1.0, huMoments[i]) * \
                math.log10(abs(huMoments[i]))
        except ValueError:
            huMoments[i] = 0 # revisit
    #if not (huMoments == 0).all():
    #    test = 4
    return huMoments.squeeze()

def get_hog_features(img_data: np.array, debug=False) -> np.array:
    fd, hog_image = hog(img_data, orientations=9, pixels_per_cell=(16, 16),
                	    cells_per_block=(2, 2), visualize=True, channel_axis=2)
    if debug:
        cv2.imwrite(f"/home/francois/MASTER/sem_imgs/org_output_{str(time.time()).replace('.', '_')}.png", img_data)  
        cv2.imwrite(f"/home/francois/MASTER/sem_imgs/hog_output_{str(time.time()).replace('.', '_')}.png", hog_image)  
        plt.axis("off")
        plt.imshow(hog_image, cmap="gray")
        plt.show()

    return fd
