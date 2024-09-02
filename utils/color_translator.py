import numpy as np
import cv2

def RGB_to_gray(image):
    if len(image.shape) == 3:
        return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])