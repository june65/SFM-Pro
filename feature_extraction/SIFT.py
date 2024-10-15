import numpy as np
import cv2
from utils import RGB_to_gray

def SIFT(image):
    sift = cv2.SIFT_create(contrastThreshold=0.01, edgeThreshold=10)
    keypoint, descriptor = sift.detectAndCompute(image, None)
    return keypoint, descriptor

'''
def SIFT(image_1,image_2):
    gray_image_1 = RGB_to_gray(image_1)
    diff_conv_layers_1 = Difference_of_Gaussian(gray_image_1)
    keypoint_1 = Keypoint_location(diff_conv_layers_1)
    np.save('assets/feature_extraction_1.npy', keypoint_1)

    gray_image_2 = RGB_to_gray(image_2)
    diff_conv_layers_2 = Difference_of_Gaussian(gray_image_2)
    keypoint_2 = Keypoint_location(diff_conv_layers_2)
    np.save('assets/feature_extraction_2.npy', keypoint_2)

    return keypoint_1, keypoint_2

def Gaussian_filter(filter_size, sigma):
    pad_size = filter_size // 2
    gaussian_weight = np.zeros((filter_size, filter_size))
    for i in range(filter_size):
        for j in range(filter_size):
            gaussian_weight[i,j] = (1/(2*np.pi*(sigma**2))) * np.exp(-((i-pad_size)**2 + (j-pad_size)**2)/(2 * (sigma**2)))
    return gaussian_weight / np.sum(gaussian_weight)

def Gaussian_image(image, sigma, filter_size):
    height, width = image.shape
    pad_size = filter_size//2
    pad_width = ((pad_size, pad_size), (pad_size, pad_size))
    padded_image = np.pad(image, pad_width, mode='constant')
    result_image = np.zeros((height, width))
    gaussian_filter = Gaussian_filter(filter_size, sigma)
    for h in range(height):
        for w in range(width):
            result_image[h, w] = np.sum(gaussian_filter * padded_image[h:h+filter_size, w:w+filter_size])
    return result_image

def Difference_of_Gaussian(image):
    filter_size = 31
    gaussian_conv = []

    for i in range(4):
        gaussian_layer = []
        for j in range(5):
            sigma = 2 ** j
            result_image = Gaussian_image(image, sigma, filter_size)
            gaussian_layer.append(result_image)
                           
        image = cv2.resize(image, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        gaussian_conv.append(gaussian_layer)

    diff_gaussian_conv = []

    for i in range(4):
        diff_gaussian_layer = []
        for j in range(1,5):
            diff_gaussian = gaussian_conv[i][j] - gaussian_conv[i][j-1]
            diff_gaussian_layer.append(diff_gaussian)
            
            #print gaussian map
            max_val = np.max(diff_gaussian)
            print_img = ((diff_gaussian / max_val) * 255).astype(np.uint8)
            cv2.imshow('diff_gaussian',print_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        diff_gaussian_conv.append(diff_gaussian_layer)

    return diff_gaussian_conv

def Keypoint_location(layers):
    keypoint_in_conv = []
    
    for i in range(4):
        height, width = layers[i][0].shape
        for j in range(2):
            base_layer = np.zeros((height, width, 3))
            base_layer[:,:,0] = layers[i][j]
            base_layer[:,:,1] = layers[i][j+1]
            base_layer[:,:,2] = layers[i][j+2]
            base_max_layer = np.pad(base_layer, ((1, 1), (1, 1), (0, 0)), mode='constant')
            base_min_layer = np.pad(base_layer, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=np.inf)
            
            max_layer = np.zeros((height, width))
            min_layer = np.full((height, width),np.inf)
            for h in range(3):
                for w in range(3):
                    for k in range(3):
                        max_layer = np.maximum(max_layer, base_max_layer[h:h+height, w:w+width, k])
                        min_layer = np.minimum(min_layer, base_min_layer[h:h+height, w:w+width, k])

            max_layer[layers[i][j+1]!= max_layer] = 0
            min_layer[layers[i][j+1]!= min_layer] = 0

            sum_layer = np.zeros((height, width))
            sum_layer[(max_layer != 0) | (min_layer != 0)] = 1

            keypoint_in_conv.append(sum_layer)

    return keypoint_in_conv[0]
'''