import numpy as np
import cv2
from utils import RGB_to_gray

def FLANN(method, keypoint_1, keypoint_2, descriptor_1, descriptor_2, image_1, image_2, K_inv):

    index_params = dict(algorithm=cv2.FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)

    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptor_1, descriptor_2, k=2)

    keypoint_1M, keypoint_2M, camerapoint_1M, camerapoint_2M = Camera_coordinate(keypoint_1, keypoint_2, matches, K_inv)

    img_matches = cv2.drawMatches(image_1, keypoint_1, image_2, keypoint_2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    resized_img = cv2.resize(img_matches, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('BF_matches',resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return np.array(matches), keypoint_1M, keypoint_2M, camerapoint_1M, camerapoint_2M


def Camera_coordinate(keypoint_1, keypoint_2, matches, K_inv):

    query_idx = [match.queryIdx for match in matches]
    train_idx = [match.trainIdx for match in matches]
    good_kp1 = np.float32([keypoint_1[ind].pt for ind in query_idx])
    good_kp2 = np.float32([keypoint_2[ind].pt for ind in train_idx])
    keypoint_1M = np.array(good_kp1).reshape(-1, 2)
    keypoint_2M = np.array(good_kp2).reshape(-1, 2)

    camerapoint_1M = keypoint_1M.T
    camerapoint_1M = np.append(camerapoint_1M, np.ones((1, len(matches))), axis=0) 
    camerapoint_1M = K_inv @ camerapoint_1M  

    camerapoint_2M = keypoint_2M.T
    camerapoint_2M = np.append(camerapoint_2M, np.ones((1, len(matches))), axis=0) 
    camerapoint_2M = K_inv @ camerapoint_2M  

    return keypoint_1M, keypoint_2M, camerapoint_1M, camerapoint_2M
