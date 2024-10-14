import numpy as np
import cv2
from utils import RGB_to_gray

def BF(method, threshold_knn, keypoint_1, keypoint_2, descriptor_1, descriptor_2, image_1, image_2, K_inv):

    if method == "NORM":
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(descriptor_1, descriptor_2)
        matches = sorted(matches, key=lambda x: x.distance)

    if method == "KNN":
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptor_1, descriptor_2, k=2)
        good = []
        good_print = []
        for m,n in matches:
            if m.distance < threshold_knn*n.distance:
                good.append(m)
                good_print.append([m])
        good = sorted(good, key=lambda x: x.distance)
        matches = good
        
    print('Matches Number :', len(matches))

    keypoint_1M, keypoint_2M, camerapoint_1M, camerapoint_2M = Camera_coordinate(keypoint_1, keypoint_2, matches, K_inv)
    '''
    img_matches = cv2.drawMatches(image_1, keypoint_1, image_2, keypoint_2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    resized_img = cv2.resize(img_matches, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('BF_matches',resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
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
