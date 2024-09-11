import numpy as np
import cv2
from utils import RGB_to_gray

def BF(keypoint_1, keypoint_2, descriptor_1, descriptor_2, image_1, image_2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptor_1, descriptor_2)
    matches = sorted(matches, key=lambda x: x.distance)

    keypoint_1M = np.float64([keypoint_1[m.queryIdx].pt for m in matches])
    keypoint_2M = np.float64([keypoint_2[m.trainIdx].pt for m in matches])
    '''
    img_matches = cv2.drawMatches(image_1, keypoint_1, image_2, keypoint_2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    resized_img = cv2.resize(img_matches, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('BF_matches',resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    return np.array(matches), keypoint_1M, keypoint_2M
