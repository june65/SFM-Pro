import cv2

def ORB(image):
    orb = cv2.ORB_create()
    keypoint, descriptor = orb.detectAndCompute(image, None)
    '''
    image_keypoints = cv2.drawKeypoints(image, keypoint, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("ORB Keypoints", image_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    return keypoint, descriptor
