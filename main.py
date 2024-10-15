import argparse
import numpy as np
from utils import ImageLoader
from feature_extraction import SIFT
from feature_matching import BF, FLANN
from essential_matrix import FivePoint, CameraMatrix
from triangulation import Triangulation
from growing_step import ThreePoint
parser = argparse.ArgumentParser()

#Data parameters
parser.add_argument('--dataset', default='data_30', help='data name')

args = parser.parse_args()

datapath = "./data/" + args.dataset+ "/"

def main():
    #Data loading
    imageset = ImageLoader(datapath)
    images = []
    K = np.loadtxt(datapath+'K.txt')
    K = np.array(K)
    K_inv = np.linalg.inv(K)

    #Feature Extraction
    keypoints = []
    descriptors = []

    #Feature Matching
    Matching_method = "KNN" #NORM, KNN, FLANN
    threshold_knn = 0.85
    kdtree_flann = 1

    #Essential Matrix
    threshold = 2.0e-4
    max_iter = 1000
    
    #Feature Extraction
    for i in range(5):
        images.append(imageset[i])
        keypoint, descriptor =SIFT(images[i])
        keypoints.append(keypoint)
        descriptors.append(descriptor)

    First = 2
    Second = 3
    print('---------------------1 Feature Matching---------------------')
    if Matching_method == "FLANN":
        matches, keypoint_1M, keypoint_2M, camerapoint_1M, camerapoint_2M = FLANN(Matching_method, keypoints[First], keypoints[Second], descriptors[First], descriptors[Second], images[First], images[Second], K_inv)
    else:
        matches, keypoint_1M, keypoint_2M, camerapoint_1M, camerapoint_2M = BF(Matching_method, threshold_knn, keypoints[First], keypoints[Second], descriptors[First], descriptors[Second], images[First], images[Second], K_inv)
    
    print('---------------------#2 FivePoint Algorithm---------------------')
    E_matrix, inlinear_TF, inlinear = FivePoint(matches, camerapoint_1M, camerapoint_2M, threshold, max_iter)
    
    print('---------------------#3 Camera Matrix---------------------')
    camera_matrix = CameraMatrix(E_matrix, camerapoint_1M, camerapoint_2M)

    print('---------------------#4 Triangulation---------------------')
    initial_point, initial_color, new_inlinear = Triangulation(images[First], camerapoint_1M, camerapoint_2M, keypoint_1M, keypoint_2M, camera_matrix, inlinear_TF, inlinear)

    print('---------------------#5 ThreePoint Algorithm---------------------')
    Third = 4
    if Matching_method == "FLANN":
        matches, keypoint_1M, keypoint_2M, camerapoint_1M, camerapoint_2M = FLANN(Matching_method, keypoints[Second], keypoints[Third], descriptors[Second], descriptors[Third], images[Second], images[Third], K_inv)
    else:
        matches, keypoint_1M, keypoint_2M, camerapoint_1M, camerapoint_2M = BF(Matching_method, threshold_knn, keypoints[Second], keypoints[Third], descriptors[Second], descriptors[Third], images[Second], images[Third], K_inv)
    
if __name__ == "__main__":
    main()