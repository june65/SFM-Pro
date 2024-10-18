import argparse
import numpy as np
from tqdm import tqdm
from utils import ImageLoader
from feature_extraction import SIFT
from feature_matching import BF, FLANN
from essential_matrix import FivePoint, CameraMatrix
from triangulation import Triangulation, Triangulation_G
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
    fivepoint_threshold = 2.0e-4
    fivepoint_max_iter = 2000

    #Growing Step
    threepoint_threshold =  1.0e-3
    threepoint_max_iter = 5000#5000

    
    print('---------------------#0 Feature Extraction---------------------')
    for i in tqdm(range(5)):
        images.append(imageset[i])
        keypoint, descriptor =SIFT(images[i])
        keypoints.append(keypoint)
        descriptors.append(descriptor)

    First = 2
    Second = 3
    print('---------------------#1 Feature Matching---------------------')
    if Matching_method == "FLANN":
        matches, keypoint_1M, keypoint_2M, camerapoint_1M, camerapoint_2M = FLANN(Matching_method, keypoints[First], keypoints[Second], descriptors[First], descriptors[Second], images[First], images[Second], K_inv)
    else:
        matches, keypoint_1M, keypoint_2M, camerapoint_1M, camerapoint_2M = BF(Matching_method, threshold_knn, keypoints[First], keypoints[Second], descriptors[First], descriptors[Second], images[First], images[Second], K_inv)
    
    print('---------------------#2 FivePoint Algorithm---------------------')
    E_matrix, inlinear_TF, inlinear = FivePoint(matches, camerapoint_1M, camerapoint_2M, fivepoint_threshold, fivepoint_max_iter)
    
    print('---------------------#3 Camera Matrix---------------------')
    initial_camera_matrix = CameraMatrix(E_matrix, camerapoint_1M, camerapoint_2M, inlinear_TF)

    print('---------------------#4 Triangulation---------------------')
    initial_point, initial_color, initial_inlinear = Triangulation(images[First], camerapoint_1M, camerapoint_2M, keypoint_1M, keypoint_2M, initial_camera_matrix, inlinear_TF, inlinear)
    
    print('---------------------#5 ThreePoint Algorithm---------------------')
    Third = 4
    if Matching_method == "FLANN":
        next_matches, next_keypoint_1M, next_keypoint_2M, next_camerapoint_1M, next_camerapoint_2M = FLANN(Matching_method, keypoints[Second], keypoints[Third], descriptors[Second], descriptors[Third], images[Second], images[Third], K_inv)
    else:
        next_matches, next_keypoint_1M, next_keypoint_2M, next_camerapoint_1M, next_camerapoint_2M = BF(Matching_method, threshold_knn, keypoints[Second], keypoints[Third], descriptors[Second], descriptors[Third], images[Second], images[Third], K_inv)
    
    new_camera_matrix = ThreePoint(matches, next_matches, initial_inlinear, initial_point, next_camerapoint_2M, threepoint_threshold, threepoint_max_iter)
    
    next_point, next_color, next_inlinear = Triangulation_G(images[Second], next_camerapoint_1M, next_camerapoint_2M, next_keypoint_1M, next_keypoint_2M, new_camera_matrix, initial_camera_matrix)

if __name__ == "__main__":
    main()