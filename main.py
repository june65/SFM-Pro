import argparse
import numpy as np
from utils import ImageLoader
from feature_extraction import SIFT
from feature_matching import BF, FLANN
from essential_matrix import FivePoint, CameraMatrix
from triangulation import Triangulation
parser = argparse.ArgumentParser()

#Data parameters
parser.add_argument('--dataset', default='data_2_sfm', help='data name')

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
    for i in range(2):
        images.append(imageset[i])
        keypoint, descriptor =SIFT(images[i])
        keypoints.append(keypoint)
        descriptors.append(descriptor)

    for i in range(1):
        for j in range(i+1,2):
            print('---------------------1 Feature Matching---------------------')
            if Matching_method == "FLANN":
                matches, keypoint_1M, keypoint_2M, camerapoint_1M, camerapoint_2M = FLANN(Matching_method, keypoints[i], keypoints[j], descriptors[i], descriptors[j], images[i], images[j], K_inv)
            else:
                matches, keypoint_1M, keypoint_2M, camerapoint_1M, camerapoint_2M = BF(Matching_method, threshold_knn, keypoints[i], keypoints[j], descriptors[i], descriptors[j], images[i], images[j], K_inv)
            
            print('---------------------#2 FivePoint Algorithm---------------------')
            E_matrix = FivePoint(matches, camerapoint_1M, camerapoint_2M, threshold, max_iter)
            
            print('---------------------#3 Camera Matrix---------------------')
            camera_matrix = CameraMatrix(E_matrix, camerapoint_1M, camerapoint_2M)

            print('---------------------#4 Triangulation---------------------')
            initial_geometry = Triangulation(images[i], camerapoint_1M, camerapoint_2M, keypoint_1M, keypoint_2M, camera_matrix)

if __name__ == "__main__":
    main()