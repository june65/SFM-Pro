import argparse
import numpy as np
from tqdm import tqdm
from utils import ImageLoader
from feature_extraction import SIFT
from feature_matching import BF, FLANN
from essential_matrix import FivePoint, CameraMatrix
from triangulation import Triangulation, Triangulation_G, Points_visual, Save_Camera
from growing_step import ThreePoint, Bundle, Noise_Bundle
parser = argparse.ArgumentParser()

#Data parameters
parser.add_argument('--dataset', default='mydata2', help='data name')

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
    threshold_knn = 0.75 #0.85 
    kdtree_flann = 1

    #Essential Matrix
    fivepoint_threshold = 2.0e-4
    fivepoint_max_iter = 2000

    #Growing Step
    threepoint_threshold = 2.0e-8
    threepoint_max_iter = 2000
    triangulation_threshold_E = 2.0e-4
    triangulation_threshold = 1 #2.0e-1
    
    print('---------------------#0 Feature Extraction---------------------')
    for i in tqdm(range(5)):
        images.append(imageset[i])
        keypoint, descriptor =SIFT(images[i])
        keypoints.append(keypoint)
        descriptors.append(descriptor)

    all_points = []
    all_colors = []
    all_point3d_idx = []
    all_camera_matrix = [np.eye(4)[:3]]
    all_keypoint1 = []
    all_keypoint2 = []
    all_identical_points = []

    #First = 31
    #Second = 30
    First = 0
    Second = 1

    print('---------------------#1 Feature Matching---------------------')
    if Matching_method == "FLANN":
        initial_matches, keypoint_1M, keypoint_2M, camerapoint_1M, camerapoint_2M = FLANN(Matching_method, keypoints[First], keypoints[Second], descriptors[First], descriptors[Second], images[First], images[Second], K_inv)
    else:
        initial_matches, keypoint_1M, keypoint_2M, camerapoint_1M, camerapoint_2M = BF(Matching_method, threshold_knn, keypoints[First], keypoints[Second], descriptors[First], descriptors[Second], images[First], images[Second], K_inv)
    
    print('---------------------#2 FivePoint Algorithm---------------------')
    E_matrix, inlinear_TF, inlinear = FivePoint(initial_matches, camerapoint_1M, camerapoint_2M, fivepoint_threshold, fivepoint_max_iter)
    
    print('---------------------#3 Camera Matrix---------------------')
    initial_camera_matrix = CameraMatrix(E_matrix, camerapoint_1M, camerapoint_2M, inlinear_TF)

    print('---------------------#4 Triangulation---------------------')
    initial_point, initial_color, initial_inlinear, initial_point3d_idx = Triangulation(images[First], camerapoint_1M, camerapoint_2M, keypoint_1M, keypoint_2M, initial_camera_matrix, inlinear_TF, inlinear)
    
    all_points.append(initial_point)
    all_colors.append(initial_color)
    all_point3d_idx.append(initial_point3d_idx)
    all_camera_matrix.append(initial_camera_matrix)
    all_keypoint1.append(keypoint_1M)
    all_keypoint2.append(keypoint_2M)

    #Points_visual(all_points, all_colors, all_point3d_idx, all_keypoint1, "Before_Bundle_noise")
    #new_all_points, new_camera_points = Noise_Bundle(all_points, all_point3d_idx, all_camera_matrix, all_keypoint1, all_keypoint2, all_identical_points, K)
    #Points_visual(new_all_points, all_colors, all_point3d_idx, all_keypoint1, "After_Bundle_noise")
    
    print('---------------------#5 ThreePoint Algorithm---------------------')
    
    orders = [2,3,4]
    for Third in orders:
        print('Matched images :', Second,Third)
        
        if Matching_method == "FLANN":
            next_matches, next_keypoint_1M, next_keypoint_2M, next_camerapoint_1M, next_camerapoint_2M = FLANN(Matching_method, keypoints[Second], keypoints[Third], descriptors[Second], descriptors[Third], images[Second], images[Third], K_inv)
        else:
            next_matches, next_keypoint_1M, next_keypoint_2M, next_camerapoint_1M, next_camerapoint_2M = BF(Matching_method, threshold_knn, keypoints[Second], keypoints[Third], descriptors[Second], descriptors[Third], images[Second], images[Third], K_inv)
        
        new_camera_matrix, identical_points = ThreePoint(initial_matches, next_matches, initial_inlinear, initial_point, next_camerapoint_2M, threepoint_threshold, threepoint_max_iter)
        
        next_point, next_color, next_point3d_idx = Triangulation_G(images[Second], next_camerapoint_1M, next_camerapoint_2M, next_keypoint_1M, next_keypoint_2M, new_camera_matrix, initial_camera_matrix, triangulation_threshold_E, triangulation_threshold)
        
        all_points.append(next_point)
        all_colors.append(next_color)
        all_point3d_idx.append(next_point3d_idx)
        all_camera_matrix.append(new_camera_matrix)
        all_keypoint1.append(next_keypoint_1M)
        all_keypoint2.append(next_keypoint_2M)
        all_identical_points.append(identical_points)
        
        Points_visual(all_points, all_colors, all_point3d_idx, all_keypoint1, "Before_Bundle")

        #redefine
        Second = Third
        initial_point = next_point
        initial_inlinear = next_point3d_idx
        initial_camera_matrix = new_camera_matrix
        initial_matches = next_matches

        print('---------------------#6 Bundle Adjustment---------------------')
        all_points, all_camera_matrix = Bundle(all_points, all_point3d_idx, all_camera_matrix, all_keypoint1, all_keypoint2, all_identical_points, K)
        Save_Camera(all_camera_matrix)
        Points_visual(all_points, all_colors, all_point3d_idx, all_keypoint1, "After_Bundle")
    
    Save_Camera(all_camera_matrix)
    
    Points_visual(all_points, all_colors, all_point3d_idx, all_keypoint1, "multi_view_keypoints")


if __name__ == "__main__":
    main()