import argparse
import numpy as np
import csv
from tqdm import tqdm
from utils import ImageLoader
from feature_extraction import SIFT, ORB
from feature_matching import BF, FLANN
from essential_matrix import FivePoint, CameraMatrix
from triangulation import Triangulation, Triangulation_G, Points_visual, Save_Camera
from growing_step import ThreePoint, Bundle, Noise_Bundle
parser = argparse.ArgumentParser()

#Data parameters
parser.add_argument('--dataset', default='data_30_copy', help='data name')

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
    Extraction_method = "SIFT" #SIFT, ORB
    keypoints = []
    descriptors = []

    #Feature Matching
    Matching_method = "KNN" #NORM, KNN, FLANN
    threshold_knn = 0.55
    kdtree_flann = 1
    
    #Essential Matrix
    fivepoint_threshold = 2.0e-4
    fivepoint_max_iter = 2000

    #Growing Step
    threepoint_threshold = 2.0e-8
    threepoint_max_iter = 2000
    triangulation_threshold_E = 2.0e-4
    triangulation_threshold = 1#2.0e-2
    
    print('---------------------#0 Feature Extraction---------------------')
    for i in tqdm(range(32)):
        images.append(imageset[i])
        keypoint, descriptor =SIFT(images[i])
        keypoints.append(keypoint)
        descriptors.append(descriptor)

    match_results = []

    with open('matches_list.csv', mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            match_results.append((int(row[0]), int(row[1]), int(row[2])))

    all_points = []
    all_colors = []
    all_point3d_idx = []
    all_camera_matrix = [np.eye(4)[:3]]
    all_keypoint1 = []
    all_keypoint2 = []
    all_identical_points = []

    orders = [12, 11]
    
    First = orders[0]
    Second = orders[1]
    #First = 15
    #Second = 14
    #First = 3
    #Second = 4
    '''
    flag = Second
    while len(orders) < 30:
        for index1, index2, match_count in match_results:
            if index1 == flag and index2 not in orders:
                orders.append(index2)
                flag = index2
                break
    orders = orders[2:]
    print(orders)
    '''

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
    
    orders = [29,28,27,26,25,24,23,22,20,19,18,17,16,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    #orders = [5,7,9,11,13,15,14,12,10,8,6,2,0,16,17,18,19,20,22,23,24,25,26,27,28,29,30,31]
    orders = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
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