import numpy as np
import cv2
from tqdm import tqdm
import random
import matlab.engine
import subprocess

eng = matlab.engine.start_matlab()
eng.addpath(r'./data/Givenfunctions', nargout=0)


def ThreePoint(matches, next_matches, inlinear, initial_point, next_camerapoint_2M, threepoint_threshold, threepoint_max_iter):
    query_idx = [match.queryIdx for match in matches] #First
    train_idx = [match.trainIdx for match in matches] #Second
    
    next_query_idx = [match.queryIdx for match in next_matches] #Second
    next_train_idx = [match.trainIdx for match in next_matches] #Third

    double_matched_points = []

    for i in range(len(next_query_idx)):
        if next_query_idx[i] in np.array(train_idx)[inlinear].tolist():
            double_matched_points.append([np.array(train_idx)[inlinear].tolist().index(next_query_idx[i]), train_idx.index(next_query_idx[i]), i]) # idx[3d_point, match1, match2]
            
    double_matched_points = np.array(double_matched_points)
    
    print("Matching points in three image:",{len(double_matched_points)})
    '''
    if len(double_matched_points) <= 100:
        subprocess.run(["c:/venv/SFMpro2/Scripts/python.exe", "c:/D_project/SFM-Pro/main.py"])
    '''
    num_points = len(double_matched_points)
    max_sum = 0
    #max_error_matrix = []
    for iter in tqdm(range(threepoint_max_iter)):
        
        random_indices = random.sample(range(num_points), 3) #Third

        point_3d = initial_point[double_matched_points[random_indices,0]] #3d_point
        rand_P2 = next_camerapoint_2M[:, double_matched_points[random_indices,2]] #match1
        input = np.concatenate((rand_P2.T, point_3d), axis=1)
        input = matlab.double(input)
        output = eng.PerspectiveThreePoint(input)
        output = np.array(output)
        if output.shape != ():
            cnt_p3p = int(len(output) / 4)
            
            for cnt_p in range(cnt_p3p):
                P = np.array(output)[(4*cnt_p):(4*cnt_p)+3, :]
                sum = 0
                #error_matrix = []
                for k in range(num_points):
                    point_4d = np.concatenate((initial_point[double_matched_points[k,0]].T,np.array([1])), axis=0)
                    point = P@point_4d
                    point = point[:2]/point[2]
                    error = np.mean((point.T - next_camerapoint_2M[:2, double_matched_points[k,2]].T)**2)
                    if error < threepoint_threshold:
                        sum += 1
                        #error_matrix.append(error)

                if max_sum < sum:
                    max_sum = sum
                    max_sum_P = P
                    #max_error_matrix = error_matrix
                    
    identical_points = double_matched_points[:,-2:]

    print('Camera Matrix :', max_sum_P)
    print('Inlinear Number :', max_sum)

    return max_sum_P, identical_points