import numpy as np
import cv2
from tqdm import tqdm
import random
import matlab.engine

eng = matlab.engine.start_matlab()
eng.addpath(r'./data/Newfunctions', nargout=0)

def Bundle(all_points, all_point3d_idx, all_camera_matrix, all_keypoint1, all_keypoint2, K):

    x = np.array([])
    for i in range(len(all_camera_matrix)):
        R = all_camera_matrix[i][:, :3]
        t = all_camera_matrix[i][:, 3]
        rvec, _ = cv2.Rodrigues(R)
        rvec = rvec.flatten()
        t = t.flatten()
        camera_vec = np.concatenate((rvec, t))
        x = np.concatenate((x, camera_vec))

    for i in range(len(all_points)):
        point3d_vec = all_points[i].flatten()
        x = np.concatenate((x, point3d_vec))

    keypoint1_vecs = []
    number_idx = 0
    for i in range(len(all_keypoint1)):
        x_idx = [-1 for _ in range(len(all_keypoint1[i]))]
        for j, index in enumerate(all_point3d_idx[i]):
            x_idx[index] = j + number_idx + 1 #matlab array call +1
        all_keypoint1_T = all_keypoint1[i].T
        x_idx = np.array(x_idx).reshape(1, -1)
        point2d_vec = np.vstack((all_keypoint1_T, x_idx))
        point2d_vec_filtered = point2d_vec[:, point2d_vec[2, :] != -1]
        keypoint1_vecs.append(point2d_vec_filtered)
        number_idx += len(all_point3d_idx[i])
    
    keypoint2_vecs = []
    number_idx = 0
    for i in range(len(all_keypoint2)):
        x_idx = [-1 for _ in range(len(all_keypoint2[i]))]
        for j, index in enumerate(all_point3d_idx[i]):
            x_idx[index] = j + number_idx + 1 #matlab array call +1
        all_keypoint2_T = all_keypoint2[i].T
        x_idx = np.array(x_idx).reshape(1, -1)
        point2d_vec = np.vstack((all_keypoint2_T, x_idx))
        point2d_vec_filtered = point2d_vec[:, point2d_vec[2, :] != -1]
        keypoint2_vecs.append(point2d_vec_filtered)
        number_idx += len(all_point3d_idx[i])

    uv = []
    for i in range(len(all_camera_matrix)):
        if i==0:
            uv.append(keypoint1_vecs[i])
        elif i==len(all_camera_matrix)-1:
            uv.append(keypoint2_vecs[i-1])
        else:
            uv.append(np.concatenate((keypoint2_vecs[i-1], keypoint1_vecs[i]), axis=1))
            
    x_matlab = matlab.double(x)
    uv_matlab = uv
    #[matlab.double(uv[i, :, :]) for i in range(uv.shape[0])]
    K_matlab = matlab.double(K)
    param = eng.struct({'uv': uv_matlab, 'K': K_matlab})
    param['nX'] = len(x) - 6 * len(all_camera_matrix)
    param['key1'] = 4
    param['key2']  = 5
    param['optimization'] = 1
    param['dof_remove'] = 0

    x_BA = eng.LM2_iter_dof(x_matlab, param)
    eng.quit()

    x_BA = np.array(x_BA[-1])
    x_BA = x_BA[6*len(all_camera_matrix):]
    x_BA = x_BA.reshape(-1,3)

    new_points = []
    start_idx = 0 
    for i in range(len(all_points)):
        length = len(all_points[i])
        new_points.append(x_BA[start_idx:start_idx + length])

    return new_points
    