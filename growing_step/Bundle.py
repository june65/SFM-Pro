import numpy as np
import cv2
from tqdm import tqdm
import random
import matlab.engine

eng = matlab.engine.start_matlab()
eng.addpath(r'./data/Newfunctions', nargout=0)

def Bundle(all_points, all_point3d_idx, all_camera_matrix, all_keypoint1, all_keypoint2, all_identical_points, K):

    all_xidx = []
    samegroup = []
    number_idx = 0
    for i in range(len(all_keypoint1)):
        x_idx = [-1 for _ in range(len(all_keypoint1[i]))]
        if samegroup != []:
            flat_samegroup = [item for sublist in samegroup for item in sublist]
        else:
            flat_samegroup = []
        for j, index in enumerate(all_point3d_idx[i]):
            if i>0: 
                if index in all_identical_points[i-1][:,1].tolist():
                    point_index = all_identical_points[i-1][:,1].tolist().index(index)
                    matched_point = all_xidx[-1][all_identical_points[i-1][point_index][0]]
                    if matched_point != -1:
                        if not matched_point in flat_samegroup:
                            samegroup.append([matched_point, np.int64(j + number_idx + 1)])
                        else :
                            for row in samegroup:
                                if matched_point in row:
                                    row.append(np.int64(j + number_idx + 1))
            x_idx[index] = j + number_idx + 1
        all_xidx.append(x_idx)
        number_idx += len(all_point3d_idx[i])
        
    if samegroup != []:
        flat_samegroup = [item for sublist in samegroup for item in sublist]
        flat_samegroup = np.array(flat_samegroup)
    
    
    total_3dpoint_num = 0 #전체 3d points 수
    for i in range(len(all_point3d_idx)):
        total_3dpoint_num += len(all_point3d_idx[i])

    change_3d = [np.int64(j+1) for j in range(total_3dpoint_num)] #전체 겹치는 점을 뺀 3dpoint
    remake_3d = [np.int64(j+1) for j in range(total_3dpoint_num)] #겹칠때 교체되는 점의 index를 넣은 3dpoint

    for i in range(total_3dpoint_num):
        flag = np.int64(i+1)
        if flag in flat_samegroup:
            for row in samegroup:
                if flag in row:
                    if min(row) != flag:
                        change_3d.remove(flag)
                        remake_3d[i] = min(row)

    #### 초기 설정
    x = np.array([])
    for i in range(len(all_camera_matrix)):
        R = all_camera_matrix[i][:, :3]
        t = all_camera_matrix[i][:, 3]
        rvec, _ = cv2.Rodrigues(R)
        rvec = rvec.flatten()
        t = t.flatten()
        camera_vec = np.concatenate((rvec, t))
        x = np.concatenate((x, camera_vec))

    number_idx = 0
    for i in range(len(all_points)):
        for j, point in enumerate(all_points[i]):
            flag = np.int64(number_idx + j + 1)
            if flag in flat_samegroup:
                for row in samegroup:
                    if flag in row:
                        if min(row) == flag:
                            point3d_vec = point.flatten()
                            x = np.concatenate((x, point3d_vec))
            else:
                point3d_vec = point.flatten()
                x = np.concatenate((x, point3d_vec))
        number_idx += len(all_points[i])

    keypoint1_vecs = []
    number_idx = 0
    for i in range(len(all_keypoint1)):
        x_idx = [-1 for _ in range(len(all_keypoint1[i]))]
        for j, index in enumerate(all_point3d_idx[i]):
            flag = j + number_idx + 1
            if flag in flat_samegroup and i!=0:
                for row in samegroup:
                    if flag in row:
                        x_idx[index] = int(change_3d.index(min(row))) + 1
            else:
                x_idx[index] = int(change_3d.index(j + number_idx + 1)) + 1
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
            flag = j + number_idx + 1
            if flag in flat_samegroup and i!=0:
                for row in samegroup:
                    if flag in row:
                        x_idx[index] = int(change_3d.index(min(row))) + 1
            else:
                x_idx[index] = int(change_3d.index(j + number_idx + 1)) + 1  #matlab array call +1
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
    param['key1'] = 2 #len(all_camera_matrix)-2 #고정 카메라
    param['key2']  = 0 #사용X
    param['optimization'] = 1
    param['dof_remove'] = 0

    x_BA = eng.LM2_iter_dof(x_matlab, param)

    x_BA = np.array(x_BA[-1])
    x_BApoint = x_BA[6*len(all_camera_matrix):]
    x_BApoint = x_BApoint.reshape(-1,3)

    new_points = []
    number_idx = 0
    for i in range(len(all_points)):
        new_point = np.empty((0, 3))
        for j in range(len(all_points[i])):
            point3d = x_BApoint[change_3d.index(remake_3d[number_idx + j])]
            new_point = np.vstack((new_point, point3d))
        new_points.append(new_point)
        number_idx += len(all_points[i])

    new_camera_matrices = []
    for i in range(len(all_camera_matrix)):
        rvec = x_BA[(6*i):(6*i)+3]
        tvec = x_BA[(6*i)+3:(6*i)+6]
        R, _ = cv2.Rodrigues(rvec)
        new_camera_matrix = np.hstack((R, tvec.reshape(-1, 1)))
        new_camera_matrices.append(new_camera_matrix)

    return new_points, new_camera_matrices
    

def Noise_Bundle(all_points, all_point3d_idx, all_camera_matrix, all_keypoint1, all_keypoint2, all_identical_points, K):

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
    param['key1'] = 1
    param['key2']  = 2
    param['optimization'] = 1
    param['dof_remove'] = 0

    x_BA = eng.LM2_iter_dof(x_matlab, param)

    x_BA = np.array(x_BA[-1])
    x_BApoint = x_BA[6*len(all_camera_matrix):]
    x_BApoint = x_BApoint.reshape(-1,3)

    new_points = []
    start_idx = 0 
    for i in range(len(all_points)):
        length = len(all_points[i])
        new_points.append(x_BApoint[start_idx:start_idx + length])
        start_idx += len(all_points[i])

    new_camera_matrices = []
    for i in range(len(all_camera_matrix)):
        rvec = x_BA[(6*i):(6*i)+3]
        tvec = x_BA[(6*i)+3:(6*i)+6]
        R, _ = cv2.Rodrigues(rvec)
        new_camera_matrix = np.hstack((R, tvec.reshape(-1, 1)))
        new_camera_matrices.append(new_camera_matrix)

    return new_points, new_camera_matrices
   