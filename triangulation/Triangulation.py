import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
import subprocess

def Triangulation(image_1, camerapoint_1M, camerapoint_2M, keypoint_1M, keypoint_2M, camera_matrix, inlier_TF, inlinear):

    main_camera = np.eye(4)
    Rt0 = main_camera[:3]
    result_camera = camera_matrix
    
    p1t = Rt0[0,:]
    p2t = Rt0[1,:]
    p3t = Rt0[2,:]
    
    q1t = result_camera[0,:]
    q2t = result_camera[1,:]
    q3t = result_camera[2,:]
    colors = []
    points = []
    new_inlinear = []
    initial_point3d_idx = []
    for k in range(len(camerapoint_1M[0,:])):

        if inlier_TF[k]:
            x1 = camerapoint_1M[0,k]
            y1 = camerapoint_1M[1,k]
            x2 = camerapoint_2M[0,k]
            y2 = camerapoint_2M[1,k]

            A = np.array([x1 * p3t - p1t, y1 * p3t - p2t, x2 * q3t - q1t, y2 * q3t - q2t])
            
            _, _, point_vector = np.linalg.svd(A)
            point = point_vector[-1, :] 
            point /= point[-1]

            if point[2]>5 and point[2]<10 and (result_camera@point.T)[2]>1 and (result_camera@point.T)[2]<10:

                points.append(point[:3])
                
                color = image_1[int(keypoint_1M[k][1]), int(keypoint_1M[k][0])]
                colors.append(color)
                new_inlinear.append(inlinear[k])
                initial_point3d_idx.append(k)

    print('3D Inlinear Number :',len(points))
    '''
    #Noise_Bundle random noise 
    sigma = 0.1
    points = [point + np.random.normal(0, sigma, size=3) for point in points]
    '''
    points = np.array(points)
    colors = np.array(colors)
    new_inlinear = np.array(new_inlinear)
    initial_point3d_idx = np.array(initial_point3d_idx)

    point_cloud = np.concatenate((points, colors), axis=1)
    point_cloud = np.concatenate((point_cloud, initial_point3d_idx.reshape(-1, 1)), axis=1)
    df = pd.DataFrame(point_cloud, columns=['x','y','z','r', 'g', 'b','inlinear_idx'])
    df.to_csv('./result/two_view_keypoints.csv', mode='w')

    '''
    colors = colors / 255.0
    X = points[:,0]
    Y = points[:,1]
    Z = points[:,2]
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.scatter3D(X, Y, Z, c=colors, marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("3D Reconstructed Points")
    plt.show()
    '''
    
    return points, colors, new_inlinear, initial_point3d_idx

def compute_essential_matrix(Rt0, result_camera):
    R0, t0 = Rt0[:, :3], Rt0[:, 3]
    R1, t1 = result_camera[:, :3], result_camera[:, 3]
    R = R1 @ R0.T
    t = t1 - R @ t0
    t_cross = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])
    E = t_cross @ R
    return E

def Triangulation_G(image_1, camerapoint_1M, camerapoint_2M, keypoint_1M, keypoint_2M, camera_matrix, last_camera_matrix, threshold_1, threshold_2):

    Rt0 = last_camera_matrix
    result_camera = camera_matrix
    
    flag_E = compute_essential_matrix(last_camera_matrix,camera_matrix)
    
    #inlinear_TF for new points
    inlinear_TF = []
    for k in range(len(camerapoint_1M[0,:])):
        x1 = camerapoint_1M[:, k]
        x2 = camerapoint_2M[:, k]
        error = np.dot(x2.T, np.dot(flag_E, x1))
        if error < threshold_1 and error > 0:
            inlinear_TF.append(True)
        else:
            inlinear_TF.append(False)
    
    p1t = Rt0[0,:]
    p2t = Rt0[1,:]
    p3t = Rt0[2,:]
    
    q1t = result_camera[0,:]
    q2t = result_camera[1,:]
    q3t = result_camera[2,:]
    colors = []
    points = []
    initial_point3d_idx = []
    for k in range(len(camerapoint_1M[0,:])):

        #if inlinear_TF[k]:
            x1 = camerapoint_1M[0,k]
            y1 = camerapoint_1M[1,k]
            x2 = camerapoint_2M[0,k]
            y2 = camerapoint_2M[1,k]

            A = np.array([x1 * p3t - p1t, y1 * p3t - p2t, x2 * q3t - q1t, y2 * q3t - q2t])
            
            _, _, point_vector = np.linalg.svd(A)
            point = point_vector[-1, :] 
            point /= point[-1]

            if point[2]>5 and point[2]<10 and (result_camera@point.T)[2]>1 and (result_camera@point.T)[2]<10:
                error = np.sum((camerapoint_2M[:2, k] - point[:2])**2)
                if error < threshold_2:
                    points.append(point[:3])
                    color = image_1[int(keypoint_1M[k,1]), int(keypoint_1M[k,0])]
                    colors.append(color)
                    initial_point3d_idx.append(k)
                
        
    print('3D Inlinear Number :',len(points))
    '''
    if len(points) < 200:
        subprocess.run(["c:/venv/SFMpro2/Scripts/python.exe", "c:/D_project/SFM-Pro/main.py"])
    '''
    '''
    #Noise_Bundle random noise 
    sigma = 0.1
    points = [point + np.random.normal(0, sigma, size=3) for point in points]
    '''
    points = np.array(points)
    colors = np.array(colors)
    initial_point3d_idx = np.array(initial_point3d_idx)

    point_cloud = np.concatenate((points, colors), axis=1)
    point_cloud = np.concatenate((point_cloud, initial_point3d_idx.reshape(-1, 1)), axis=1)
    df = pd.DataFrame(point_cloud, columns=['x','y','z','r', 'g', 'b','inlinear_idx'])
    df.to_csv('./result/three_view_keypoints.csv', mode='w')

    '''
    colors = colors / 255.0
    X = points[:,0]
    Y = points[:,1]
    Z = points[:,2]
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.scatter3D(X, Y, Z, c=colors, marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("3D Reconstructed Points")
    plt.show()
    '''
    return points, colors, initial_point3d_idx