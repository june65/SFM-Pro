import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

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


def Triangulation_G(image_1, camerapoint_1M, camerapoint_2M, keypoint_1M, keypoint_2M, camera_matrix, last_camera_matrix):

    #main_camera = np.eye(4)
    #Rt0 = main_camera[:3]
    Rt0 = last_camera_matrix
    result_camera = camera_matrix
    #last_camera_matrix = np.vstack((last_camera_matrix, [0, 0, 0, 1]))
    #camera_matrix_inv = np.linalg.inv(last_camera_matrix)
    #result_camera = (camera_matrix_inv @ np.vstack((camera_matrix, [0, 0, 0, 1])))[:3]

    
    print('result_camera Matrix :', result_camera)


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
            color = image_1[int(keypoint_1M[k,1]), int(keypoint_1M[k,0])]
            colors.append(color)
            initial_point3d_idx.append(k)
        
    print('3D Inlinear Number :',len(points))
    
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