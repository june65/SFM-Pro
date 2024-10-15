import numpy as np
import cv2
import matplotlib.pyplot as plt

def Triangulation(image_1, camerapoint_1M, camerapoint_2M, keypoint_1M, keypoint_2M, camera_matrix, inlier_TF):

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

            if (result_camera@point.T)[2] > 0:

                points.append(point)
                color = image_1[int(keypoint_1M[k][1]), int(keypoint_1M[k][0])]
                colors.append(color)

    points = np.array(points)
    colors = np.array(colors)
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
    
    return points, colors