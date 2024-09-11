import numpy as np
import cv2
import matplotlib.pyplot as plt

def Triangulation(image_1, keypoint_1M, keypoint_2M, camera_matrix, datapath):

    K = np.loadtxt(datapath+'K.txt')

    Rt0 = np.hstack((np.eye(3), np.zeros((3, 1))))
    p1t = Rt0[0,:]
    p2t = Rt0[1,:]
    p3t = Rt0[2,:]

    result_camera = K @ camera_matrix
    q1t = result_camera[0,:]
    q2t = result_camera[1,:]
    q3t = result_camera[2,:]
    colors = []
    points = []
    for k in range(len(keypoint_1M)):
        
        x1 = keypoint_1M[k][0]
        y1 = keypoint_1M[k][1]
        x2 = keypoint_2M[k][0]
        y2 = keypoint_2M[k][1]

        A = np.array([x1 * p3t - p1t, y1 * p3t - p2t, x2 * q3t - q1t, y2 * q3t - q2t])
        
        _, _, point_vector = np.linalg.svd(A)
        point = point_vector[-1, :] 
        point /= point[-1]
        points.append(point)
        color = image_1[int(keypoint_1M[k][1]), int(keypoint_2M[k][0])]
        colors.append(color)
    points = np.array(points)
    X = points[:,0]
    Y = points[:,1]
    Z = points[:,2]
    colors = np.array(colors)
    colors = colors / 255.0
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.scatter3D(X, Y, Z, c=colors, marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.title("3D Reconstructed Points")
    plt.show()