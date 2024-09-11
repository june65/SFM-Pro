import numpy as np

def compute_depth(X, P):
    m3 = P[2, 2]
    T = P[2, 3] 
    sign_det_M = np.sign(np.linalg.det(P[:, :3])) 
    w = (P @ X)[2] 
    depth_value = (sign_det_M * w) / (T * m3)
    return depth_value

def CameraMatrix(fundamental_matrix, keypoint_1M, datapath):

    K = np.loadtxt(datapath+'K.txt')
    W = np.array([[0.0, -1.0, 0.0],
     [1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0]])
    
    U, S, Vt = np.linalg.svd(fundamental_matrix)

    camera_matrix = np.array([
        np.column_stack((U @ W @ Vt, U[:, 2])), 
        np.column_stack((U @ W @ Vt, -U[:, 2])), 
        np.column_stack((U @ W.T @ Vt, U[:, 2])), 
        np.column_stack((U @ W.T @ Vt, -U[:, 2])) 
    ])
    
    result_camera = []

    for camera in camera_matrix:
        is_valid = True
        for k in range(len(keypoint_1M)):
            x1 = np.array([*keypoint_1M[k], 1])
            x1 = K @ x1 
            X = np.array([*x1, 1]) 
            depth = compute_depth(X, camera)
            if depth < 0:
                is_valid = False

        if is_valid:
            print("Optimal camera pose found:")
            print(camera)
            result_camera = camera

    return camera_matrix