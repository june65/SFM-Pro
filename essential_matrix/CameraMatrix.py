import numpy as np
import subprocess

def CameraMatrix(fundamental_matrix, camerapoint_1M, camerapoint_2M, inlinear_TF):

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
    
    max_camera = []
    max_camera_num = 0
    for camera in camera_matrix:
        camera_num = 0
        Rt0 = np.hstack((np.eye(3), np.zeros((3, 1))))
        p1t = Rt0[0,:]
        p2t = Rt0[1,:]
        p3t = Rt0[2,:]

        result_camera = camera
        q1t = result_camera[0,:]
        q2t = result_camera[1,:]
        q3t = result_camera[2,:]

        for k in range(len(camerapoint_1M[0,:])):
            if inlinear_TF[k]:
            
                x1 = camerapoint_1M[0,k]
                y1 = camerapoint_1M[1,k]
                x2 = camerapoint_2M[0,k]
                y2 = camerapoint_2M[1,k]

                A = np.array([x1 * p3t - p1t, y1 * p3t - p2t, x2 * q3t - q1t, y2 * q3t - q2t])
                
                _, _, point_vector = np.linalg.svd(A)
                point = point_vector[-1, :] 
                point /= point[-1]

                if point[2]>0 and (result_camera@point.T)[2] > 0:
                    camera_num += 1

        if camera_num > max_camera_num:
            if camera[0,0] > 0:
                max_camera_num = camera_num
                max_camera = camera

    print('Camera Matrix :', max_camera)
    print('Inlinear Number :', max_camera_num)
    if len(max_camera) == 0:
        subprocess.run(["c:/venv/SFMpro2/Scripts/python.exe", "c:/D_project/SFM-Pro/main.py"])
    return max_camera