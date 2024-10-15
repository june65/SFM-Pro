import numpy as np
import cv2
from tqdm import tqdm
import random
import matlab.engine

eng = matlab.engine.start_matlab()
eng.addpath(r'./data/Givenfunctions', nargout=0)

def FivePoint(matches, camerapoint_1M, camerapoint_2M, threshold, max_iter):
    num_points = len(matches)
    maxpoint_E = []
    maxpoint = 0
    maxinlinear_TF = []
    maxinlinear = []
    for t in tqdm(range(max_iter)):
        random_indices = random.sample(range(num_points), 5)
        rand_P1 = camerapoint_1M[:, random_indices]
        rand_P2 = camerapoint_2M[:, random_indices]
        P1_matlab = matlab.double(rand_P1.tolist())
        P2_matlab = matlab.double(rand_P2.tolist())

        E = eng.calibrated_fivepoint(P1_matlab, P2_matlab)
        E = np.array(E)
        for j in range(len(E[0])):
            flag_E = E[:, j].reshape(3, 3)
            point = 0
            inlinear_TF = []
            inlinear = []
            for k in range(num_points):
                x1 = camerapoint_1M[:, k]
                x2 = camerapoint_2M[:, k]
                error = np.dot(x2.T, np.dot(flag_E, x1))
                if error < threshold and error > 0:
                    point += 1
                    inlinear_TF.append(True)
                    inlinear.append(k)
                else:
                    inlinear_TF.append(False)

            if maxpoint < point:
                maxpoint = point
                maxpoint_E = flag_E
                maxinlinear_TF = inlinear_TF
                maxinlinear = inlinear
    print('Essential Matrix :', maxpoint_E)
    print('Inlinear Number :', maxpoint)

    return maxpoint_E, maxinlinear_TF, inlinear
