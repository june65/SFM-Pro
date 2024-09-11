import numpy as np
import cv2
from tqdm import tqdm
import random
import matlab.engine

eng = matlab.engine.start_matlab()
eng.addpath(r'./data/Givenfunctions', nargout=0)

def FivePoint(matches, keypoint_1M, keypoint_2M, threshold, max_iter):
    num_points = len(matches)
    maxpoint_E = []
    maxpoint = 0
    for t in tqdm(range(max_iter)):
        random_indices = random.sample(range(num_points), 5)
        rand_matches = matches[random_indices]
        rand_P1 = keypoint_1M[random_indices]
        rand_P2 = keypoint_2M[random_indices]
        rand_P1 = np.vstack((rand_P1.T, np.ones((1, 5))))
        rand_P2 = np.vstack((rand_P2.T, np.ones((1, 5))))
        P1_matlab = matlab.double(rand_P1.tolist())
        P2_matlab = matlab.double(rand_P2.tolist())

        E = eng.calibrated_fivepoint(P1_matlab, P2_matlab)
        E = np.array(E)
        for j in range(len(E[0])):
            flag_E = E[:, j].reshape(3, 3)
            point = 0
            for k in range(num_points):
                x1 = np.array([*keypoint_1M[k], 1]).reshape(3, 1)
                x2 = np.array([*keypoint_2M[k], 1]).reshape(3, 1)
                error = abs(np.dot(x2.T, np.dot(flag_E, x1)))[0][0]
                if error < threshold:
                    point += 1

            if maxpoint < point:
                maxpoint = point
                maxpoint_E = flag_E
    print(maxpoint_E)
    print(maxpoint)

    return maxpoint_E
