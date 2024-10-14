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
            for k in range(num_points):
                x1 = camerapoint_1M[:, k]
                x2 = camerapoint_2M[:, k]
                error = np.dot(x2.T, np.dot(flag_E, x1))
                if error < threshold:
                    point += 1

            if maxpoint < point:
                maxpoint = point
                maxpoint_E = flag_E
    print(maxpoint_E)
    print(maxpoint)

    return maxpoint_E
