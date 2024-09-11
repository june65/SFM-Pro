import numpy as np
import cv2
from tqdm import tqdm
import random
import matlab.engine

eng = matlab.engine.start_matlab()
eng.addpath(r'./data/Givenfunctions', nargout=0)

def FivePoint(matches, keypoint_1M, keypoint_2M, threshold, max_iter):
    num_points = len(matches)
    for t in tqdm(range(max_iter)):
        random_indices = random.sample(range(num_points), 5)
        rand_matches = matches[random_indices]
        rand_Q1 = keypoint_1M[random_indices]
        rand_Q2 = keypoint_2M[random_indices]
        rand_Q1 = np.vstack((rand_Q1.T, np.ones((1, 5))))
        rand_Q2 = np.vstack((rand_Q2.T, np.ones((1, 5))))
        Q1_matlab = matlab.double(rand_Q1.tolist())
        Q2_matlab = matlab.double(rand_Q2.tolist())

        E = eng.calibrated_fivepoint(Q1_matlab, Q2_matlab)
        E = np.array(E)

    return matches
