import argparse
from utils import ImageLoader
from feature_extraction import SIFT
from feature_matching import BF
from essential_matrix import FivePoint
parser = argparse.ArgumentParser()

#Data parameters
parser.add_argument('--dataset', default='data_2_sfm', help='data name')

args = parser.parse_args()

datapath = "./data/" + args.dataset+ "/"

def main():
    #Data loading
    imageset = ImageLoader(datapath)
    images = []
    keypoints = []
    descriptors = []
    threshold = 2.0e-4
    max_iter = 5000

    #Feature extraction
    for i in range(2):
        images.append(imageset[i])
        keypoint, descriptor =SIFT(images[i])
        keypoints.append(keypoint)
        descriptors.append(descriptor)

    for i in range(1):
        for j in range(i+1,2):
            #Feature matching
            matches, keypoint_1M, keypoint_2M = BF(keypoints[i], keypoints[j], descriptors[i], descriptors[j], images[i], images[j])
            
            #Essential matrix
            F_matrix = FivePoint(matches, keypoint_1M, keypoint_2M, threshold, max_iter)
            
            

if __name__ == "__main__":
    main()