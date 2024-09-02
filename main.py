import argparse
from utils import ImageLoader
from feature_extraction import SIFT
parser = argparse.ArgumentParser()

#Data parameters
parser.add_argument('--dataset', default='data_2_sfm', help='data name')

args = parser.parse_args()

datapath = "./data/" + args.dataset+ "/"

def main():
    #Data loading
    imageset = ImageLoader(datapath)
    image_1 = imageset[0]
    image_2 = imageset[1]
    
    keypoint = SIFT(image_1, image_2)

if __name__ == "__main__":
    main()