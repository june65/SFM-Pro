import cv2
import numpy as np
import os

class ImageLoader():

    def __init__(self, datapath="./data/data_30/"):
        self.images = []
        for filename in sorted(os.listdir(datapath)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg','.ppm')):
                img_path = os.path.join(datapath, filename)
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                self.images.append(img)
                
                
    def __getitem__(self, idx=0):
        return self.images[idx]
