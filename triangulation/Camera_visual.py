import numpy as np
import csv

def Save_Camera(camera_matrices, filename="./result/All_Camera.csv"):
    flattened_matrices = [matrix.flatten() for matrix in camera_matrices]

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Matrix_{}_{}'.format(i+1, j+1) for i in range(3) for j in range(4)])
        writer.writerows(flattened_matrices)

def Load_Camera(filename="./result/All_Camera.csv"):
    camera_matrices = []
    with open(filename, mode='r', newline='') as file:
        reader = csv.reader(file)
        next(reader) 
        for row in reader:
            flat_matrix = np.array(row, dtype=float)
            matrix = flat_matrix.reshape(3, 4)
            camera_matrices.append(matrix)
    return camera_matrices

# 카메라 메트릭스 리스트 불러오기
# loaded_matrices = Load_Camera()