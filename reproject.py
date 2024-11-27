import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

camera_matrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0]
])
data = pd.read_csv('./result/threepoint/multi_view_keypoints.csv') 

points_3d = data[['x', 'y', 'z']].values.T

points_3d_homogeneous = np.vstack((points_3d, np.ones((1, points_3d.shape[1]))))

points_2d_homogeneous = camera_matrix @ points_3d_homogeneous

points_2d = points_2d_homogeneous[:2] / points_2d_homogeneous[2]

plt.figure(figsize=(10, 10))
plt.scatter(points_2d[0], points_2d[1], c=data[['r', 'g', 'b']].values / 255, s=5)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Reprojected Points")
plt.axis('equal') 
plt.gca().invert_yaxis() 
plt.show()
