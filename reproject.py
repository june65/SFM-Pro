import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load camera matrix
camera_matrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0]
])

# Load the point cloud data from CSV
data = pd.read_csv('./result/threepoint/multi_view_keypoints.csv')  # Replace with your CSV file name

# Extract the x, y, z coordinates
points_3d = data[['x', 'y', 'z']].values.T

# Add a row of ones for homogeneous coordinates
points_3d_homogeneous = np.vstack((points_3d, np.ones((1, points_3d.shape[1]))))

# Reproject the 3D points into 2D
points_2d_homogeneous = camera_matrix @ points_3d_homogeneous

# Convert from homogeneous to 2D coordinates
points_2d = points_2d_homogeneous[:2] / points_2d_homogeneous[2]

# Plot the points on an image with equal scaling for x and y axes
plt.figure(figsize=(10, 10))
plt.scatter(points_2d[0], points_2d[1], c=data[['r', 'g', 'b']].values / 255, s=5)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Reprojected Points")
plt.axis('equal')  # Set equal scaling for both axes
plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
plt.show()
