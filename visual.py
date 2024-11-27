import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load 3D point cloud data from CSV file
point_cloud_csv_path = './result/mydata/After_Bundle.csv'  # Replace with your actual file path
point_cloud_data = pd.read_csv(point_cloud_csv_path)

# Load camera matrix data from CSV file
camera_csv_path = './result/mydata/All_Camera.csv'  # Replace with your actual file path
camera_matrices = pd.read_csv(camera_csv_path)

# Extract camera position by decomposing matrices
camera_positions = []
for _, row in camera_matrices.iterrows():
    # Build the 3x4 matrix
    camera_matrix = np.array([
        [row['Matrix_1_1'], row['Matrix_1_2'], row['Matrix_1_3'], row['Matrix_1_4']],
        [row['Matrix_2_1'], row['Matrix_2_2'], row['Matrix_2_3'], row['Matrix_2_4']],
        [row['Matrix_3_1'], row['Matrix_3_2'], row['Matrix_3_3'], row['Matrix_3_4']]
    ])
    # Split into R (3x3) and t (3x1)
    R = camera_matrix[:, :3]
    t = camera_matrix[:, 3]
    # Compute camera position: C = -R^T * t
    C = -np.linalg.inv(R) @ t
    camera_positions.append(C)

camera_positions = np.array(camera_positions)

# Function to draw camera as a pyramid
def draw_camera(ax, camera_position, R, scale=0.1, height_scale=0.1, color='red'):
    # Camera base (square near plane)
    square_size = scale
    height = height_scale
    base_points = np.array([
        [-square_size, -square_size, height],
        [square_size, -square_size, height],
        [square_size, square_size, height],
        [-square_size, square_size, height],
    ])
    apex = np.array([0, 0, 0])  # Camera optical center
    
    # Transform points to world coordinates
    base_points_world = (R.T @ base_points.T).T + camera_position
    apex_world = camera_position

    # Draw base (square)
    for i in range(4):
        x = [base_points_world[i][0], base_points_world[(i + 1) % 4][0]]
        y = [base_points_world[i][1], base_points_world[(i + 1) % 4][1]]
        z = [base_points_world[i][2], base_points_world[(i + 1) % 4][2]]
        ax.plot(x, y, z, color=color)

    # Draw edges to apex
    for i in range(4):
        x = [base_points_world[i][0], apex_world[0]]
        y = [base_points_world[i][1], apex_world[1]]
        z = [base_points_world[i][2], apex_world[2]]
        ax.plot(x, y, z, color=color)

# Plotting the 3D points and cameras with visualization
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
filtered_point_cloud_data = point_cloud_data[(point_cloud_data['z'] >= 1) & (point_cloud_data['z'] <= 8)]

# Plot 3D points
ax.scatter(filtered_point_cloud_data['x'], filtered_point_cloud_data['y'], filtered_point_cloud_data['z'],
           c=filtered_point_cloud_data[['r', 'g', 'b']] / 255.0, marker='s', s=0.7, label='3D Points')

# Plot cameras as pyramids
for i, camera_position in enumerate(camera_positions):
    # Build R matrix from camera matrices
    camera_matrix = np.array([
        [camera_matrices.loc[i, 'Matrix_1_1'], camera_matrices.loc[i, 'Matrix_1_2'], camera_matrices.loc[i, 'Matrix_1_3']],
        [camera_matrices.loc[i, 'Matrix_2_1'], camera_matrices.loc[i, 'Matrix_2_2'], camera_matrices.loc[i, 'Matrix_2_3']],
        [camera_matrices.loc[i, 'Matrix_3_1'], camera_matrices.loc[i, 'Matrix_3_2'], camera_matrices.loc[i, 'Matrix_3_3']],
    ])
    draw_camera(ax, camera_position, camera_matrix, scale=0.3, height_scale=0.3, color='red')


# Set equal scale for all axes (including camera positions)3
all_x = np.append(filtered_point_cloud_data['x'].values, camera_positions[:, 0])
all_y = np.append(filtered_point_cloud_data['y'].values, camera_positions[:, 1])
all_z = np.append(filtered_point_cloud_data['z'].values, camera_positions[:, 2])

max_range = np.array([
    all_x.max() - all_x.min(),
    all_y.max() - all_y.min(),
    all_z.max() - all_z.min()
]).max() / 2

mid_x = (all_x.max() + all_x.min()) * 0.5
mid_y = (all_y.max() + all_y.min()) * 0.5
mid_z = (all_z.max() + all_z.min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)


fig.patch.set_alpha(0)  # 전체 배경 투명화
ax.set_facecolor((1.0, 1.0, 1.0, 0.0))  # 투명 배경
ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # X축선 제거
ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Y축선 제거
ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Z축선 제거
ax.grid(False)  # 눈금 제거
ax.set_xticks([])  # X축 눈금 제거
ax.set_yticks([])  # Y축 눈금 제거
ax.set_zticks([])  # Z축 눈금 제거
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))  # X축 테두리 제거
ax.yaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))  # Y축 테두리 제거
ax.zaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))  # Z축 테두리 제거

ax.set_xlabel('')
ax.set_ylabel('')
ax.set_zlabel('')
ax.set_title('')
ax.legend()

plt.show()
