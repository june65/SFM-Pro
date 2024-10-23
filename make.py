import numpy as np
import pandas as pd

def load_point_cloud_from_csv(csv_path):
    df = pd.read_csv(csv_path, sep=',', header=None)
    point_cloud_data = np.array(df.values[1:, 1:], dtype=np.float32)
    
    points = point_cloud_data[:, :3]
    colors = point_cloud_data[:, 3:6]
    
    return points, colors

def save_point_cloud_as_ply(points, colors, output_path):
    num_points = points.shape[0]
    
    ply_header = f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    ply_data = np.hstack([points, colors]).astype(np.float32)
    
    with open(output_path, 'w') as ply_file:
        ply_file.write(ply_header)
        for point_color in ply_data:
            ply_file.write(f"{point_color[0]} {point_color[1]} {point_color[2]} {int(point_color[3])} {int(point_color[4])} {int(point_color[5])}\n")

    print(f"Point cloud successfully saved to {output_path}")

csv_path_two_views = './result/two_view_keypoints.csv'
ply_output_path_two_views = './result/two_view_keypoints.ply'

points_two_views, colors_two_views = load_point_cloud_from_csv(csv_path_two_views)
save_point_cloud_as_ply(points_two_views, colors_two_views, ply_output_path_two_views)

csv_path_three_views = './result/three_view_keypoints.csv'
ply_output_path_three_views = './result/three_view_keypoints.ply'

points_three_views, colors_three_views = load_point_cloud_from_csv(csv_path_three_views)
save_point_cloud_as_ply(points_three_views, colors_three_views, ply_output_path_three_views)

csv_path_Before_Bundle = './result/Before_Bundle.csv'
ply_output_path_Before_Bundle = './result/Before_Bundle.ply'

points_Before_Bundle, colors_Before_Bundle = load_point_cloud_from_csv(csv_path_Before_Bundle)
save_point_cloud_as_ply(points_Before_Bundle, colors_Before_Bundle, ply_output_path_Before_Bundle)

csv_path_After_Bundle = './result/After_Bundle.csv'
ply_output_path_After_Bundle = './result/After_Bundle.ply'

points_After_Bundle, colors_After_Bundle = load_point_cloud_from_csv(csv_path_After_Bundle)
save_point_cloud_as_ply(points_After_Bundle, colors_After_Bundle, ply_output_path_After_Bundle)
