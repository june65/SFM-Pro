import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def Points_visual(all_points, all_colors, all_point3d_idx, all_keypoint1, filename):
    
    zero_point = np.empty((0, 3))
    zero_color = np.empty((0, 3)) 
    zero_point3d_idx = np.array([])
    start_index = 0
    for i in range(len(all_points)):
        zero_point = np.vstack((zero_point, all_points[i]))
        zero_color = np.vstack((zero_color, all_colors[i]))
        new_point3d_idx = all_point3d_idx[i] + start_index
        zero_point3d_idx = np.concatenate((zero_point3d_idx, new_point3d_idx))
        start_index += len(all_keypoint1[i])
        
    zero_point = np.array(zero_point)
    zero_color = np.array(zero_color)

    point_cloud = np.concatenate((zero_point, zero_color), axis=1)
    point_cloud = np.concatenate((point_cloud, zero_point3d_idx.reshape(-1, 1)), axis=1)
    df = pd.DataFrame(point_cloud, columns=['x','y','z','r', 'g', 'b','inlinear_idx'])
    df.to_csv('./result/'+filename+'.csv', mode='w')

    zero_color = zero_color / 255.0
    X = zero_point[:,0]
    Y = zero_point[:,1]
    Z = zero_point[:,2]
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.scatter3D(X, Y, Z, c=zero_color, marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.title("3D Reconstructed Points")
    plt.show()