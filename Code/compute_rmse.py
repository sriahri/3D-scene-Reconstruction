import numpy as np

import cv2
import matplotlib.image as mimg

def compute_rmse(depth_map1, depth_map2):
    # Check if both depth maps have the same shape
    if depth_map1.shape != depth_map2.shape:
        print(depth_map1.shape, depth_map2.shape)
        raise ValueError("Depth maps must have the same shape.")

    # Flatten the depth maps to compute the RMSE
    depth_map1_flat = depth_map1.flatten()
    depth_map2_flat = depth_map2.flatten()

    # Compute the squared differences
    squared_diff = (depth_map1_flat - depth_map2_flat) ** 2

    # Compute the mean squared error (MSE)
    mse = np.mean(squared_diff)

    # Compute the root mean squared error (RMSE)
    rmse = np.sqrt(mse)

    return rmse

# Assuming depth_map1 and depth_map2 are the two depth maps (numpy arrays) you want to compare
# Make sure both depth maps have the same dimensions

# Call the compute_rmse function
depth_map_h = "depth_image_saved.jpg"
depth_map_o = "0000000000_depth.jpg"
ground_depth = "/home/labmember/Desktop/project/hamlyn_data/hamlyn_data/rectified01/depth01/0000000000.png"
depth_map_h = mimg.imread(depth_map_h)
depth_map_o = mimg.imread(depth_map_o)
ground_depth = mimg.imread(ground_depth)
ground_depth = ground_depth.astype(np.float32)/np.iinfo(np.uint16).max
rmse_value = compute_rmse(depth_map_h, depth_map_o)

print(f"RMSE between the two depth maps: {rmse_value}")


from skimage.metrics import mean_squared_error

mse = mean_squared_error(depth_map_h, depth_map_o)
rmse = np.sqrt(mse)
print(rmse)