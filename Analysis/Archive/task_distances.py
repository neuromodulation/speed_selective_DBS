# Calculate length of target trajectories

import numpy as np

# Screen properties
pixel_w = 1920
pixel_h = 1080
cm_w = 33.5
cm_h = 19.4
cm_pixel = 33.5/1920

# Target positions
target_pos_x = [390, 465, 615, 690]
target_dis_y = 1832.5 - 87.5
target_dis_x = [75, 150, 225, 300]
dis = np.zeros(5)
dis[0] = target_dis_y * cm_pixel
for i, x_dis in enumerate(target_dis_x):
    dis[i+1] = np.sqrt(x_dis**2 + target_dis_y**2) * cm_pixel

print("test")
