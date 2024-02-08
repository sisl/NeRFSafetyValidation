# create a signed distance field from the collision map

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

GRANULARITY = 40 # how many points to sample per world meter

# import the collision map from collision_map.npy
collision_map = np.load("collision_map.npy")

# invert the collision map
collision_map = ~collision_map

# create a signed distance field from the collision map
sdf = scipy.ndimage.distance_transform_edt(collision_map)

# plot the signed distance field in 3D where the distance is color coded
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.voxels(sdf < 0.1, edgecolor='k')
# ax.set_aspect('equal')

# plt.show()

# multiply the signed distance field by the resolution to get the distance in meters
sdf /= GRANULARITY # 40 is the resolution of the collision map. This is the same as the GRANULARITY in createCollisionMap.py

print(sdf)

# save the signed distance field to a file
np.save("sdf.npy", sdf)