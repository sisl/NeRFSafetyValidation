import bmesh
import bpy
import numpy as np


D = bpy.data
C = bpy.context

COLLISION_MESHES = [
    D.objects["Mesh_0"],
    D.objects["Mesh_1"],
    D.objects["Mesh_2"],
    D.objects["Mesh_3"],
    D.objects["Mesh_4"],
    D.objects["Mesh_5"],
    D.objects["Mesh_6"],
    D.objects["Mesh_7"],
]

GRANULARITY = 40 # how many points to sample per world meter. 
# 1 meter in the grid world is approximately 20 meters in the real world
# 1 grid cell at granularity 40 approximates to 1/2 a meter in the real world
# this means our drone is approximately 50 cm wide and 50 cm long (and 50 cm tall)

# define the range of the collision map in world coordinates. We got these from looking at the scene in Blender
START_X = -1.4
END_x = 1
START_Y = -1.3
END_Y = 1
START_Z = -0.1
END_Z = 0.5

def worldToIndex(world, start, granularity):
    return int(np.floor((world - start) * granularity))

def indexToWorld(index, start, granularity):
    return index / granularity + start

X_RANGE = worldToIndex(END_x, START_X, GRANULARITY)
Y_RANGE = worldToIndex(END_Y, START_Y, GRANULARITY)
Z_RANGE = worldToIndex(END_Z, START_Z, GRANULARITY)

print(X_RANGE, Y_RANGE, Z_RANGE)

# create a numpy 3D array of False values
collision_map = np.zeros((X_RANGE, Y_RANGE, Z_RANGE), dtype=bool)
print(collision_map.shape)

# iterate over every vertex in every mesh in world coordinates
# find the corresponding index in the collision_map and set it to True
for mesh in COLLISION_MESHES:
    bm = bmesh.new()
    bm.from_mesh(mesh.data)
    bm.transform(mesh.matrix_world)
    for v in bm.verts:
        x = worldToIndex(v.co.x, START_X, GRANULARITY)
        y = worldToIndex(v.co.y, START_Y, GRANULARITY)
        z = worldToIndex(v.co.z, START_Z, GRANULARITY)
        if 0 <= x < X_RANGE and 0 <= y < Y_RANGE and 0 <= z < Z_RANGE:
            collision_map[x, y, z] = True
        else:
            print("out of range")
    bm.free()

# print the number of occupied cells
print(np.sum(collision_map))
# print the average index of occupied cells transformed back to world coordinates. This is the center of mass of the occupied cells
print(np.mean(np.argwhere(collision_map), axis=0) / GRANULARITY + [START_X, START_Y, START_Z])

print(collision_map)

# save the collision map to a file
np.save("collision_map.npy", collision_map)