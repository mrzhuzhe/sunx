from scene import Scene
import taichi as ti
from taichi.math import *

scene = Scene(exposure=1)
#scene.set_floor(-0.05, (1.0, 1.0, 1.0))
#scene.set_background_color((1.0, 0, 0))

scene.set_floor(-0.85, (1.0, 1.0, 1.0))
scene.set_background_color((0.5, 0.5, 0.4))
scene.set_directional_light((1, 1, -1), 0.2, (1, 0.8, 0.6))

@ti.kernel
def initialize_voxels():
    # Your code here! :-)
    for i in range(10):
        for j in range(10):
            for k in range(10):
                is_light = int(ti.random() > 0.7)
                if i < 8-k or i > k+1 or j < 8-k or j > k+1:
                    scene.set_voxel(vec3(i, j, k), 1 + is_light , vec3(0.1, 0.9, 0.1))                


initialize_voxels()

scene.finish()
