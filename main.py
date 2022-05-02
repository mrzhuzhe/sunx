from turtle import color
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
    n = 60
    for i, j, k in ti.ndrange((-n, n), (-n, 0), (-n, n)):
        x = ivec3(i, j, k)
        if  n * n * 0.4 < x.dot(x) < n * n * 0.5:
            scene.set_voxel(vec3(i, j, k), 1, vec3(0.9, 0.3, 0.3))

    n = 3
    for i, j, k in ti.ndrange((-10*n, 10*n), (-4*n, 4*n), (-4*n, 4*n)):
        is_light = int(ti.random() > 0.7)
        if k % 2 == 0:
            scene.set_voxel(vec3(i, j, k*ti.sin(i* 0.2)), 1 + is_light , vec3(0.9, 0.9, 0.1)) 
        else:
            scene.set_voxel(vec3(i, j, k*ti.sin(i* 0.2)), 1 + is_light , vec3(0.1, 0.1, 0.1))  

initialize_voxels()

scene.finish()
