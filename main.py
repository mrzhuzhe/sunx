from scene import Scene
import taichi as ti
from taichi.math import *

scene = Scene(exposure=1)
scene.set_floor(-0.85, (1.0, 1.0, 1.0))
scene.set_background_color((0.5, 0.5, 0.4))
scene.set_directional_light((1, 1, -1), 0.2, (1, 0.8, 0.6))

ti.func
def rotate2d(x, y, r):
    return x*ti.cos(r) - y*ti.sin(r) , y*ti.cos(r) + x*ti.sin(r)

@ti.kernel
def initialize_voxels():
    n = 60
    for i, j, k in ti.ndrange((-n, n), (-n, 0), (-n, n)):        
        x = ivec3(i, j, k)
        if  x.dot(x) < n * n * 0.5:
            if n * n * 0.4 < x.dot(x):
                _j, _k = rotate2d(j, k, -45)
                scene.set_voxel(vec3(i, _j, _k), 1, vec3(0.9, 0.3, 0.3))        
        if ti.pow(i*0.05, 2) + ti.pow(k*0.05 - ti.pow((i*0.05)**2, 0.333), 2) <= 1:
            _j, _k = rotate2d(j/10, k-10, -45)
            scene.set_voxel(vec3(i, _j, _k), 1 , vec3(0.9, 0.9, 0.1)) 
initialize_voxels()
scene.finish()