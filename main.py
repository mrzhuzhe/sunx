from scene import Scene
import taichi as ti
from taichi.math import *

scene = Scene(exposure=1)
scene.set_floor(-0.85, (1.0, 1.0, 1.0))
scene.set_background_color((0.5, 0.5, 0.4))
scene.set_directional_light((1, 1, -1), 0.2, (1, 0.8, 0.6))

@ti.kernel
def initialize_voxels():
    n = 60
    for i, j, k in ti.ndrange((-n, n), (-n, 0), (-n, n)):
        
        x = ivec3(i, j, k)
        if  x.dot(x) < n * n * 0.5:
            if n * n * 0.4 < x.dot(x):
                scene.set_voxel(vec3(i, j*ti.cos(-45) - k*ti.sin(-45) , k*ti.cos(-45) + j*ti.sin(-45)), 1, vec3(0.9, 0.3, 0.3))
        
        if ti.pow(i*0.05, 2) + ti.pow(k*0.05 - ti.pow((i*0.05)**2, 0.333), 2) <= 1:
            # heart
            #scene.set_voxel(vec3(i,  j/10,  -k+10), 1 , vec3(0.9, 0.9, 0.1)) 
            
            # rotate k   x cos - y sin  y cos + xq sin  https://www.khanacademy.org/computing/computer-programming/programming-games-visualizations/programming-3d-shapes/a/rotating-3d-shapes
            scene.set_voxel(vec3(i,  j/10*ti.cos(-45) - (k-10) *ti.sin(-45) ,  (k-10) *ti.cos(-45) + j/10*ti.sin(-45) ), 1 , vec3(0.9, 0.9, 0.1)) 

initialize_voxels()

scene.finish()
