from scene import Scene
import taichi as ti
from taichi.math import *

scene = Scene(exposure=1)
scene.set_floor(-0.85, (1.0, 1.0, 1.0))
scene.set_background_color((0.5, 0.5, 0.4))
scene.set_directional_light((1, 1, -1), 0.2, (1, 0.8, 0.6))

# rotate k   x cos - y sin  y cos + xq sin  https://www.khanacademy.org/computing/computer-programming/programming-games-visualizations/programming-3d-shapes/a/rotating-3d-shapes
ti.func
def rotate2d(x, y, r):
    return x*ti.cos(r) - y*ti.sin(r) , y*ti.cos(r) + x*ti.sin(r)

@ti.kernel
def initialize_voxels():
    """
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
    """
    
    #"""
    n = 500    
    r = 120
    dt = 0.01
    for i in range(n):
        x = ti.random() * r - r/2
        y = 0
        z = 0
        for j in ti.static(range(5)):            
            x = x + 10 * (y - x) * dt
            y = y + (x* (28 - z) -y) * dt
            z = z + (x * y - 8/3 * z) * dt
            scene.set_voxel(vec3(x, y, z), 1 , vec3(0, 0, 1)) 
    #"""
    


initialize_voxels()

scene.finish()
