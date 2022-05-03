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
    for i, j, k in ti.ndrange((-n, n), (-3, 3), (-n, n)):
        """
        x = ivec3(i, j, k)
        if  x.dot(x) < n * n * 0.5:
            if n * n * 0.4 < x.dot(x):
                scene.set_voxel(vec3(i, j, k), 1, vec3(0.9, 0.3, 0.3))
            else:
                if -10<i<10 and -10<j<10: 
                    scene.set_voxel(vec3(i+15, j-20, -17-j+i), 1 , vec3(0.9, 0.9, 0.1)) 
                    scene.set_voxel(vec3(i-15, j-20, -15-i), 1 , vec3(0.9, 0.9, 0.1)) 
                    scene.set_voxel(vec3(i, j-20, -20-j), 1 , vec3(0.9, 0.9, 0.1)) 
        """
        
        if ti.pow(i*0.05, 2) + ti.pow(k*0.05 - ti.pow((i*0.05)**2, 0.333), 2) <= 1:
            #scene.set_voxel(vec3(i, 5*ti.sin(i*0.05)**2 + 5*ti.cos(k*0.05)**2,  -k), 1 , vec3(0.9, 0.9, 0.1)) 
            #scene.set_voxel(vec3(i,  5*ti.sin(k*0.05),  -k), 1 , vec3(0.9, 0.9, 0.1)) 
            scene.set_voxel(vec3(i,  j,  -k), 1 , vec3(0.9, 0.9, 0.1)) 
            #scene.set_voxel(vec3(i,  -5-5*ti.sin(k*0.05),  -k), 1 , vec3(0.9, 0.9, 0.1)) 
    """
    
    n = 3
    for i, j, k in ti.ndrange((-10*n, 10*n), (-4*n, 4*n), (-4*n, 4*n)):
        is_light = int(ti.random() > 0.7)
        if k%2==0:
            scene.set_voxel(vec3(i, j, k + 3*ti.sin(i* 0.2)), 1 + is_light , vec3(0.9, 0.9, 0.1)) 
        else:
            scene.set_voxel(vec3(i, j, k + 3*ti.sin(i* 0.2)), 1 + is_light , vec3(0.1, 0.1, 0.1)) 
    """

initialize_voxels()

scene.finish()
