from scene import Scene
import taichi as ti
from taichi.math import *

scene = Scene(exposure=1)
scene.set_floor(-0.85, (1.0, 1.0, 1.0))
scene.set_background_color((0.5, 0.5, 0.4))
scene.set_directional_light((1, 1, -1), 0.2, (1, 0.8, 0.6))

@ti.kernel
def initialize_voxels():
    n = 30
    """   
    for x, y, z in ti.ndrange((-n, n), (-n, n), (-n, n)):
        a = ivec3(x, y, z)
        b = ivec3(10 * (y - x), (x* (28 - z) -y),  (x * y - 8/3 * z)) 
        if (a - b).norm() < 25:
            scene.set_voxel(vec3(x, y, z), 1 , vec3(0.9, 0.9, 0.1)) 
    """
    x0 = ivec3(0, 1, 1)
    x = x0[0]
    y = x0[1]
    z = x0[2]
    dt = 0.1
    for i in ti.static(range(60)):
        j = ivec3(10 * (y - x), (x* (28 - z) -y),  (x * y - 8/3 * z)) * dt
        x = x + j[0]
        y = x + j[1]
        z = x + j[2]
        scene.set_voxel(vec3(x, y, z), 1 , vec3(0.9, 0.9, 0.1)) 

initialize_voxels()

scene.finish()
