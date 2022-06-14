from scene import Scene
import taichi as ti
from taichi.math import *

color = (0.6, 0.2, 0.2)

scene = Scene(voxel_edges=0, exposure=10)
scene.set_floor(-0.05, (1.0, 1.0, 1.0))
scene.set_directional_light((-1, 1, -1), 0.1, (0.5, 0.5, 0.5))


@ti.func
def makeup(end, num):
    for m in range(1, end):
        for j in range(-3, 4):
            scene.set_voxel(vec3(41+m, num, j), 1, vec3(0.6, 0.2, 0.2))

@ti.kernel
def initialize_voxels():
    for x in ti.ndrange((0, 42)):
        if x % 14 == 0 or x == 4:
            continue
        for i in range(-3, 4):
            for j in range(-3, 4):
                scene.set_voxel(vec3(x, 56+i, j), 1, vec3(0.6, 0.2, 0.2))
    makeup(7, 59); makeup(6, 58); makeup(5, 57); makeup(4, 56); makeup(3, 55); makeup(2, 54)
    for y in ti.ndrange((0, 60)):
        if y % 15 == 0 and y < 50:
            continue
        for i in range(-3, 4):
            for j in range(-3, 4):
                scene.set_voxel(vec3(i, y, j), 1, vec3(0.6, 0.2, 0.2))
    for z in ti.ndrange((-49, 4)):
        if z == -34 or z == -19:
            continue
        for i in range(-3, 4):
            for j in range(-3, 4):
                scene.set_voxel(vec3(i, j, z), 1, vec3(0.6, 0.2, 0.2))

initialize_voxels()
scene.finish()