from re import S
import taichi as ti
ti.init()

a = ti.field(ti.f32, (42, 63))
b = ti.Vector.field(3, ti.f32, 4)
c =  ti.Matrix.field(2, 2, ti.f32, (3, 5))

loss = ti.field(ti.f32, ())

a[3, 4] = 1
print("a[3, 4]", a[3, 4])

b[2] = [6, 7, 8]
print("b[0]", b[0], "b[0]", b[2])

loss[None] = 3
print(loss[None])


@ti.kernel
def hello(i: ti.i32):
    a = 40
    print('hello world', a + i)

hello(2)



@ti.kernel
def calc() -> ti.i32:
    s = 0
    for i in range(10):
        s += i
    return S

print("calc", calc())