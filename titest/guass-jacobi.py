import taichi as ti
import random

ti.init()

n = 20

A = ti.field(ti.f32, (n, n))
x = ti.field(ti.f32, n)
new_x = ti.field(ti.f32, n)
b = ti.field(ti.f32, n)

#print(x)
@ti.kernel
def iterate():
    for i in range(n):
        r = b[i]
        for j in range(n):
            #if i != j:
                # r = r - A[i, j-1] * x [j-1] - A[i] 
            #    r -= A[i, j] * x[j]
            if i > j:
                r -= A[i, j] * x[j]
            if i < j:
                r -= A[i, j] * new_x[j]
                
        new_x[i] = r / A[i, i]
        
    for i in range(n):
        x[i] = new_x[i]

@ti.kernel
def residual() -> ti.f32:
    res = 0.0
    
    for i in range(n):
        r = b[i] * 1.0
        for j in range(n):
            r -= A[i, j] * x[j]
        res += r * r
        
    return res

for i in range(n):
    for j in range(n):
        A[i, j] = random.random() - 0.5

    A[i, i] += n * 0.1
    
    b[i] = random.random() * 100

for i in range(100):
    iterate()
    print(f'iter {i}, residual={residual():0.10f}')
    

for i in range(n):
    lhs = 0.0
    for j in range(n):
        lhs += A[i, j] * x[j]
    assert abs(lhs - b[i]) < 1e-4
