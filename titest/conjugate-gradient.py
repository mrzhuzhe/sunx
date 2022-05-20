# not work yet
from numpy import ndarray
import taichi as ti
import random

ti.init()

n = 20

A = ti.field(ti.f32, (n, n))
#A = ti.Vector.field(n, ti.f32, (n,))
x = ti.field(ti.f32, n)
#x = ti.Vector.field(n, ti.f32, ())
b = ti.field(ti.f32, n)
alpha = ti.field(ti.f32, ())
beta = ti.field(ti.f32, ())
rk = ti.field(ti.f32, n)
pk = ti.field(ti.f32, n)
#apk = ti.field(ti.f32, n)
#rint(A.shape, x)
#print(A @ x )

@ti.kernel
def matmul(res: ti.template(), a: ti.template(), b: ti.template()):
    for i in range(n):
        for j in range(n):
            res[i] += a[i, j] * b[j]

@ti.kernel
def add(ans: ti.template(), a: ti.template(), k: ti.f32, b: ti.template()):
    for i in ans:
        ans[i] = a[i] + k * b[i]


@ti.kernel
def dot(a: ti.template(), b: ti.template()) -> ti.f32:
    ans = 0.0
    for i in range(n):
        ans += a[i] * (b[i])
    return ans


def cg(tol=1e-5):
    apk = ti.field(ti.f32, n)
    matmul(apk, A, pk)
    rkrk = dot(rk, rk)

    alpha = rkrk / dot(pk, apk)
    #x = x + alpha * pk
    add(x, x, alpha, pk)
    #rk = rk + alpha * apk
    add(rk, rk, alpha, apk)
    beta = dot(rk, rk) / rkrk
    #pk = -rk + beta * pk
    add(pk, rk, -beta, pk)
    add(pk, ti.field(ti.f32, n), -1, pk)

def iterate():
    cg()
    print(f'iter {i}, residual={residual():0.10f}')

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

    
#print(A, b, x)
matmul(rk, A, x)
add(rk, rk, -1, b)
add(pk, ti.field(ti.f32, n), -1, rk)

for i in range(100):
    iterate()
    

for i in range(n):
    lhs = 0.0
    for j in range(n):
        lhs += A[i, j] * x[j]
    assert abs(lhs - b[i]) < 1e-4

