import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from itertools import product
from warnings import warn
from sklearn.datasets import make_spd_matrix


def LinearCG(A, b, x0, tol=1e-5):
    xk = x0
    rk = np.dot(A, xk) - b
    pk = -rk
    rk_norm = np.linalg.norm(rk)
    
    num_iter = 0
    curve_x = [xk]
    while rk_norm > tol:
        apk = np.dot(A, pk)
        rkrk = np.dot(rk, rk)
        
        alpha = rkrk / np.dot(pk, apk)
        xk = xk + alpha * pk
        rk = rk + alpha * apk
        beta = np.dot(rk, rk) / rkrk
        pk = -rk + beta * pk
        print(rkrk)
        num_iter += 1
        curve_x.append(xk)
        rk_norm = np.linalg.norm(rk)
        print('Iteration: {} \t x = {} \t residual = {:.4f}'.
              format(num_iter, xk, rk_norm))
    
    print('\nSolution: \t x = {}'.format(xk))
        
    return np.array(curve_x)

np.random.seed(0)
A = make_spd_matrix(4, random_state=0)
x_star = np.random.random(4)
b = np.dot(A, x_star)

print('A\n', A, '\n')
print('b\n', b, '\n')
print('The solution x* should be\n', x_star)

x0 = np.array([-3, -4, -3, -4])
xs = LinearCG(A, b, x0)