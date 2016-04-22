from helpers import *
from scipy.misc import derivative
import scipy.integrate as integrate
import numpy
from matplotlib import pyplot as plt


def fem(f, p, q, r, alpha, beta, A, B, basis, nodes):
    def create_function(x, q):
        size = len(q)
        y = 0
        for i in range(size):
            y += basis[i](x) * q[i]
        return y
    size = len(basis)
    matrix = numpy.zeros((size, size))
    b = numpy.zeros(size)
    for i in range(size):
        x_left = nodes[i - 1] if i > 0 else nodes[i]
        x_right = nodes[i + 1] if i < size - 1 else nodes[i]
        for j in range(size):
            result = integrate.quad(lambda x: derivative(p, x, dx=1e-6) * basis[i](x, True) * basis[j](x), x_left, x_right)[0]
            result += integrate.quad(lambda x: p(x) * basis[i](x, True) * basis[j](x, True), x_left, x_right)[0]
            result += integrate.quad(lambda x: q(x) * basis[i](x, True) * basis[j](x), x_left, x_right)[0]
            result += integrate.quad(lambda x: r(x) * basis[i](x) * basis[j](x), x_left, x_right)[0]
            result += beta * basis[i](nodes[-1]) * basis[j](nodes[-1]) + alpha * basis[i](nodes[0]) * basis[j](nodes[0])
            matrix[i][j] = result
        b[i] = scalar_product(f, basis[i], x_left, x_right) + beta * B * basis[i](nodes[-1]) + alpha * A * basis[i](nodes[0])
    print(matrix)
    solution = numpy.linalg.solve(matrix, b)
    return lambda x: create_function(x, solution)

p = lambda x: 1
q = lambda x: 10**3 * (1 - x**7)
r = lambda x: -10**3
alpha = 0
beta = 0
A = 0
B = 0
a = -1
b = 1
def func(x):
    return 10**3

def u(x):
    return 5 * (x * numpy.exp(100) - x - 5 * numpy.exp(20 * x) + 5) / (numpy.exp(100) - 1)

nodes = numpy.linspace(a, b, 20, endpoint=True)

basis = create_basis(nodes)

s = fem(func, p, q, r, alpha, beta, A, B, basis, nodes)

xs = numpy.linspace(a, b, 100, endpoint=True)

ys = [s(i) for i in xs]
plt.plot(xs, ys)
plt.show()