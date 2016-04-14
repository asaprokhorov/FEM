import numpy
import scipy.integrate as integrate


def scalar_product(u, v, a, b):
    return integrate.quad(lambda x: numpy.float64(u(x) * v(x)), a, b, epsabs=1e-9, epsrel=1e-9)[0]


def norm(f, a, b):
    return numpy.sqrt(scalar_product(f, f, a, b))


def error(f, u, a, b):
    return norm(lambda x: f(x) - u(x), a, b)


def create_single_basis(i, nodes):
    return lambda x: x**i


def create_single_courant_basis(i, nodes):

    def courant(x, i, nodes):
        x_i = nodes[i]
        x_left = nodes[i-1] if i > 0 else nodes[0] - (nodes[1] - nodes[0])
        x_right = nodes[i+1] if i < len(nodes) - 1 else nodes[-1] + (nodes[-1] - nodes[-2])
        if x_left < x <= x_i:
            return (x - x_left) / (x_i - x_left)
        elif x_i < x < x_right:
            return (x_right - x) / (x_right - x_i)
        else:
            return 0
    return lambda x: courant(x, i, nodes)



def create_basis(nodes):
    size = len(nodes)
    basis = []
    for i in range(size):
        basis.append(create_single_courant_basis(i, nodes))
    return basis
