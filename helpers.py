import numpy
import scipy.integrate as integrate


def scalar_product(u, v, a, b):
    return integrate.quad(lambda x: u(x) * v(x), a, b)[0]


def norm(f, a, b):
    return numpy.sqrt(scalar_product(f, f, a, b))


def error(f, u, a, b):
    return norm(lambda x: f(x) - u(x), a, b) / norm(f, a, b)


def create_single_basis(i):
    return lambda x: x**i


def create_single_courant_basis(i, nodes):
    def courant(x, i, nodes):
        if i == 0:
            if nodes[1] < x < nodes[0]:
                return 0
            else:
                return (nodes[1] - x) / (nodes[1] - nodes[0])
        elif i == len(nodes) - 1:
            if nodes[-1] < x < nodes[-2]:
                return 0
            else:
                return (x - nodes[-2]) / (nodes[-1] / nodes[-2])
        else:
            if nodes[i + 1] < x < nodes[i - 1]:
                return 0
            elif x <= nodes[i]:
                return (x - nodes[i - 1]) / (nodes[i] - nodes[i - 1])
            else:
                return (nodes[i + 1] - x) / (nodes[i + 1] - nodes[i])
    return lambda x: courant(x, i, nodes)


def create_basis(nodes):
    size = len(nodes)
    basis = []
    for i in range(size):
        basis.append(create_single_courant_basis(i, nodes))
    return basis
