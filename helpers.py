import numpy
import scipy.integrate as integrate
from scipy.misc import derivative


def scalar_product(u, v, a, b):
    return integrate.quad(lambda x: numpy.float64(u(x) * v(x)), a, b)[0]


# def norm(f, a, b):
#     return numpy.sqrt(scalar_product(f, f, a, b))
#
#
# def derivative_norm(f, a, b):
#     return numpy.sqrt(integrate.quad(lambda x: derivative(f, x, dx=1e-6) ** 2, a, b)[0])
#
#
# def error(f, u, a, b):
#     return norm(lambda x: f(x) - u(x), a, b)


def create_single_courant_basis(i, nodes):
    def courant(x, i, nodes):
        x_i = nodes[i]
        x_left = nodes[i - 1] if i > 0 else nodes[0] - (nodes[1] - nodes[0])
        x_right = nodes[i + 1] if i < len(nodes) - 1 else nodes[-1] + (nodes[-1] - nodes[-2])
        if x_left < x <= x_i:
            return (x - x_left) / (x_i - x_left)
        elif x_i < x < x_right:
            return (x_right - x) / (x_right - x_i)
        else:
            return 0

    def courant_derivative(x, i, nodes):
        x_i = nodes[i]
        x_left = nodes[i - 1] if i > 0 else nodes[0] - (nodes[1] - nodes[0])
        x_right = nodes[i + 1] if i < len(nodes) - 1 else nodes[-1] + (nodes[-1] - nodes[-2])
        if x_left < x <= x_i:
            return 1. / (x_i - x_left)
        elif x_i < x < x_right:
            return -1. / (x_right - x_i)
        else:
            return 0

    return lambda x, derivative=False: courant(x, i, nodes) if not derivative else courant_derivative(x, i, nodes)


def create_basis(nodes):
    size = len(nodes)
    basis = []
    for i in range(size):
        basis.append(create_single_courant_basis(i, nodes))
    return basis


def create_single_bubble_basis(i, nodes):
    def bubble(x, i, nodes):
        x_left = nodes[i]
        x_right = nodes[i + 1]
        x_i = (nodes[i] + nodes[i + 1]) / 2
        if x_left < x <= x_i:
            return (x - x_left) / (x_i - x_left)
        elif x_i < x < x_right:
            return (x_right - x) / (x_right - x_i)
        else:
            return 0

    def bubble_derivative(x, i, nodes):
        x_left = nodes[i]
        x_right = nodes[i + 1]
        x_i = (nodes[i] + nodes[i + 1]) / 2
        if x_left < x <= x_i:
            return 1. / (x_i - x_left)
        elif x_i < x < x_right:
            return -1. / (x_right - x_i)
        else:
            return 0

    return lambda x, derivative=False: bubble(x, i, nodes) if not derivative else bubble_derivative(x, i, nodes)


def create_bubble_basis(nodes):
    size = len(nodes) - 1
    basis = []
    for i in range(size):
        basis.append(create_single_bubble_basis(i, nodes))
    return basis


class State:
    def __init__(self, size, solution, dual_solution, nodes,
                 straight_norms, dual_norms, dual_norm, f_norms,
                 error_norms, fn):
        self.size = size
        self.solution = solution
        self.dual_solution = dual_solution
        self.straight_norms = straight_norms
        self.dual_norms = dual_norms
        self.dual_norm = dual_norm
        self.nodes = nodes
        self.f_norms = f_norms
        self.error_norms = error_norms
        self.fn = fn

def new_norm(m, beta, sigma, alpha, solution, a, b, end):
    last = alpha * solution(end) ** 2 if b == end else 0
    return integrate.quad(
        lambda x: m(x) * derivative(solution, x, dx=1e-6) ** 2 + beta(x) * derivative(solution, x, dx=1e-6) * solution(
            x) + sigma(x) * solution(x) ** 2, a, b)[
               0] + last


def f_norm(sigma, f, alpha, _u, a, b, end):
    last = _u ** 2 * alpha if b == end else 0
    return integrate.quad(lambda x: f(x) ** 2 / sigma(x), a, b)[0] + last


def dual_norm(m, beta, sigma, f, alpha, _u, solution, a, b, end):
    last = solution(end) ** 2 / alpha if b == end else 0
    return integrate.quad(lambda x: (solution(x) ** 2) / m(x) + (derivative(solution, x, dx=1e-6) ** 2) / sigma(x)
                                     - beta(x) * solution(x) * derivative(solution, x, dx=1e-6) / (m(x) * sigma(x))
                          , a, b)[0] + last
