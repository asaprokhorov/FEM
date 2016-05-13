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
    for k in range(size):
        x_left = nodes[k - 1] if k > 0 else nodes[k]
        x_right = nodes[k + 1] if k < size - 1 else nodes[k]
        for i in range(size):
            result = integrate.quad(lambda x: derivative(p, x, dx=1e-6) * basis[i](x, True) * basis[k](x), x_left, x_right)[0]
            result += integrate.quad(lambda x: p(x) * basis[i](x, True) * basis[k](x, True), x_left, x_right)[0]
            result += integrate.quad(lambda x: q(x) * basis[i](x, True) * basis[k](x), x_left, x_right)[0]
            result += integrate.quad(lambda x: r(x) * basis[i](x) * basis[k](x), x_left, x_right)[0]
            result += beta * basis[k](nodes[-1]) * basis[i](nodes[-1]) + alpha * basis[k](nodes[0]) * basis[i](nodes[0])
            matrix[k][i] = result
        b[k] = scalar_product(f, basis[k], x_left, x_right) + beta * B * basis[k](nodes[-1]) + alpha * A * basis[k](
            nodes[0])
    solution = numpy.linalg.solve(matrix, b)
    return lambda x: create_function(x, solution)


def h_adaptive_fem(f, p, q, r, alpha, beta, A, B, basis, nodes, accuracy):
    solution = fem(f, p, q, r, alpha, beta, A, B, basis, nodes)

    size = len(nodes) - 1
    bubble_basis = create_bubble_basis(nodes)
    coefficients = []
    for i in range(size):
        e_i = \
            integrate.quad(lambda x: derivative(p, x, dx=1e-6) * bubble_basis[i](x) * bubble_basis[i](x, True),
                           nodes[i],
                           nodes[i + 1])[0]
        e_i += integrate.quad(lambda x: p(x) * bubble_basis[i](x, True) ** 2, nodes[i], nodes[i + 1])[0]
        e_i += integrate.quad(lambda x: q(x) * bubble_basis[i](x) * bubble_basis[i](x, True), nodes[i], nodes[i + 1])[0]
        e_i += integrate.quad(lambda x: r(x) * bubble_basis[i](x) ** 2, nodes[i], nodes[i + 1])[0]
        e_i -= p(nodes[-1]) * bubble_basis[i](nodes[-1], True) * bubble_basis[i](nodes[-1])
        e_i += p(nodes[0]) * bubble_basis[i](nodes[0], True) * bubble_basis[i](nodes[0])

        f_i = scalar_product(f, bubble_basis[i], nodes[i], nodes[i + 1])
        f_i -= \
            integrate.quad(lambda x: derivative(p, x, dx=1e-6) * bubble_basis[i](x) * derivative(solution, x, dx=1e-6),
                           nodes[i], nodes[i + 1])[0]
        f_i -= integrate.quad(
            lambda x: p(x) * bubble_basis[i](x, True) * derivative(solution, x, dx=1e-6), nodes[i],
            nodes[i + 1])[0]
        f_i -= integrate.quad(
            lambda x: q(x) * bubble_basis[i](x) * derivative(solution, x, dx=1e-6), nodes[i],
            nodes[i + 1])[0]
        f_i -= integrate.quad(lambda x: r(x) * bubble_basis[i](x, True) * solution(x), nodes[i], nodes[i + 1])[0]
        f_i += p(nodes[-1]) * bubble_basis[i](nodes[-1]) * derivative(solution, nodes[-1], dx=1e-6)
        f_i -= p(nodes[0]) * bubble_basis[i](nodes[0]) * derivative(solution, nodes[0], dx=1e-6)
        coefficients.append(f_i / e_i)
    eh = [coefficients[i] ** 2 * integrate.quad(lambda x: bubble_basis[i](x) ** 2, nodes[i], nodes[i + 1])[0] for i in range(size)]
    e_average = sum(eh)

    new_nodes = []
    new_nodes = []
    needs_repeat = False
    for i in range(size):
        new_nodes.append(nodes[i])
        deviation = numpy.sqrt(size * eh[i] / e_average)
        print(deviation)
        if deviation - 1 > accuracy:
            new_nodes.append((nodes[i] + nodes[i+1]) / 2)
            needs_repeat = True
    new_nodes.append(nodes[-1])
    print("average:{0}\t size:{1}".format(e_average ,len(new_nodes)))
    plt.plot(new_nodes, [solution(new_nodes[i]) for i in range(len(new_nodes))], 'go--')
    plt.show()
    if needs_repeat:
        new_basis = create_basis(new_nodes)
        return h_adaptive_fem(f, p, q, r, alpha, beta, A, B, new_basis, new_nodes, accuracy)
    else:
        return solution

p = lambda x: 1
q = lambda x: 10 ** 3 * (1 - x ** 7)
r = lambda x: -10 ** 3
alpha = 10 ** 12
beta = 10 ** 12
a = -1
b = 1
A = 0
B = 0


def func(x):
    return 10 ** 3


def u(x):
    return 5 * (x * numpy.exp(100) - x - 5 * numpy.exp(20 * x) + 5) / (numpy.exp(100) - 1)


nodes = numpy.linspace(a, b, 10, endpoint=True)

basis = create_basis(nodes)

s = h_adaptive_fem(func, p, q, r, alpha, beta, A, B, basis, nodes, 0.5)

xs = numpy.linspace(a, b, 100, endpoint=True)

ys = [s(i) for i in xs]
yu = [u(i) for i in xs]
plt.plot(xs, ys)
# plt.plot(xs, yu)
plt.show()
