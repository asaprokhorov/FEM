from helpers import *
import matplotlib.pyplot as plt

class State:
    def __init__(self, size, error, func, nodes):
        self.size = size
        self.error = error
        self.function = func
        self.nodes = nodes


def LSM(f, nodes, basis):
    def create_function(x, q, basis):
        size = len(q)
        y = 0
        for i in range(size):
            y += basis[i](x)*q[i]
        return y
    size = len(nodes)
    matrix = numpy.zeros((size, size))
    b = numpy.zeros(size)
    matrix[0][0] = scalar_product(basis[0], basis[0], nodes[0], nodes[1])
    matrix[0][1] = scalar_product(basis[0], basis[1], nodes[0], nodes[1])
    b[0] = scalar_product(f, basis[0], nodes[0], nodes[1])
    for i in range(1, size - 1):
        matrix[i][i-1] = scalar_product(basis[i], basis[i-1], nodes[i-1], nodes[i])
        matrix[i][i] = scalar_product(basis[i], basis[i], nodes[i-1], nodes[i + 1])
        matrix[i][i+1] = scalar_product(basis[i], basis[i+1], nodes[i], nodes[i + 1])
        b[i] = scalar_product(f, basis[i], nodes[i-1], nodes[i+1])

    matrix[-1][-1] = scalar_product(basis[-1], basis[-1], nodes[-2], nodes[-1])
    matrix[-1][-2] = scalar_product(basis[-1], basis[-2], nodes[-2], nodes[-1])
    b[-1] = scalar_product(f, basis[-1], nodes[-2], nodes[-1])

    result = numpy.linalg.solve(matrix, b)
    return lambda x: create_function(x, result, basis)


def draw(state, func, a, b):
    xs = []
    ys = []

    nodes_y = [func(state.nodes[i]) for i in range(state.size)]

    yf = []
    size = 100
    h = (b - a) / size
    for i in range(size + 1):
        x = a + h * i
        xs.append(x)
        ys.append(state.function(x))
        yf.append(func(x))
    plt.plot(xs, ys)
    plt.plot(xs, yf)
    plt.plot(state.nodes, nodes_y, 'go')

    plt.show()


def h_adaptive_LSM(f, nodes, basis, accuracy, states):
    un = LSM(f, nodes, basis)
    common_error = error(f, un, nodes[0], nodes[-1]) /  norm(f, nodes[0], nodes[-1])
    size = len(nodes)
    state = State(size, common_error, un, nodes)
    states.append(state)
    if common_error < accuracy:
        return un
    errors = [error(f, un, nodes[i], nodes[i + 1]) for i in range(size - 1)]
    error_average = 0
    for i in range(len(errors)):
        error_average += errors[i] ** 2
    error_average = numpy.sqrt(error_average / len(errors))
    max_error = max(errors)
    print("points:{0}\terror:{1}\taverage:{2}\tmax:{3}".format(size, common_error, error_average, max_error))
    new_nodes = []
    for i in range(size - 1):
        new_nodes.append(nodes[i])
        if errors[i] / error_average >  1 + accuracy:
            new_nodes.append(nodes[i] + (nodes[i + 1] - nodes[i]) / 2)
    new_nodes.append(nodes[-1])
    return h_adaptive_LSM(f, new_nodes, create_basis(new_nodes), accuracy, states)
