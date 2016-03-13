from helpers import *


class State:
    def __init__(self, size, error, func):
        self.size = size
        self.error = error
        self.function = func


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
    for i in range(size):
        for j in range(size):
            matrix[i][j] = scalar_product(basis[i], basis[j], nodes[0], nodes[-1])
        b[i] = scalar_product(f, basis[i], nodes[0], nodes[-1])
    result = numpy.linalg.solve(matrix, b)
    return lambda x: create_function(x, result, basis)


def h_adaptive_LSM(f, nodes, basis, accuracy, states):
    un = LSM(f, nodes, basis)
    size = len(nodes)
    common_error = error(f, un, nodes[0], nodes[-1])
    states.append(State(size, common_error, un))
    print("points:{0}\terror:{1}".format(size, common_error))
    new_nodes = []
    needs_repeat = False
    for i in range(size - 1):
        new_nodes.append(nodes[i])
        if error(f, un, nodes[i], nodes[i + 1]) > accuracy:
            new_nodes.append((nodes[i + 1] + nodes[i]) / 2)
            needs_repeat = True
    new_nodes.append(nodes[-1])
    if needs_repeat:
        return h_adaptive_LSM(f, new_nodes, create_basis(new_nodes), accuracy, states)
    else:
        return un
