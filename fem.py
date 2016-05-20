from helpers import *
from scipy.misc import derivative
import scipy.integrate as integrate
import numpy
from matplotlib import pyplot as plt
import sys
from PyQt5.QtWidgets import *
from PyQt5 import QtCore


class State:
    def __init__(self, size, norm_u, e_l, derivative_norm_u, e_h, func, nodes):
        self.size = size
        self.norm_u = norm_u
        self.e_l = e_l
        self.derivative_norm_u = derivative_norm_u
        self.e_h = e_h
        self.function = func
        self.nodes = nodes



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
            result = \
            integrate.quad(lambda x: derivative(p, x, dx=1e-6) * basis[i](x, True) * basis[k](x), x_left, x_right)[0]
            result += integrate.quad(lambda x: p(x) * basis[i](x, True) * basis[k](x, True), x_left, x_right)[0]
            result += integrate.quad(lambda x: q(x) * basis[i](x, True) * basis[k](x), x_left, x_right)[0]
            result += integrate.quad(lambda x: r(x) * basis[i](x) * basis[k](x), x_left, x_right)[0]
            result += beta * basis[k](nodes[-1]) * basis[i](nodes[-1]) + alpha * basis[k](nodes[0]) * basis[i](nodes[0])
            matrix[k][i] = result
        b[k] = scalar_product(f, basis[k], x_left, x_right) + beta * B * basis[k](nodes[-1]) + alpha * A * basis[k](
            nodes[0])
    solution = numpy.linalg.solve(matrix, b)
    return lambda x: create_function(x, solution)


def h_adaptive_fem(f, p, q, r, alpha, beta, A, B, basis, nodes, accuracy, states):
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
        f_i -= integrate.quad(lambda x: r(x) * bubble_basis[i](x) * solution(x), nodes[i], nodes[i + 1])[0]
        f_i += p(nodes[-1]) * bubble_basis[i](nodes[-1]) * derivative(solution, nodes[-1], dx=1e-6)
        f_i -= p(nodes[0]) * bubble_basis[i](nodes[0]) * derivative(solution, nodes[0], dx=1e-6)
        coefficients.append(f_i / e_i)
    eh = [coefficients[i] ** 2 * derivative_norm(bubble_basis[i], nodes[i], nodes[i + 1]) ** 2 for i in range(size)]
    el = [coefficients[i] ** 2 * norm(bubble_basis[i], nodes[i], nodes[i + 1]) ** 2 for i in range(size)]
    e_average = sum(eh)
    uh_h = sum([derivative_norm(solution, nodes[i], nodes[i + 1]) ** 2 for i in range(size)])
    uh_l = sum([norm(solution, nodes[i], nodes[i + 1]) ** 2 for i in range(size)])
    new_nodes = []
    needs_repeat = False
    for i in range(size):
        new_nodes.append(nodes[i])
        deviation = numpy.sqrt(size * eh[i] / (sum(eh) + uh_h))
        if deviation > accuracy:
            new_nodes.append((nodes[i] + nodes[i + 1]) / 2)
            needs_repeat = True
    new_nodes.append(nodes[-1])
    state = State(len(nodes), numpy.sqrt(uh_l), numpy.sqrt(sum(el)), numpy.sqrt(uh_h), numpy.sqrt(sum(eh)), solution, nodes)
    states.append(state)
    print("average:{0}\t size:{1}".format(e_average, size))
    if needs_repeat:
        new_basis = create_basis(new_nodes)
        return h_adaptive_fem(f, p, q, r, alpha, beta, A, B, new_basis, new_nodes, accuracy, states)
    else:
        return solution


# p = lambda x: 1
# q = lambda x: 10**3 * (1 - x ** 7)
# r = lambda x: -10**3
# alpha = 10 ** 12
# beta = 10 ** 12
# a = -1
# b = 1
# A = 0
# B = 0
#
#
# def func(x):
#     return 1000


p = lambda x: 1
q = lambda x: 20
r = lambda x: 0
alpha = 10 ** 12
beta = 10 ** 12
a = 0
b = 5
A = 0
B = 0


def func(x):
    return 100

nodes = numpy.linspace(a, b, 3, endpoint=True)

basis = create_basis(nodes)

states = []

s = h_adaptive_fem(func, p, q, r, alpha, beta, A, B, basis, nodes, 0.1, states)

def u_real(x):
    return 5 * (x * numpy.exp(100) - x - 5 * numpy.exp(20*x) + 5) / (numpy.exp(100)-1)

def draw(row):
    xs = []
    ys = []

    nodes = states[row.row()].nodes
    un = states[row.row()].function
    size = states[row.row()].size
    nodes_0 = [0 for i in range(size)]
    nodes_y = [un(nodes[i]) for i in range(size)]

    yf = []
    size = 1000
    h = (b - a) / size
    for i in range(size + 1):
        x = a + h * i
        xs.append(x)
        ys.append(un(x))
        yf.append(u_real(x))
    plt.plot(xs, ys, 'b', xs, yf, 'g--', nodes, nodes_y, 'b^', nodes, nodes_0, 'rs')
    h = nodes[-1] - nodes[0]
    plt.xlim([nodes[0] - 0.05 * h, nodes[-1] + 0.05 * h])
    h = nodes[-1] - nodes[0]
    plt.show()

# ui pyqt
app = QApplication(sys.argv)

listView = QTableWidget()
listView.setRowCount(len(states))
listView.setColumnCount(7)
listView.setHorizontalHeaderItem(0, QTableWidgetItem("Size"))
listView.setHorizontalHeaderItem(1, QTableWidgetItem("Uh_L"))
listView.setHorizontalHeaderItem(2, QTableWidgetItem("e_L"))
listView.setHorizontalHeaderItem(3, QTableWidgetItem("Error L %"))
listView.setHorizontalHeaderItem(4, QTableWidgetItem("Uh_H"))
listView.setHorizontalHeaderItem(5, QTableWidgetItem("e_H"))
listView.setHorizontalHeaderItem(6, QTableWidgetItem("Error H %"))
for i in range(len(states)):
    listView.setItem(i, 0, QTableWidgetItem("{0}".format(states[i].size)))
    listView.setItem(i, 1, QTableWidgetItem("{0:.5}".format(states[i].norm_u)))
    listView.setItem(i, 2, QTableWidgetItem("{0:.5}".format(states[i].e_l)))
    listView.setItem(i, 3, QTableWidgetItem("{0:.5}".format(states[i].e_l / states[i].norm_u * 100)))
    listView.setItem(i, 4, QTableWidgetItem("{0:.5}".format(states[i].derivative_norm_u)))
    listView.setItem(i, 5, QTableWidgetItem("{0:.5}".format(states[i].e_h)))
    listView.setItem(i, 6, QTableWidgetItem("{0:.5}".format(states[i].e_h / states[i].derivative_norm_u * 100)))
listView.doubleClicked.connect(draw)
listView.setWindowState(QtCore.Qt.WindowMaximized)
listView.show()
sys.exit(app.exec_())