# from new_fem import solve_fem
# from dual_fem import solve_fem as solve_dual_fem
from h_adaptive import h_adaptive_fem
import numpy
from matplotlib import pyplot as plt
from helpers import new_norm, dual_norm, f_norm
from PyQt5.QtWidgets import *
from PyQt5 import QtCore
import sys

a = 0
b = 1

nodes = numpy.linspace(a, b, 3, endpoint=True)

# m = lambda x: 1
# beta = lambda x: 1
# sigma = lambda x: 1
# f = lambda x: x ** 2 + 2 * x - 2
# _u = 3
# alpha = 1

# m = lambda x: 1
# beta = lambda x: 1
# sigma = lambda x: 1
# f = lambda x: (1 + numpy.pi ** 2) * numpy.sin(numpy.pi * x) + numpy.pi * numpy.cos(numpy.pi * x)
# _u = -numpy.pi
# alpha = 1

# m = lambda x: 1
# sigma = lambda x: 1000
# f = lambda x: 1000
# _u = 0# m = lambda x: 1
# beta = lambda x: 1
# sigma = lambda x: 1
# f = lambda x: (1 + numpy.pi ** 2) * numpy.sin(numpy.pi * x) + numpy.pi * numpy.cos(numpy.pi * x)
# _u = -numpy.pi
# alpha = 1

# alpha = 10e7

# m = lambda x: 1
# beta = lambda x: 10**3 * (1 - x ** 7)
# sigma = lambda x: 10**3
# alpha = 10 ** 12
# _u = 0
# a = 0
# b = 1
#
# f = lambda x: 1000

m = lambda x: 1
beta = lambda x: 3000 * (2/3 - x)
sigma = lambda x: 1
alpha = 10 ** 12
_u = 0
a = 0
b = 1

f = lambda x: 3000

accuracy = 0.1

states = []

states = h_adaptive_fem(m, beta, sigma, f, alpha, _u, nodes, accuracy, states)

def draw_error(row):
    nodes = states[row.row()].nodes
    size = states[row.row()].size - 1
    u_h = states[row.row()].straight_norms
    phi_h = states[row.row()].dual_norms
    f_norms = states[row.row()].f_norms
    norm_error = [f_norms[i] - u_h[i] - phi_h[i] for i in range(size)]
    error = states[row.row()].error_norms
    xs = []
    ys = []
    nodes_y = []
    nodes_e = []
    e_s = []
    for i in range(size):
        x = nodes[i]
        xs.append(x)
        xs.append(nodes[i + 1])
        ys.append(error[i])
        ys.append(error[i])
        nodes_y.append(error[i])
        nodes_e.append(norm_error[i])
        e_s.append(norm_error[i])
        e_s.append(norm_error[i])
    nodes_y.append(nodes_y[-1])
    nodes_e.append(nodes_e[-1])
    plt.plot(xs, ys, 'b', nodes, nodes_y, 'rs')
    h = nodes[-1] - nodes[0]
    plt.xlim([nodes[0] - 0.05 * h, nodes[-1] + 0.05 * h])
    plt.show()
    plt.plot(xs, e_s, 'g', nodes, nodes_e, 'y^')
    h = nodes[-1] - nodes[0]
    plt.xlim([nodes[0] - 0.05 * h, nodes[-1] + 0.05 * h])
    plt.show()

def draw(row):
    xs = []
    ys = []
    yds = []
    i = row.row()
    size = states[i].size
    nodes = states[i].nodes
    u_h = sum(states[i].straight_norms)
    phi_h = sum(states[i].dual_norms)
    norm_error = sum(states[i].f_norms) - u_h - phi_h
    error = sum(states[i].error_norms)
    un = states[row.row()].solution
    dual = states[row.row()].dual_solution

    nodes_0 = [0 for i in range(size)]
    nodes_y = [un(nodes[i]) for i in range(size)]
    nodes_yd = [dual(nodes[i]) for i in range(size)]
    size = 1000
    h = (b - a) / size
    for i in range(size + 1):
        x = a + h * i
        xs.append(x)
        ys.append(un(x))
        yds.append(dual(x))
    plt.plot(xs, ys, 'b', nodes, nodes_y, 'b^', nodes, nodes_0, 'rs')
    h = nodes[-1] - nodes[0]
    plt.xlim([nodes[0] - 0.05 * h, nodes[-1] + 0.05 * h])
    plt.show()
    plt.plot(xs, yds, 'g', nodes, nodes_yd, 'g^', nodes, nodes_0, 'rs')
    plt.xlim([nodes[0] - 0.05 * h, nodes[-1] + 0.05 * h])
    plt.show()


# ui pyqt
app = QApplication(sys.argv)

listView = QTableWidget()
listView.setRowCount(len(states))
listView.setColumnCount(8)
# size, solution, dual_solution, nodes, norm, dual, fn, error
listView.setHorizontalHeaderItem(0, QTableWidgetItem("Size"))
listView.setHorizontalHeaderItem(1, QTableWidgetItem("||u_h||^2"))
listView.setHorizontalHeaderItem(2, QTableWidgetItem("||e_h||^2"))
listView.setHorizontalHeaderItem(3, QTableWidgetItem("||e_h||^2/||u_h||^2"))
listView.setHorizontalHeaderItem(4, QTableWidgetItem("p"))
listView.setHorizontalHeaderItem(5, QTableWidgetItem("||phi_h||^2"))
listView.setHorizontalHeaderItem(6, QTableWidgetItem("norm error"))
listView.setHorizontalHeaderItem(7, QTableWidgetItem("norm error, %"))
print("""
\\begin{table}[H]
\\centering
\\begin{tabular}{|l|l|l|l|l|l|l|l|}
\\hline
N   & $ ||u_h||_V^2 $ & $ ||e_h||_V^2 $ & $ \\frac{||e_h||_V}{||u_h||_V}, \\% $ & $ p $ & $ ||\\phi_h||^2 $ & $ ||f||^2 - ||u_h||^2 - ||\\phi_h||^2 $ & $ \\sqrt{\\frac{||f||^2 - ||u_h||^2 - ||\\phi_h||^2}{||u_h||^2 + ||\\phi_h||^2}} \\% $ \\\\ \\hline""")
listView.setEditTriggers(QAbstractItemView.NoEditTriggers)
for i in range(len(states)):
    size = states[i].size
    u_h = sum(states[i].straight_norms)
    phi_h = states[i].dual_norm
    fn = states[i].fn
    norm_error = fn - u_h - phi_h
    norm_error_percent = numpy.sqrt(abs(norm_error / (u_h + phi_h))) * 100
    error = sum(states[i].error_norms)
    error_percent = numpy.sqrt(error / u_h) * 100
    p = ''
    if i > 0:
        p = numpy.log(numpy.sqrt(sum(states[i - 1].error_norms) / error)) / numpy.log(size / states[i - 1].size)
    print("{0} & {1:.5} & {2:.5} & {3:.5} & {4:.5} & {5:.5} & {6:.5} & {7:.5} \\\\ \\hline".format(size - 1, u_h, error, error_percent, p, phi_h, norm_error, norm_error_percent))
    listView.setItem(i, 0, QTableWidgetItem("{0}".format(size - 1)))
    listView.setItem(i, 1, QTableWidgetItem("{0:.5}".format(u_h)))
    listView.setItem(i, 2, QTableWidgetItem("{0:.5}".format(error)))
    listView.setItem(i, 3, QTableWidgetItem("{0:.5}".format(error_percent)))
    listView.setItem(i, 4, QTableWidgetItem("{0:.5}".format(p)))
    listView.setItem(i, 5, QTableWidgetItem("{0:.5}".format(phi_h)))
    listView.setItem(i, 6, QTableWidgetItem("{0:.5}".format(norm_error)))
    listView.setItem(i, 7, QTableWidgetItem("{0:.5}".format(norm_error_percent)))

print("""
\\end{tabular}
\\caption{Характеристики апроксимацiї на рівномірній сiтцi.}
\\end{table}
""")
listView.doubleClicked.connect(draw)
listView.setWindowState(QtCore.Qt.WindowMaximized)
listView.show()
sys.exit(app.exec_())