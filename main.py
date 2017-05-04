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

m = lambda x: 1
sigma = lambda x: 1
f = lambda x: x#x ** 2 - 2
_u = 3
alpha = 1

# m = lambda x: 1
# sigma = lambda x: 1
# f = lambda x: (1 + numpy.pi ** 2) * numpy.sin(numpy.pi * x)
# _u = -numpy.pi
# alpha = 1


accuracy = 0.1

states = []

states = h_adaptive_fem(m, sigma, f, alpha, _u, nodes, accuracy, states)

def draw_error(row):
    f_norms = states[row.row()].f_norms
    norm = states[row.row()].norm
    dual_norm = states[row.row()].dual
    nodes = states[row.row()].nodes
    fn = states[row.row()].fn
    size = states[row.row()].size
    errors = states[row.row()].errors
    xs = []
    ys = []
    nodes_y = []
    for i in range(size - 1):
        x = nodes[i]
        xs.append(x)
        xs.append(nodes[i + 1])
        ys.append(errors[i])
        ys.append(errors[i])
        nodes_y.append(errors[i])
    nodes_y.append(nodes_y[-1])
    plt.plot(xs, ys, 'b', nodes, nodes_y, 'rs')
    h = nodes[-1] - nodes[0]
    plt.xlim([nodes[0] - 0.05 * h, nodes[-1] + 0.05 * h])
    plt.show()

def draw(row):
    # size, solution, dual_solution, nodes, norm, dual, fn, error, f_norms

    xs = []
    ys = []
    yds = []
    nodes = states[row.row()].nodes
    un = states[row.row()].solution
    dual = states[row.row()].dual_solution
    size = states[row.row()].size
    f_norms = states[row.row()].f_norms
    norm = states[row.row()].norm
    dual_norm = states[row.row()].dual
    fn = states[row.row()].fn

    print(norm)
    print(dual_norm)
    print(f_norms)
    print([numpy.sqrt((size - 1) * abs(f_norms[i] - (norm[i] + dual_norm[i])) / fn) for i in range(size - 1)])
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
    plt.plot(xs, ys, 'b', nodes, nodes_y, 'b^', xs, yds, 'g', nodes, nodes_yd, 'g^', nodes, nodes_0, 'rs')
    h = nodes[-1] - nodes[0]
    plt.xlim([nodes[0] - 0.05 * h, nodes[-1] + 0.05 * h])
    plt.show()

# ui pyqt
app = QApplication(sys.argv)

listView = QTableWidget()
listView.setRowCount(len(states))
listView.setColumnCount(5)
# size, solution, dual_solution, nodes, norm, dual, fn, error
listView.setHorizontalHeaderItem(0, QTableWidgetItem("Size"))
listView.setHorizontalHeaderItem(1, QTableWidgetItem("||u_h||"))
listView.setHorizontalHeaderItem(2, QTableWidgetItem("||phi_h||"))
listView.setHorizontalHeaderItem(3, QTableWidgetItem("||f||"))
listView.setHorizontalHeaderItem(4, QTableWidgetItem("error"))
# print("""
# \\begin{table}[H]
# \\centering
# \\begin{tabular}{|l|l|l|l|l|l|l|l|}
# \\hline
# N   & $||u_h||_L$ & $||e_h||_L$ & $\\frac{||e_h||_L}{||u_h||_L}, \\%$ & $||u_h||_H$ & $||e_h||_H$ & $\\frac{||e_h||_H}{||u_h||_H}, \\%$ & $p_H$   \\\\ \\hline""")
listView.setEditTriggers(QAbstractItemView.NoEditTriggers)
for i in range(len(states)):
    listView.setItem(i, 0, QTableWidgetItem("{0}".format(states[i].size)))
    listView.setItem(i, 1, QTableWidgetItem("{0:.5}".format(sum(states[i].norm))))
    listView.setItem(i, 2, QTableWidgetItem("{0:.5}".format(sum(states[i].dual))))
    listView.setItem(i, 3, QTableWidgetItem("{0:.5}".format(states[i].fn)))
    listView.setItem(i, 4, QTableWidgetItem("{0:.5}".format(states[i].error)))
# print("""
# \\end{tabular}
# \\caption{Характеристики апроксимацiї на рівномірній сiтцi.}
# \\end{table}
# """)
listView.doubleClicked.connect(draw_error)
listView.setWindowState(QtCore.Qt.WindowMaximized)
listView.show()
sys.exit(app.exec_())