from new_fem import h_adaptive_fem
# from dual_fem import h_adaptive_fem
import numpy
from matplotlib import pyplot as plt
from PyQt5.QtWidgets import *
from PyQt5 import QtCore
import sys

a = 0
b = 1

nodes = numpy.linspace(a, b, 3, endpoint=True)

m = lambda x: 1
sigma = lambda x: 1
f = lambda x: x ** 2 - 2
_u = -1
alpha = -1

# m = lambda x: 1
# sigma = lambda x: 1
# f = lambda x: (1 + numpy.pi ** 2) * numpy.sin(numpy.pi * x)
# _u = -numpy.pi
# alpha = 1

accuracy = 0.01


states = []

states = h_adaptive_fem(m, sigma, f, alpha, _u, nodes, accuracy, states)

def draw(row):
    xs = []
    ys = []

    nodes = states[row.row()].nodes
    un = states[row.row()].function
    size = states[row.row()].size
    nodes_0 = [0 for i in range(size)]
    nodes_y = [un(nodes[i]) for i in range(size)]
    size = 1000
    h = (b - a) / size
    for i in range(size + 1):
        x = a + h * i
        xs.append(x)
        ys.append(un(x))
    plt.plot(xs, ys, 'b', nodes, nodes_y, 'b^', nodes, nodes_0, 'rs')
    h = nodes[-1] - nodes[0]
    plt.xlim([nodes[0] - 0.05 * h, nodes[-1] + 0.05 * h])
    h = nodes[-1] - nodes[0]
    plt.show()

# ui pyqt
app = QApplication(sys.argv)

listView = QTableWidget()
listView.setRowCount(len(states))
listView.setColumnCount(9)
listView.setHorizontalHeaderItem(0, QTableWidgetItem("Size"))
listView.setHorizontalHeaderItem(1, QTableWidgetItem("Uh_L"))
listView.setHorizontalHeaderItem(2, QTableWidgetItem("e_L"))
listView.setHorizontalHeaderItem(3, QTableWidgetItem("||e||L / ||uh||L %"))
listView.setHorizontalHeaderItem(4, QTableWidgetItem("pL"))
listView.setHorizontalHeaderItem(5, QTableWidgetItem("Uh_H"))
listView.setHorizontalHeaderItem(6, QTableWidgetItem("e_H"))
listView.setHorizontalHeaderItem(7, QTableWidgetItem("||e||H / ||uh||H %"))
listView.setHorizontalHeaderItem(8, QTableWidgetItem("pH"))
# print("""
# \\begin{table}[H]
# \\centering
# \\begin{tabular}{|l|l|l|l|l|l|l|l|}
# \\hline
# N   & $||u_h||_L$ & $||e_h||_L$ & $\\frac{||e_h||_L}{||u_h||_L}, \\%$ & $||u_h||_H$ & $||e_h||_H$ & $\\frac{||e_h||_H}{||u_h||_H}, \\%$ & $p_H$   \\\\ \\hline""")
listView.setEditTriggers(QAbstractItemView.NoEditTriggers)
for i in range(len(states)):
    listView.setItem(i, 0, QTableWidgetItem("{0}".format(states[i].size)))
    listView.setItem(i, 1, QTableWidgetItem("{0:.5}".format(states[i].norm_u)))
    listView.setItem(i, 2, QTableWidgetItem("{0:.5}".format(states[i].e_l)))
    listView.setItem(i, 3, QTableWidgetItem("{0:.5}".format(states[i].e_l / states[i].norm_u * 100)))
    listView.setItem(i, 5, QTableWidgetItem("{0:.5}".format(states[i].derivative_norm_u)))
    listView.setItem(i, 6, QTableWidgetItem("{0:.5}".format(states[i].e_h)))
    listView.setItem(i, 7, QTableWidgetItem("{0:.5}".format(states[i].e_h / states[i].derivative_norm_u * 100)))
    p_h = ""
    p_l = ""
    if i > 0:
        # p_h = (numpy.log(states[0].e_h) - numpy.log(states[i].e_h)) / (numpy.log(states[i].size) - numpy.log(states[0].size))
        # p_l = (numpy.log(states[0].e_l) - numpy.log(states[i].e_l)) / (numpy.log(states[i].size) - numpy.log(states[0].size))
        # p_h = numpy.log(states[i].e_h - states[i - 1].e_h) / numpy.log(states[i].size - states[i - 1].size)
        # p_l = numpy.log(states[i].e_l - states[i - 1].e_l) / numpy.log(states[i].size - states[i - 1].size)
        p_h = numpy.log(states[i - 1].e_h / states[i].e_h) / numpy.log(states[i].size / states[i - 1].size)
        p_l = numpy.log(states[i - 1].e_l / states[i].e_l) / numpy.log(states[i].size / states[i - 1].size)
    # print("{0}  & {1:.5} & {2:.5} & {3:.5} & {4:.5} & {5:.5} & {6:.5} & \\\\ \\hline".format(states[i].size, states[i].norm_u, states[i].e_l, states[i].e_l / states[i].norm_u * 100, states[i].derivative_norm_u, states[i].e_h, states[i].e_h / states[i].derivative_norm_u * 100, p_h))
    listView.setItem(i, 8, QTableWidgetItem("{0:.5}".format(p_h)))
    listView.setItem(i, 4, QTableWidgetItem("{0:.5}".format(p_l)))
# print("""
# \\end{tabular}
# \\caption{Характеристики апроксимацiї на рівномірній сiтцi.}
# \\end{table}
# """)
listView.doubleClicked.connect(draw)
listView.setWindowState(QtCore.Qt.WindowMaximized)
listView.show()
sys.exit(app.exec_())