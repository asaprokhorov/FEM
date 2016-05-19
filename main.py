import matplotlib.pyplot as plt
from lsm import *
import sys
from PyQt5.QtWidgets import *
from PyQt5 import QtCore
import numpy as np

def func(x):
    return numpy.sin(numpy.exp(x*3))
size = 2
a = np.float64(-1.)
b = np.float64(1.)
accuracy = np.float64(0.03)
h = (b - a) / size
nodes = []
for i in range(size + 1):
    x = a + i * h
    nodes.append(x)


states = []

solution = h_adaptive_LSM(func, nodes, create_basis(nodes), accuracy, states)

def draw(row):
    xs = []
    ys = []

    nodes = states[row.row()].nodes
    un = states[row.row()].function
    size = states[row.row()].size
    nodes_0 = [0 for i in range(size)]
    nodes_y = [un(nodes[i]) for i in range(size)]

    yf = []
    size = 300
    h = (b - a) / size
    for i in range(size + 1):
        x = a + h * i
        xs.append(x)
        ys.append(un(x))
        yf.append(func(x))
    plt.plot(xs, ys, 'b', xs, yf, 'g--', nodes, nodes_y, 'b^', nodes, nodes_0, 'r^')
    h = nodes[-1] - nodes[0]
    plt.xlim([nodes[0] - 0.05 * h, nodes[-1] + 0.05 * h])
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()

# ui pyqt
app = QApplication(sys.argv)

listView = QTableWidget()
listView.setRowCount(len(states))
listView.setColumnCount(2)
listView.setHorizontalHeaderItem(0, QTableWidgetItem("Size"))
listView.setHorizontalHeaderItem(1, QTableWidgetItem("Error %"))
for i in range(len(states)):
    listView.setItem(i, 0, QTableWidgetItem("{0}".format(states[i].size)))
    listView.setItem(i, 1, QTableWidgetItem("{0:.5} %".format(states[i].error * 100)))
listView.doubleClicked.connect(draw)
listView.setWindowState(QtCore.Qt.WindowMaximized)
listView.show()
sys.exit(app.exec_())