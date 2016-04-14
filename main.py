import matplotlib.pyplot as plt
from lsm import *
import sys
from PyQt5.QtWidgets import *
import numpy as np

def func(x):
    return numpy.sin(numpy.exp(x*3))
size = 20
a = np.float64(-1.)
b = np.float64(1.)
accuracy = np.float64(0.07)
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

    nodes_y = [func(states[row.row()].nodes[i]) for i in range(states[row.row()].size)]

    yf = []
    size = 100
    h = (b - a) / size
    for i in range(size + 1):
        x = a + h * i
        xs.append(x)
        ys.append(states[row.row()].function(x))
        yf.append(func(x))
    plt.plot(xs, ys)
    plt.plot(xs, yf)
    plt.plot(states[row.row()].nodes, nodes_y, 'go')

    plt.show()

# ui pyqt
app = QApplication(sys.argv)

listView = QTableWidget()
listView.setRowCount(len(states))
listView.setColumnCount(3)
listView.setHorizontalHeaderItem(0, QTableWidgetItem("Size"))
listView.setHorizontalHeaderItem(1, QTableWidgetItem("Error %"))
for i in range(len(states)):
    listView.setItem(i, 0, QTableWidgetItem("{0}".format(states[i].size)))
    listView.setItem(i, 1, QTableWidgetItem("{0:.5} %".format(states[i].error * 100)))
listView.doubleClicked.connect(draw)

listView.show()
sys.exit(app.exec_())