import matplotlib.pyplot as plt
from lsm import *
import sys
from PyQt5.QtWidgets import *



def func(x):
    return numpy.exp(numpy.sin(x*3))


size = 5
a = -1
b = 1
h = (b - a) / size
nodes = []
for i in range(size + 1):
    x = a + i * h
    nodes.append(x)

accuracy = 0.005
states = []

solution = h_adaptive_LSM(func, nodes, create_basis(nodes), accuracy, states)


def draw(row):
    xs = []
    ys = []
    size = 100
    h = (b - a) / size
    for i in range(size + 1):
        x = a + h * i
        xs.append(x)
        ys.append(states[row.row()].function(x))
    plt.plot(xs, ys)
    plt.show()


# ui pyqt
app = QApplication(sys.argv)

listView = QTableWidget()
listView.setRowCount(len(states))
listView.setColumnCount(2)
listView.setHorizontalHeaderItem(0, QTableWidgetItem("Size"))
listView.setHorizontalHeaderItem(1, QTableWidgetItem("Error"))
for i in range(len(states)):
    listView.setItem(i, 0, QTableWidgetItem("{0}".format(states[i].size)))
    listView.setItem(i, 1, QTableWidgetItem("{0:.5}".format(states[i].error)))
listView.doubleClicked.connect(draw)
listView.show()
sys.exit(app.exec_())