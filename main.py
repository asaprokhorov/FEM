import matplotlib.pyplot as plt
from lsm import *



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

xs = []
ys = []
size = 100
h = (b - a) / size
for i in range(size + 1):
    x = a + h * i
    xs.append(x)
    ys.append(solution(x))
plt.plot(xs, ys)
plt.show()

#
# import sys
# from PyQt5.QtWidgets import *
# from PyQt5.QtCore import *
#
# if __name__ == '__main__':
#
#     app = QApplication(sys.argv)
#
#     listView = QTableWidget()
#     listView.setRowCount(len(states))
#     listView.setColumnCount(2)
#     for i in range(len(states)):
#         listView.setItem(i, 0, QTableWidgetItem(states[i].size.__str__()))
#         listView.setItem(i, 1, QTableWidgetItem(states[i].error.__str__()))
#     listView.doubleClicked.connect(draw)
#     listView.show()
#     sys.exit(app.exec_())