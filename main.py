# import matplotlib.pyplot as plt
# from lsm import *
#
#
# def func(x):
#     return numpy.exp(numpy.sin(x*3))
#
#
# size = 5
# a = -1
# b = 1
# h = (b - a) / size
# nodes = []
# for i in range(size + 1):
#     x = a + i * h
#     nodes.append(x)
#
# accuracy = 0.05
# states = []
#
# solution = h_adaptive_LSM(func, nodes, create_basis(nodes), accuracy, states)
#
# xs = []
# ys = []
# size = 100
# h = (b - a) / size
# for i in range(size + 1):
#     x = a + h * i
#     xs.append(x)
#     ys.append(solution(x))
#
#
# plt.plot(xs, ys)
# plt.show()


from tkinter import Tk, Frame
import tktable

root = Tk()
table = tktable.Table(root, rows=5, cols=5)
table.pack()
root.mainloop()
