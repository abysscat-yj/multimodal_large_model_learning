# encoding=utf-8

import numpy as np
import matplotlib.pylab as plt
import matplotlib

matplotlib.use("TkAgg")


def relu(x):
    """
    ReLU（Rectified Linear Unit，线性整流单元） 是目前神经网络中最常用、最经典的激活函数之一。
    可以把 ReLU 想象成一个 单向阀门 或 二极管：只允许正电流（正信号）通过，负电流（负信号）直接截断。这让模型能“保留激活的神经元”，同时忽略无意义的负输入。
    对比：Sigmoid 像“概率阀门”，平滑但容易迟钝。ReLU 像“开关”，干脆利落，高效易训。
    ReLU 让神经元学会对不同输入自动“开/关”响应，从而让网络在不同样本下选择不同的特征组合表达。
    """
    return np.maximum(0, x)


print(round(relu(3.1415926), 2))

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.ylim(-1.0, 5.5)
plt.show()
