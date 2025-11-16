# encoding=utf-8

import numpy as np
import matplotlib.pylab as plt
import matplotlib

matplotlib.use("TkAgg")


def tanh(x):
    """
    Tanh（Hyperbolic Tangent）双曲正切函数。图像是一条 “S” 形曲线，但是以 0 为中心对称的。
    可以理解成一个改进版的 Sigmoid 函数，但对称且更灵敏。Sigmoid 把所有输入压缩到 (0, 1)，Tanh 把输入压缩到 (-1, 1)
    相比 Sigmoid（只能给出 0~1 的“强度”），Tanh 能表达“正信号”和“负信号”，相当于“支持”和“反对”两种态度。比 Sigmoid 更适合神经网络训练，因为输出居中不会让梯度偏移。
    """
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


print(round(tanh(3.1415926), 2))

X = np.arange(-5.0, 5.0, 0.1)
Y = tanh(X)
plt.plot(X, Y)
plt.ylim(-1.1, 1.1)
plt.show()
