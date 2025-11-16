# encoding=utf-8

import numpy as np
import matplotlib.pylab as plt
import matplotlib

matplotlib.use("TkAgg")


def sigmoid(x):
    """
    Sigmoid 函数（也叫逻辑函数 / logistic function）是一个常用在机器学习和神经网络中的数学函数，会把任意实数 x 压缩（映射）到 0 到 1 之间
    """
    return 1 / (1 + np.exp(-x))


# Sigmoid 函数的核心特性：它对「中间区域」非常敏感，但一旦超过某个阈值，输出就迅速“饱和”，也就是靠近 1 或 0
# Sigmoid 的数学本质是一种「概率压缩」，它的增长速度是由e-x控制的，而指数函数变化极快。
# 缺陷：会导致在极值区间（例如x>3或x<-3）函数变化太慢、导数基本为0 => 梯度消失，是神经网络训练变慢或失效，所以后来大家更喜欢用 ReLU 等激活函数
print(round(sigmoid(3.1415926), 2))

X = np.arange(-5.0, 5.0, 0.1)
print(X)
Y = sigmoid(X)
print(Y)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1)
plt.show()
