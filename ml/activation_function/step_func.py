# encoding=utf-8

import numpy as np
import matplotlib.pylab as plt
import matplotlib

matplotlib.use("TkAgg")

"""
阶跃函数（Step Function）是一种“开关式”的函数：
当输入大于 0 时输出 1，否则输出 0。
它最早用于模拟神经元“是否被激活”的行为。

阶跃函数是最早的激活函数，像“开关”一样判断信号是否足够强。
它奠定了神经网络“激活机制”的基础，
但因不能求导，现在更多用于教学演示或理论说明。
"""


def step_function(x):
    return np.array(x > 0, dtype=int)


print(step_function(0))

X = np.arange(-5.0, 5.0, 0.1)
Y = step_function(X)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1)  # 指定图中绘制的y轴的范围
plt.show()
