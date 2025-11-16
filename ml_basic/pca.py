import numpy as np

"""
PCA（主成分分析, Principal Component Analysis）
PCA 是一种“把高维数据压缩成低维”的数学方法，目的是在损失尽量少信息的前提下，让数据更简单、更易理解。

现实中数据常常维度太高，比如：
    一张图片有几千个像素（上千维）
    一个人的健康体检报告有几十个指标（几十维）

维度太高会带来：
    模型计算量大；
    数据可视化困难；
    特征之间可能冗余、相关。

PCA 的目标：
    减少维度（从 100 维→2维）
    保留主要信息（比如方差最大方向）
    去除特征间的相关性（让新坐标轴彼此正交）
    
核心思想：找“最大方差方向”

假设你有一堆二维数据点：
   *
      *       *
   *        *
         *      *
如果你画出这些点的散点图，会发现它们大致沿着某个“斜着的方向”分布。

PCA 就是在问：“哪条直线能最好地穿过这些点，让所有点到这条线的距离平方和最小？”
这条线的方向，就是第一个 主成分（PC1）。
它是数据变化最强、信息量最多的方向。
第二个主成分（PC2）是和第一个垂直的方向，解释剩下的次大方差。
"""

# 1. 构造一个 3D 数据集
np.random.seed(0)
X = np.dot(np.random.rand(3,3), np.random.randn(3,100)).T  # [100, 3]
print("原始数据维度:", X.shape)

# 2. 数据中心化
X_centered = X - X.mean(axis=0)

# 3. 计算协方差矩阵
cov = np.cov(X_centered, rowvar=False)

# 4. 求特征值和特征向量
eig_vals, eig_vecs = np.linalg.eigh(cov)

# 5. 按特征值大小排序
idx = np.argsort(eig_vals)[::-1]
eig_vals = eig_vals[idx]
eig_vecs = eig_vecs[:, idx]

# 6. 计算解释方差比例
explained_var_ratio = eig_vals / eig_vals.sum()
cum_explained_var = np.cumsum(explained_var_ratio)

print("\n每个主成分的解释方差比例:")
for i, ratio in enumerate(explained_var_ratio):
    print(f"PC{i+1}: {ratio:.4f} ({cum_explained_var[i]:.4f} 累计)")

# 7. 取前2个主成分做降维
W = eig_vecs[:, :2]
X_pca = X_centered @ W
print("\n降维后数据维度:", X_pca.shape)

# 输出前5个样本对比
print("\n原始数据(前5个样本，3维):\n", X[:5])
print("\n降维后数据(前5个样本，2维):\n", X_pca[:5])