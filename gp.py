import matplotlib.pyplot as plt
import numpy as np


# https://www.zhihu.com/question/46631426?sort=created
# 高斯核函数
def gaussian_kernel(x1, x2, l=0.5, sigma_f=0.2):
    m, n = x1.shape[0], x2.shape[0]
    dist_matrix = np.zeros((m, n), dtype=float)
    for i in range(m):
        for j in range(n):
            dist_matrix[i][j] = np.sum((x1[i]) - x2[j]) ** 2
    return sigma_f ** 2 * np.exp(-0.5 / l ** 2 * dist_matrix)


# 假设我们的未知函数的groudtruth是 sin函数，我们通过 gp来逐步拟合出来它的形状
# 生成观测值,取sin函数没有别的用意,单纯就是为了计算出Y def gety(X)
def getY(X):
    X = np.asarray(X)
    Y = np.sin(X)
    return Y.tolist()


# 根据观察点X,修正生成高斯过程新的均值和协方差
def update(X, Y, X_star):  # X是观测点
    X = np.asarray(X)
    X_star = np.asarray(X_star)
    K_YY = gaussian_kernel(X, X)  # K(X, X)
    K_ff = gaussian_kernel(X_star, X_star)  # K(X*, X*)
    K_Yf = gaussian_kernel(X, X_star)  # K(X, X*)
    K_fY = K_Yf.T  # K(X*』X)协方差矩阵是对称的,因此分块互为转置
    K_YY_inv = np.linalg.inv(K_YY + 1e-8 * np.eye(len(X)))  # (N, N)
    mu_star = K_fY.dot(K_YY_inv).dot(Y)  # (100,1)
    cov_star = K_ff - K_fY.dot(K_YY_inv).dot(K_Yf)  # (100,100)
    return mu_star, cov_star


# 绘制高斯过程的先验
f, ax = plt.subplots(2, 1, sharex=True, sharey=True)
X_pre = np.arange(0, 10, 0.1)
mu_pre = np.array([0] * len(X_pre))
Y_pre = mu_pre
cov_pre = gaussian_kernel(X_pre, X_pre)
uncertainty = 1.96 * np.sqrt(
    np.diag(cov_pre))  # 取95%置信区间, 对于正态分布总体的一个随机样本,有如下关系:x+1S包含所有数据的68.26%;x+1.96S包含所有数据的95%;x±2.58S包含所 有数据的99%。
ax[0].fill_between(X_pre, Y_pre + uncertainty, Y_pre - uncertainty, alpha=0.1)
ax[0].plot(X_pre, Y_pre, Label="expection")
ax[0].legend()

# 绘制基于观测值的高斯过程后验
X_sample = np.array([0, 1, 2, 3, 4, 5, 6, 6.18]).reshape(-1, 1)  # 4*1矩阵
Y_sample = getY(X_sample)
X_star = np.arange(0, 10, 0.1).reshape(-1, 1)
mu_star, cov_star = update(X_sample, Y_sample, X_star)
Y_star = mu_star.ravel()
uncertainty = 1.96 * np.sqrt(np.diag(cov_star))  # 取95%置信区间
ax[1].fill_between(X_star.ravel(), Y_star + uncertainty, Y_star - uncertainty, alpha=0.1)
ax[1].plot(X_star, Y_star, Label="expection")
ax[1].scatter(X_sample, Y_sample, Label="observationpoint", c="red", marker="x")
ax[1].legend()
plt.show()
