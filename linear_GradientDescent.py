import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# 线性回归实现 梯度下降方法
# y = ax + b
# a, b 是标量
# x, y 是向量


# 模型 y = ax + b, 需要训练出 a 和 b 的值
def model(a, b, x):
    # numpy的广播特征
    # print(a, b, x)
    ret = a*x+b
    # print("ret :  ", ret)
    # ret = [a*i+b for i in x]
    # print(ret)
    return ret


# 代价|损失函数
def cost_function(a, b, x, y, n=5):
    # 1/(2n) * sum( square(yi -yi`) )
    return 0.5 / n * (np.square(y - a * x - b)).sum()


# 梯度下降方法 对a,b进行优化
def optimize(a, b, x, y, n=5):
    # a = a-alpha*(损失函数对a求偏导)
    alpha = 1e-1
    y_hat = model(a, b, x)
    da_t = y_hat - y
    da = (1.0 / n) * ((y_hat - y) * x).sum()
    db = (1.0 / n) * (y_hat - y).sum()
    a = a - alpha * da
    b = b - alpha * db
    return a, b


# 使用x, y 向量训练出 a, b两个标量
def train(x, y, cost=10000):
    # 初始化 a=b=0
    a = b = 0.0
    # 进入循环 直到到达循环结束条件
    # 循环结束条件：损失函数最小值   #这儿我是用控制循环次数来结束循环
    while True:
        # 优化方法 使用梯度下降的算法，对a,b进行优化
        a, b=optimize(a, b, x, y)
        cost = cost - 1
        if cost == 0:
            break
    return a, b


if __name__ == "__main__":
    # 数据提取 使用pandas
    # ds 的类型：<class 'pandas.core.frame.DataFrame'>
    ds = pd.read_csv("LinearRegression.csv")

    print(type(np.array(ds.iloc[:, [1]])),np.array(ds.iloc[:, [1]]))
    # 将数据变成一维
    # np.array(ds.iloc[:, [1]])
    # 将 <class 'pandas.core.frame.DataFrame'> 转换为 <class 'numpy.ndarray'>
    # narray.reshape(-1) 将 numpy.narray 的多维转换为一维
    x = np.array(ds.iloc[:, [1]]).reshape(-1)
    print(type(x), x)
    y = np.array(ds.iloc[:, [2]]).reshape(-1)
    # x = [float(i) for i in x]
    # y = [float(i) for i in y]
    # print(type(np.array(ds.iloc[:, [1]]).reshape(-1)),np.array(ds.iloc[:, [1]]).reshape(-1) )
    # print(type(x), x)
    # print(type(y), y)
    # 使用 6 张图片显示
    count = 1
    for i in range(1, 7):
        plt.subplot(3, 2, i)
        a, b = train(x, y, count)
        plt.plot(x, y, "bo")
        plt.plot(x, model(a, b, x))
        count = count * 3
    plt.show()


