# source codes of MPEG WG4 VCM proposal m63692
# Haiqiang Wang, et al., "[VCM] Improvements of the BD-rate model using monotonic
# curve-fitting method," ISO/IEC JTC 1/SC 29/WG 4, Doc. m63692，Geneva, CH – July 2023.
# contact: walleewang@tencent.com

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def func_cubic_2(x, b, extra=0.0):
    # 三项式函数
    return b[0] * np.power(x, 3) + b[1] * np.power(x, 2) + b[2] * np.asarray(x) + b[3] - extra


def fit_cubic(x, y, m_min=0, m_max=100):
    def func_cubic(b, x):
        # 三项式无偏置
        s = b[0] * np.power(x, 3) + b[1] * np.power(x, 2) + b[2] * np.asarray(x) + b[3]
        return s

    def objective(b):
        # 误差平方和
        return np.sum(np.power(func_cubic(b, x) - y, 2))

    def const_1st_derivative(b):
        # 一阶导数
        return 3 * b[0] * np.power(x, 2) + 2 * b[1] * np.asarray(x) + b[2]

    cons = (dict(type='ineq', fun=const_1st_derivative))
    init = np.array([1., 1., 1., 1.])
    res = minimize(objective, x0=init, method='SLSQP', constraints=cons)
    if not res.success:
        print(res)
        raise ValueError('optimization failed')
    return res.x


def cal_error(x_a, y_a, coef_a):
    total_err = 0
    for idx, val in enumerate(x_a):
        error = func_cubic_2(val, coef_a, y_a[idx])
        total_err += error ** 2
    return total_err


def main_loop(x_a, y_a):
    m_min = -1
    m_max = -1

    x_a = np.log10(x_a)

    coef_a = fit_cubic(x_a, y_a, m_min, m_max)
    # coef_a = result
    print("-----------for function a--------------------")
    print(f"coefficient a: {coef_a}")
    print(f"total error for six points is : {cal_error(x_a, y_a, coef_a)}")

    p = np.poly1d(coef_a)

    # # 创建x轴上的一系列点，用于绘图
    x_plot = np.linspace(min(x_a), max(x_a), 400)

    # # 使用拟合的多项式计算这些点的y值
    y_plot = p(x_plot)

    # # 绘图
    plt.figure(figsize=(10, 6))
    # plt.scatter(x_a, y_a, color='red', label='Data points')
    plt.scatter(x_a, y_a, color='blue', label='Data points')

    plt.plot(x_plot, y_plot, label='Cubic fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()


# x_a = [19614.84, 12378.01, 5522.57, 3214.22, 1799.61, 894.12]
# y_a = [44.85, 44.87, 43.75, 32.05, 27.37, 25.14]
x_a = [5631.53, 3525.00, 1510.55, 868.94, 485.06, 242.01]
y_a = [44.62, 44.24, 43.18, 41.39, 36.60, 28.84]
error = main_loop(x_a, y_a)
print(error)
