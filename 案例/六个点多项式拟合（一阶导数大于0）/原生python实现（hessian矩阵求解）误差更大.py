#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/31 21:41
# @Author  : AI_magician
# @File    : 原生python实现.py
# @Project : PyCharm
# @Version : 1.0,
# @Contact : 1928787583@qq.com",
# @License : (C)Copyright 2003-2023, AI_magician",
# @Function:


learning_rate = 0.00000001
num_iterations = 1000


# 定义目标函数
def objective(params, x, y):
    a, b, c, d = params
    # y_pred = a * x ** 3 + b * x ** 2 + c * x + d
    # residuals = y - y_pred
    return sum((y[i] - (a * x[i] ** 3 + b * x[i] ** 2 + c * x[i] + d)) ** 2 for i in range(len(x)))


# 定义目标函数的梯度
def gradient(params, x, y):
    a, b, c, d = params
    grad_a = -2 * (y - (a * x ** 3 + b * x ** 2 + c * x + d)) * x ** 3
    grad_b = -2 * (y - (a * x ** 3 + b * x ** 2 + c * x + d)) * x ** 2
    grad_c = -2 * (y - (a * x ** 3 + b * x ** 2 + c * x + d)) * x
    grad_d = -2 * (y - (a * x ** 3 + b * x ** 2 + c * x + d))
    return [grad_a, grad_b, grad_c, grad_d]


# 定义约束条件
def constraint(x, params):
    a, b, c, d = params
    derivative = 3 * a * x ** 2 + 2 * b * x + c  # x 的一阶导数恒大于0
    # print(derivative)
    return derivative


def adjust_gradient(gradient):
    adjusted_gradient = []
    # print(gradient)
    for grad in gradient:
        adjusted_gradient.append(max(grad, 0))
    return adjusted_gradient


# Define the Hessian of the objective function
def hessian(params, x, y):
    a, b, c, d = params
    # hess_a_a = 6 * sum(x_i ** 6 for x_i in x)
    # hess_a_b = 2 * sum(x_i ** 5 for x_i in x)
    # hess_a_c = 2 * sum(x_i ** 4 for x_i in x)
    # hess_a_d = 2 * sum(x_i ** 3 for x_i in x)
    # hess_b_a = 2 * sum(x_i ** 5 for x_i in x)
    # hess_b_b = 2 * sum(x_i ** 4 for x_i in x)
    # hess_b_d = 2 * sum(x_i ** 3 for x_i in x)
    # hess_b_c = 2 * sum(x_i ** 2 for x_i in x)
    # hess_c_a = 2 * sum(x_i ** 4 for x_i in x)
    # hess_c_b = 2 * sum(x_i ** 3 for x_i in x)
    # hess_c_c = 2 * sum(x_i ** 2 for x_i in x)
    # hess_c_d = 2 * sum(x_i ** 1 for x_i in x)
    # hess_d_a = 2 * sum(x_i ** 3 for x_i in x)
    # hess_d_b = 2 * sum(x_i ** 2 for x_i in x)
    # hess_d_c = 2 * sum(x_i ** 1 for x_i in x)
    # hess_d_d = 2 * len(x)
    hess_a_a = 6 * x ** 6
    hess_a_b = 2 * x ** 5
    hess_a_c = 2 * x ** 4
    hess_a_d = 2 * x ** 3
    hess_b_a = 2 * x ** 5
    hess_b_b = 2 * x ** 4
    hess_b_c = 2 * x ** 3
    hess_b_d = 2 * x ** 2
    hess_c_a = 2 * x ** 4
    hess_c_b = 2 * x ** 3
    hess_c_c = 2 * x ** 2
    hess_c_d = 2 * x ** 1
    hess_d_a = 2 * x ** 3
    hess_d_b = 2 * x ** 2
    hess_d_c = 2 * x ** 1
    hess_d_d = 2 * 1
    return [[hess_a_a, hess_a_b, hess_a_c, hess_a_d],
            [hess_b_a, hess_b_b, hess_b_c, hess_b_d],
            [hess_c_a, hess_c_b, hess_c_c, hess_c_d],
            [hess_d_a, hess_d_b, hess_d_c, hess_d_d]]


# 定义 SLSQP 算法
def slsqp_algorithm(objective, gradient, constraint, x, y, initial_params, max_iter=100000):
    params = initial_params
    for i in range(max_iter):
        for i in range(len(x)):
            # i = 0
            # 计算 Hessian 矩阵的近似值
            grad = gradient(params, x[i], y[i])
            hess = hessian(params, x[i], y[i])
            # 检查约束条件
            params_update = [params[j] - learning_rate * sum([hess[j][k] * grad[k] for k in range(len(params))]) / 4 for j in
                      range(len(params))]
            if constraint(x[i], params_update) < 0:
                # 如果约束条件不满足，则将梯度向量的方向调整为保证一阶导数恒大于0
                print("constraint!")
                grad = adjust_gradient(grad)
                print(grad)

            # 更新参数0
            params = [params[j] - learning_rate * sum([hess[j][k] * grad[k] for k in range(len(params))]) for j in
                      range(len(params))]

        # 检查约束条件

    return params


# 初始化参数
initial_params = [1., 1., 1., 1.]  # 初始参数值

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # x_a = [19614.84, 12378.01, 5522.57, 3214.22, 1799.61, 894.12]
    # y_a = [44.85, 44.87, 43.75, 32.05, 27.37, 25.14]
    x_a = [5631.53, 3525.00, 1510.55, 868.94, 485.06, 242.01]
    y_a = [44.62, 44.24, 43.18, 41.39, 36.60, 28.84]
    x_a = np.log10(x_a)
    x_a = list(x_a)
    # 使用 SLSQP 算法求解非线性优化问题

    coef_a = slsqp_algorithm(objective, gradient, constraint, x_a, y_a, initial_params)

    # 打印结果
    print("优化结果:")
    print("a =", coef_a[0])
    print("b =", coef_a[1])
    print("c =", coef_a[2])
    print("D =", coef_a[3])

    print("-----------for trinomials function --------------------")

    print(f"coefficients: {coef_a}")

    print(f"total error for six points is : {objective(x=x_a, y=y_a, params=coef_a)}")

    print(f"constraint: {constraint(x=x_a[0], params=coef_a)}")

    p = np.poly1d(coef_a)

    # # 创建x轴上的一系列点，用于绘图
    x_plot = np.linspace(min(x_a), max(x_a), 400)

    # # 使用拟合的多项式计算这些点的y值
    y_plot = p(x_plot)

    # # 绘图
    plt.figure(figsize=(10, 6))
    # plt.scatter(x_a, y_a, color='red', label='Data points')
    plt.scatter(x_a, y_a, color='blue', label='Data points')

    plt.plot(x_plot, y_plot, label='polynomial fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
