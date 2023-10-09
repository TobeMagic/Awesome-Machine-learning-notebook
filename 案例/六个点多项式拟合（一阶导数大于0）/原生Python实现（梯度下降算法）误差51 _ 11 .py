#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/30 16:03
# @Author  : AI_magician
# @File    : 原生Python实现（梯度下降算法）误差51 _ 11 .py
# @Project : PyCharm
# @Version : 1.0,
# @Contact : 1928787583@qq.com",
# @License : (C)Copyright 2003-2023, AI_magician",
# @Function: Use original python to fix a three project curve


import matplotlib.pyplot as plt
from typing import List
import numpy as np


def find_square_root(number, epsilon):
    guess = number / 2  # 初始猜测为number的一半

    while abs(guess * guess - number) > epsilon:
        guess = (guess + number / guess) / 2

    return guess


number = 16
epsilon = 1e-6


class Trinomials:
    parameter = [1., 1., 1., 1.]  # init parameter (which can use np.random to generate)
    learning_rate = 0.00005

    # m_min = 0
    # m_max = 100
    # max_iterations = 50

    def __init__(self):
        pass

    @staticmethod
    def func_polynomial(x, b: List[float]):
        # 三项式函数
        return b[0] * x ** 3 + b[1] * x ** 2 + b[2] * x + b[3]

    @staticmethod
    def const_1st_derivative(x, b: List[float]):
        """
        :param x: Single
        :param b: List
        :return:  first derivative
        """
        # 一阶导数
        return 3 * b[0] * x ** 2 + 2 * b[1] * x + b[2]

    def gradient_descent_with_constraints(self, x, y, b):
        """
        :param x: Single Float
        :param y: Single Float
        :param b: List Float
        :return: Parameter
        :description: gradient descent algorithm
        """
        # for i in range(max_iterations):
        # 计算关于每个参数的偏导数（梯度）
        gradients = [
            # 2 * (self.func_polynomial(x, b) - y) * 3 * b[0] * x ** 2,
            # 2 * (self.func_polynomial(x, b) - y) * 2 * b[1] * x,
            # 2 * (self.func_polynomial(x, b) - y) * b[2] * x,
            # 2 * (self.func_polynomial(x, b) - y)]
            2 * (self.func_polynomial(x, b) - y) * x ** 3,
            2 * (self.func_polynomial(x, b) - y) * x ** 2,
            2 * (self.func_polynomial(x, b) - y) * x,
            2 * (self.func_polynomial(x, b) - y)]
        # print("梯度：", gradients)

        # def constraints():
        b_update = [0] * len(b)
        for i, b_i in enumerate(b):
            b_update[i] = b_i - gradients[i] * self.learning_rate
        # self.learning_rate *= 0.3
        if self.const_1st_derivative(x, b_update) > 0:
            return b_update
        else:
            return b

    def fit(self, x, y, m_min=0, m_max=100, learning_rate=0.1, max_iterations=5000):
        # temp_x = [0] * len(x)
        # temp_y = [0] * len(y)
        # x = list(x)
        # y = list(y)
        # for i in range(len(x)):
        #     x.append(x[i] + 0.1)
        #     y.append(y[i] + 0.1)
        # temp_x = x + [0.001 for i in range(len(x))]
        # temp_y = y + [0.001 for i in range(len(y))]
        # print(temp_x, temp_y)
        # print(x , y)
        # x += temp_x
        # y += temp_y
        # x_mean = sum(x) / len(x)
        # print(x_mean)
        # x_standard_deviation = find_square_root(sum((x_mean - x_i) ** 2 for x_i in x) / len(x), epsilon)
        # x = [(x_i - x_mean) / x_standard_deviation for x_i in x]
        # print(x_standard_deviation)
        # print(x)
        for index in range(10000):
            for i in range(len(x)):
                self.parameter = self.gradient_descent_with_constraints(x[i], y[i], self.parameter)
                print("参数", self.parameter)
            # self.learning_rate *= 0.03
            # print(self.learning_rate)
        return self.parameter

    def error_square(x_a, y_a, coef_a):
        total_err = sum((Trinomials.func_polynomial(xi, coef_a) - yi) ** 2 for xi, yi in zip(x_a, y_a))

        return total_err


if __name__ == "__main__":
    # x_a = [19614.84, 12378.01, 5522.57, 3214.22, 1799.61, 894.12]
    # y_a = [44.85, 44.87, 43.75, 32.05, 27.37, 25.14]
    x_a = [5631.53, 3525.00, 1510.55, 868.94, 485.06, 242.01]
    y_a = [44.62, 44.24, 43.18, 41.39, 36.60, 28.84]
    x_a = np.log10(x_a)
    trinomials = Trinomials()
    coef_a = Trinomials.fit(trinomials, x=x_a, y=y_a)

    print("-----------for trinomials function --------------------")

    print(f"coefficients: {coef_a}")

    print(f"total error for six points is : {Trinomials.error_square(x_a, y_a, coef_a)}")

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
