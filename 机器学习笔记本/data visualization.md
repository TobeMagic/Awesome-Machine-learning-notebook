## t-SNE 可视化

t-SNE（t-Distributed Stochastic Neighbor Embedding）是一种降维和可视化技术，用于将高维数据映射到二维或三维空间中。它是一种非线性的降维方法，旨在保留原始数据之间的局部相似性关系。由Laurens van der Maaten和Geoffrey Hinton于2008年提出。在此之前，常用的降维方法如PCA（Principal Component Analysis）等主要关注全局结构，而缺乏对局部结构的捕捉能力。

t-SNE通过计算样本之间的相似度，并尝试在低维嵌入空间中保持这些相似度关系。它使用随机梯度下降等优化算法来最小化高维空间和低维嵌入空间之间的Kullback-Leibler散度。结果是，具有类似特征的样本会在低维投影中更接近。

因为t-SNE能够捕捉到复杂、非线性结构以及聚类效应，所以它通常被用于可视化高维数据集中不同类别或群组之间的分布关系。例如，在机器学习领域，可以使用t-SNE将特征向量表示为二维或三维点云图，并观察不同类别样本之间的分离程度。

## 决策树可视化

scikit-learn（sklearn）的`tree`模块提供了一个方便的函数`plot_tree`，用于可视化决策树模型。你可以使用以下步骤来使用`plot_tree`函数进行可视化（以iris数据集为例）：

1. 导入必要的库和模块：在Python脚本中，导入`tree`模块和`matplotlib.pyplot`库：

2. 可视化决策树：使用`plot_tree`函数可视化决策树模型。

3. 调用函数进行可视化：在你的代码中，调用`visualize_decision_tree`函数并传入决策树模型、特征名称和类别名称作为参数：

```python
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def visualize_decision_tree(decision_tree, feature_names, class_names):
    plt.figure(dpi=300)
    tree.plot_tree(decision_tree, feature_names=feature_names, class_names=class_names,
                   filled=True, rounded=True)
    plt.show()
clf = tree.DecisionTreeClassifier(random_state=0)
iris = load_iris()
clf = clf.fit(iris.data, iris.target)
visualize_decision_tree(clf,feature_names = iris.feature_names, class_names = iris.target_names)
```

结果如下图：

每一个节点都有分类阈值以及其gini指数和样本状态和类别状态。

![image-20231007101015991](data visualization.assets/image-20231007101015991.png)

   在上面的代码中，`decision_tree`是你的决策树模型，`feature_names`是特征的名称列表，`class_names`是类别的名称列表。`visualize_decision_tree`函数使用`plot_tree`函数将决策树模型绘制为图形。   运行代码后，你将看到绘制出的决策树图形。

请注意，`plot_tree`函数提供了一些可选参数，可以用于自定义图形的外观。你可以查阅scikit-learn的文档以了解更多关于`plot_tree`函数的详细信息和可选参数的使用方式。

## 回归可视化方案

在评估回归模型效果时，可以使用多种可视化方案来直观地比较实际值和预测值之间的差异。以下是几种常见的回归模型评估可视化方案和相应的Python代码模板：

1.  对角线图：对角线图用于比较实际值和预测值之间的差异。通过将实际值和预测值绘制在同一个图表上，并**绘制一条对角线**（理想情况下实际值等于预测值），可以**直观地观察到预测值偏离对角线的程度**。代码模板如下：

```python
import matplotlib.pyplot as plt

# 绘制对角线图
plt.scatter(y_actual, y_predicted)
plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Diagonal Plot - Actual vs. Predicted')
plt.show()
```

![image-20231021103801800](data visualization.assets/image-20231021103801800.png)

2.  残差图：残差图用于检查回归模型的拟合情况。它将实际值和预测值之间的差异（即残差）绘制在y轴上，将实际值绘制在x轴上。通过观察残差图的分布，可以检查模型是	存在系统性的误差或模型是否满足对误差的假设。代码模板如下：

```python
import matplotlib.pyplot as plt

# 计算残差
residuals = y_actual - y_predicted

# 绘制残差图
plt.scatter(y_actual, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
```

![image-20231021103942039](data visualization.assets/image-20231021103942039.png)

3.  拟合曲线图：可以绘制拟合曲线来可视化模型的拟合效果（**只适合单变量**）。代码模板如下：

```python
import numpy as np
import matplotlib.pyplot as plt

# 生成一系列x值
x = np.linspace(min(x_actual), max(x_actual), 100)

# 预测对应的y值
y_predicted = model.predict(x)

# 绘制拟合曲线图
plt.scatter(x_actual, y_actual, label='Actual')
plt.plot(x, y_predicted, color='r', label='Predicted')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Fitted Curve')
plt.legend()
plt.show()
```

这些可视化方案提供了不同的角度和方法来评估回归模型的效果。根据数据和模型的特点，可以选择适合的可视化方案或结合多种方案来全面评估模型的性能。

## 分类可视化方案

 决策边界可视化 以 `Perceptron` 为例

在训练好高精度的模型，我们可以通过有效的可视化直观看到分类效果，相比于混淆矩阵等分类指标更加直观。如下示例就可以看出iris数据集的Sepal (花萼）相比 Petal （花瓣）更难分类

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
# 加载鸢尾花数据集 
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data[:,2:], data.target, test_size=0.2)

# 创建并训练感知器模型
perceptron = Perceptron()
perceptron.fit(X_train, y_train)

# 绘制散点图（每个类别用不同颜色表示）
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train) # just draw the length and width of sepal ，
# and the c paremeter get the array will draw different  color in different digital
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

# 添加决策边界到图中
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min,y_max)) # depend on the two x and y lenth decide the array shape return the x and y axis np-array with interval 1 
# both have the same shape
# print(np.arange(x_min, x_max))
# print(np.arange(y_min,y_max))
# print(xx)
# print(xx.ravel())
# print(yy)
# print(yy.ravel())
Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()]) # draw the decision boundary (predict the per coordinate pair )
# print(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape) # replace to the every grid 
print(Z)
plt.contourf(xx ,yy ,Z,alpha=0.3)
plt.show()
accuary = sum(perceptron.predict(X_test) == y_test)/len(y_test) 
print(accuary)
```

![image-20230819093519419](data%20visualization.assets/image-20230819093519419.png)

或者我们的粒度再细一点，设置`np.arange(x_min, x_max, 0.2)`

![image-20230820111805061](data%20visualization.assets/image-20230820111805061.png)

![image-20230819093953006](data%20visualization.assets/image-20230819093953006.png)

对应的Prediction grid (可以看到反过来就是绘制等高线对应的图片）:

```
[[0 1 1 1 1 1 1 1]
 [0 0 1 1 1 1 1 1]
 [0 0 0 1 1 1 1 1]
 [2 2 2 2 2 2 2 1]
 [2 2 2 2 2 2 2 2]]
```

详解使用函数：

#### np.meshgrid()

`np.meshgrid()`函数用于生成一个二维网格，它以两个一维数组作为参数，分别表示 x 轴和 y 轴上的坐标点。该函数返回两个二维数组，这些**数组中的每个元素都代表了在坐标平面上某一点的 x 和 y 坐标。**

让我们来详细解释一下`np.meshgrid()`函数的具体用法：

```python
xx, yy = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min,y_max))
```

- `np.arange(x_min, x_max)`：这是一个 NumPy 函数，用于创建一个从 `x_min` 到 `x_max - 1` 的连续整数序列。它将作为参数传递给 `np.meshgrid()` 函数，并**指定了在 x 轴方向上生成网格点所需的范围。**

- `np.arange(y_min,y_max)`：类似地同上

- `xx, yy = np.meshgrid(...)`：通过调用`np.meshgrid(np.arange(x_min, x_max), np.arange(y_min,y_max))`可以得到 xx 和 yy 这两个变量。其中 xx 是一个与 y 方向长度相同、横轴值变化而纵轴不变（即 y 方向不变）的二维数组；yy 是一个与 x 方向长度相同、纵轴值变化而横轴不变（即 x 方向不变）的二维数组。

这个函数对于在整个坐标空间上进行预测和可视化非常有用，因为它生成了一个包含所有可能组合的坐标点网格。

#### np.ravel() & np.c_

`np.ravel()`函数用于将多维数组展平为一维数组。它会按照 **C 风格（行优先）的顺序**来展开数组。

`np.c_()`用于**按列连接**两个或多个数组。它可以将**一维数组沿着列方向进行拼接**，**生成一个新的二维数组**。

#### plt.contourf()

`plt.contourf()`用于绘制**等高线填充图**。它可以根据数据的值来为不同区域着色，并在图表上显示出这些颜色区域之间的边界。

让我们详细解释一下`plt.contourf()`函数的具体用法：

```python
plt.contourf(X, Y, Z)
```

- `X`：表示 x 坐标点的二维数组或网格矩阵。

- `Y`：表示 y 坐标点的二维数组或网格矩阵。

- `Z`：表示对应于 `(X, Y)` **网格点位置处某种属性（例如，高度、温度等）的数值。**

通过传递以上参数给`plt.contourf()`函数，我们可以生成一个由**等高线填充区域**组成的图表。其中每个填充区域都代表了相应坐标点处属性数值所在范围内部分。	

此外，您还可以使用其他参数来自定义等高线填充图：

- `levels`: 通过设置 levels 参数来指定要显示哪些特定数值范围内部分，默认情况下会自动选择合适数量和范围。

- `colors`: 可以使用 colors 参数来指定所使用颜色映射（colormap），也可以直接传递一个颜色列表作为参数进行手动设置。

通过使用`plt.contourf()`函数，您可以以视觉方式展示二维数据的分布情况，并更好地理解和呈现数据。

#### 总结

>  总体而言，整个可视化原理也比较清晰明了。大概流程如下：
>
>  1.  根据对应的数据数组特征的Min和Max确定对应的数据范围（Arrange)
>  2.  根据数据范围通过`meshgrip`生成对应表格二维数组（返回每一个点的x和y的值（shape `(len(x),len(y)`)
>  3.  对数据进行铺平操作（`np.ravel()`)和拼接成数组(`np.c_`)对作为特征数据进行预测网格的每一个点。
>  4.  通过`plt.contourf`对网格点的每一个预测结果作为其属性画不同颜色等高线实现决策边界的绘制。🎉



