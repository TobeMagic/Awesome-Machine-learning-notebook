## 评估方法

我们对模型要求是泛化能力，在评估模型重点就是划分：训练集、验证集和测试集，在训练集进行训练，在验证集进行评估，找到最优参数，再在测试集进行最后一次测试确定 最终最优超参数后，将验证集和训练集一起训练，最终用于评估测试集效果，这其中的重点便是坚决不能出现信息泄露，以下是一个例子：

使用训练集的均值和方差来标准化测试集的原因是为了保持数据的一致性和可比性。以下是几个关键原因：

1. 数据分布一致性：在典型的机器学习问题中，我们假设训练数据和测试数据来自同一分布。通过使用训练集的均值和方差来标准化测试集，我们可以确保在测试集上应用相同的数据转换，从而保持数据分布的一致性。
2. 特征归一化一致性：标准化（或称为归一化）是将不同特征的值缩放到相似范围的常见预处理步骤。通过使用训练集的均值和方差来标准化测试集，我们可以确保在不同数据集上应用相同的缩放规则，以便保持特征之间的一致性。
3. 模型泛化能力：通过在测试集上应用与训练集相同的标准化方法，我们可以更好地评估模型在真实世界数据上的泛化能力。如果我们使用测试集的均值和方差来标准化测试集，那么在评估模型性能时，我们会**将测试集的信息泄漏到模型中，导致评估结果过于乐观**。



在评估模型评价指标中，我们通常使用交叉验证方法和相关技术进行模型评估，

| 名称                                                         | 介绍                                                         | 优点                                                         | 缺点                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| K折交叉验证（K-fold Cross Validation）                       | 将训练集划分为K个互斥的子集，称为折叠。每次选择一个子集作为验证集，其余K-1个子集作为训练集。重复K次，每次选择不同的验证集。最后，计算K次验证结果的平均值作为模型的性能评估。 | - 充分利用了数据，提供了相对可靠的模型性能评估。<br>- 可以在**有限的数据集**上进行多次评估，减小了对单次划分的依赖。<br>- 可以用于调整模型的超参数。 | - 计算成本较高，需要训练和评估模型K次。<br>- 当数据集较大时，可能会浪费较多的计算资源。 |
| 打乱数据的重复K折交叉验证（Shuffle K-fold Cross Validation） | 就是在每次划分K个子集前打乱数据，该方法在Kaggle竞赛中非常有用，一共要训练P * K个模型 （P 是重复次数） | - 更加精确                                                   | - 同样的计算成本更高了                                       |
| 留一验证（Leave-One-Out Validation）                         | 特殊的K折交叉验证，其中K被设置为训练集的样本数。**对于每个样本，将其从训练集中移除，然后使用剩余的样本进行训练，并在被移除的样本上进行测试。**（单个样本） | - 在小样本数据集上提供了**最准确**的模型性能评估。<br>- 不会因为随机划分而引入偏差。 | - 计算成本**非常高**，特别是对于大型数据集。<br>- 对于大部分问题，一般不推荐使用，除非**数据集非常小**。 |
| 留P验证（Leave-P-Out Validation）                            | 类似于留一验证的交叉验证方法，**其中P个样本被从训练集中移除**，然后使用剩余的样本进行训练，并在被移除的样本上进行测试。 | - 在计算成本和模型性能之间提供了一种平衡选择。<br>- 可以根据问题的需求灵活选择P的大小。 | - 当P的值较大时，计算成本仍然较高。<br>- 可能需要进行多次评估以获取更稳定的性能评估结果。 |
| 自助法（Bootstrap）                                          | 通过**有放回**地从原始训练集中随机采样形成新的训练集。由于采样是有放回的，某些样本可能在新的训练集中出现多次，而其他样本可能在新的训练集中完全缺失。使用新的训练集进行模型训练，并使用原始训练集进行模型评估。 | - 有效地利用数据，尤其在数据集较小或样本分布不均匀的情况下。<br>- 可以提供对模型性能的较稳定估计。 | - 采样的随机性可能导致每次训练集的差异较大。<br>- 计算成本较高，可能需要进行多次评估以获取更稳定的性能评估结果。 |
| 时间序列交叉验证（Time Series Cross Validation）             | 专门针对**时间序列数据**设计的交叉验证方法。常见的方法包括滚动窗口交叉验证、时间序列分割交叉验证等，这些方法将数据集**按照时间顺序**划分为训练集和测试集，以模拟模型在未来时间点上的预测能力。 | - 能够正确反映出时间的因果关系，模拟模型在未来时间点上的预测能力。<br>- 可以用于调整模型的超参数。<br>- 可以提供对模型在不同时间段上的性能评估。 | - 对于较短的时间序列数据，可能会导致训练集和测试集之间的重叠较大。<br>- 时间序列的特殊性可能需要额外的处理和考虑。 |

这是一些常见的交叉验证方法和相关技术，每种方法都有其适用的情况和特点。根据具体问题和数据集的特点，选择合适的交叉验证方法可以提供可靠的模型性能评估和参数调优。 **在获取对应的指标后，我们需要绘制对应的指标结果，并对数据进行处理（删除或者平滑），以更好发现图片的规律寻找最优的训练轮次**。

### **K折交叉验证**

其中以下是一个 K折交叉验证 的模板代码

```python
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# 准备数据
X = # 特征数据
y = # 标签数据

# 定义模型
model = SVC()

# 定义K折交叉验证的折数
k = 5

# 创建K折交叉验证对象
kf = KFold(n_splits=k)

# 定义一个列表，用于保存每个折的模型评估结果
scores = []

# 进行K折交叉验证
for train_index, test_index in kf.split(X):
    # 划分训练集和测试集
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 在测试集上进行预测
    y_pred = model.predict(X_test)
    
    # 计算模型评估指标（这里以准确率为例）
    accuracy = accuracy_score(y_test, y_pred)
    
    # 将评估结果添加到列表中
    scores.append(accuracy)

# 计算K折交叉验证的平均性能指标
average_score = sum(scores) / k

# 打印每个折的评估结果和平均性能指标
for i, score in enumerate(scores):
    print(f"Fold {i+1} Accuracy: {score}")
    
print(f"Average Accuracy: {average_score}")
```

## 评估指标

### 分类

分类评估指标（以下代码均可在`sklearn.metrics`找到）:

1. 精确度（Accuracy）：分类正确的样本数占总样本数的比例。
2. 灵敏度（Sensitivity/Recall）：真实正类中被正确预测为正类的样本数占总的真实正类样本数的比例。
3. 特异度（Specificity）：真实负类中被正确预测为负类的样本数占总的真实负类样本数的比例。
4. 精确率（Precision）: 被预测为正类的样本中真正是正类的样本数占被预测为正类的样本数的比例。
5. F1值（F1-score）：综合考虑精确率和灵敏度，是**精确率和灵敏度的调和平均数**。
6. AUC值（Area Under the ROC Curve）：ROC曲线下方的面积，用于表示**分类器的整体性能**。

当对一个分类模型进行评估时，通常需要使用多个评估指标来综合考虑其性能。

#### 精确度（Accuracy）

精确度是指分类正确的样本数占总样本数的比例，是最简单直接的评估指标。

精确度计算公式如下：	

$$
Accuracy = \frac{TP + TN}{TP + FP + TN + FN} 
$$

其中，$TP$ 表示真正类（True Positive）的样本数，即被分类器正确预测为正类的样本数；$TN$ 表示真负类（True Negative）的样本数，即被分类器正确预测为负类的样本数；$FP$ 表示误报样本（False Positive）的样本数，即被分类器错误地预测为正类的样本数；$FN$ 表示漏报样本（False Negative）的样本数，即被分类器错误地预测为负类的样本数。

```python
from sklearn.metrics import accuracy_score

y_true = [0, 1, 0, 1]
y_pred = [0, 1, 1, 1]

accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)
```



#### 灵敏度（Sensitivity/Recall）

**灵敏度也称召回率**，是指真实正类中被正确预测为正类的样本数占总的真实正类样本数的比例。灵敏度能够反映出分类器对于正样本的识别能力。

灵敏度计算公式如下：

$$
Sensitivity = \frac{TP}{TP + FN} 
$$

```python
from sklearn.metrics import recall_score

recall = recall_score(y_true, y_pred)
print("Sensitivity/Recall:", recall)
```

#### 特异度（Specificity）

特异度是指真实负类中被正确预测为负类的样本数占总的真实负类样本数的比例。特异度能够反映出分类器对于负样本的识别能力。

特异度计算公式如下：

$$
Specificity = \frac{TN}{FP + TN}
$$

#### 精确率（Precision）

精确率是指被预测为正类的样本中真正是正类的样本数占被预测为正类的样本数的比例，能够反映出分类器对于正样本的预测准确性。

精确率计算公式如下：

$$
Precision = \frac{TP}{TP + FP} 
$$

#### F1值（F1-score）

F1值是**综合考虑精确率和灵敏度的调和平均数**，能够综合评价分类器的预测准确性和召回率。

F1值计算公式如下：

$$
F1 = 2 * \frac{Precision * Sensitivity}{Precision + Sensitivity} = \frac{2 * TP}{2 * TP + FP + FN}
$$

#### AUC值（Area Under the ROC Curve）

AUC（Area Under the Curve）是一种常用的评估分类模型性能的指标，通常用于ROC曲线（Receiver Operating Characteristic curve）分析。AUC表示ROC曲线下方的面积，其取值范围在0到1之间。

以下是对AUC指标的详细解释：

**1. ROC曲线：**

- ROC曲线是以二分类模型为基础绘制出来的一条图形。(如果是多分类，则需要绘制多条)

- 它展示了**当分类器阈值变化**时，真阳率（True Positive Rate, TPR）与假阳率（False Positive Rate, FPR）之间的关系。

- TPR表示**正确预测为正例样本占所有实际正例样本比例**（sensitivity\recall）；FPR表示**错误预测为正例样本占所有实际负例样本比例**（1 - specificity）。

   >  以下是绘制ROC曲线的步骤：
   >
   >  1. 收集模型预测结果和相应的真实标签。这些结果包括模型对每个样本的预测概率或分数以及它们对应的真实标签（0表示负例，1表示正例）。
   >
   >  2. **根据预测概率或分数对样本进行排序**。从高到低排列，使得排名最高的样本具有最大的预测概率或分数。
   >
   >  3. 选择一个**分类阈值**，并**根据该阈值将样本划分为正例和负例**。例如，如果阈值设置为0.5，则所有预测概率大于等于0.5的样本被视为正例，而小于0.5则被视为负例。
   >
   >  4. 计算此时的真正例率（TPR）和假正例率（FPR）。
   >
   >     TPR = TP / (TP + FN)
   >     
   >     FPR = FP / (FP + TN)
   >
   >  5. 重复步骤3和4，**使用不同分类阈值来计算一系列不同点对应的TPR和FPR**。这些点构成了ROC曲线上的各个坐标。
   >
   >  6. 绘制ROC曲线，以FPR作为x轴，TPR作为y轴。通过连接这些坐标点可以得到一条典型情况下具有平滑形状且递增趋势的曲线。
   >
   >  在理想情况下，ROC曲线会靠近左上角，并且与对角线之间存在较大距离。该区域被认为是模型性能最佳、**具有高度可区分能力和较小误判率的区域。**
   
   下面是一个使用Keras进行二分类训练并绘制ROC曲线的模板代码。此代码还包括计算交叉验证平均ROC曲线以及绘制置信区域的部分。
   
   ```python
   import numpy as np
   import matplotlib.pyplot as plt
   from sklearn.metrics import roc_curve, auc
   from sklearn.model_selection import StratifiedKFold
   from keras.models import Sequential
   from keras.layers import Dense
   
   # 创建模型（示例）
   def create_model():
       model = Sequential()
       model.add(Dense(16, input_dim=8, activation='relu'))
       model.add(Dense(8, activation='relu'))
       model.add(Dense(1, activation='sigmoid'))
       model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
       return model
   
   # 交叉验证绘制ROC曲线
   def plot_roc_cv(X, y, n_splits=5):
       cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
       tprs = []
       aucs = []
       mean_fpr = np.linspace(0, 1, 100)
       
       fig, ax = plt.subplots()
       for i, (train, test) in enumerate(cv.split(X, y)):
           model = create_model()
           model.fit(X[train], y[train], epochs=10, batch_size=32, verbose=0)
           y_pred = model.predict(X[test]).ravel()
           fpr, tpr, thresholds = roc_curve(y[test], y_pred)
           tprs.append(np.interp(mean_fpr, fpr, tpr))
           tprs[-1][0] = 0.0
           roc_auc = auc(fpr, tpr)
           aucs.append(roc_auc)
           ax.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc))
       
       ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random Guess')
       
       mean_tpr = np.mean(tprs, axis=0)
       mean_tpr[-1] = 1.0
       mean_auc = auc(mean_fpr, mean_tpr)
       std_auc = np.std(aucs)
       ax.plot(mean_fpr, mean_tpr, color='b', label='Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2)
       
       std_tpr = np.std(tprs, axis=0)
       tprs_upper = np.minimum(mean_tpr + 2 * std_tpr, 1)
       tprs_lower = np.maximum(mean_tpr - 2 * std_tpr, 0)
       ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='gray', alpha=0.2, label='95% Confidence Interval')
       
       ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title='Receiver Operating Characteristic')
       ax.legend(loc='lower right')
       plt.show()
   
   # 示例数据
   X = np.random.rand(100, 8)
   y = np.random.randint(0, 2, 100)
   
   # 绘制ROC曲线
   plot_roc_cv(X, y, n_splits=5)
   ```
   
   你可以将示例数据替换为你的实际数据，然后运行代码以获得交叉验证的平均ROC曲线和置信区域的绘图结果。



<img src="evaluation & metrics.assets/image-20231013144429358.png" alt="image-20231013144429358" style="zoom: 50%;" />

**2. 如何运用到多分类：**

在多分类问题中，我们可以将每个类别作为正例，并**计算出多个二分类子问题的ROC曲线**，并通过求解这些子问题下**各自点集合并取平均值**来获得整体的多类别ROC曲线（宏平均）。

为了绘制多类别的ROC曲线，在每个子问题上执行以下步骤：

-  将当前类别标记为正例，其他所有类别标记为负例。
-  计算预测概率或得分，并按照阈值确定预测结果。
-  根据不同阈值下的真阳率和假阳率绘制ROC曲线。

AUC更关注分类器**在不同阈值下判定真假阳性的表现**，因此它提供了一种更全面且相对鲁棒的评估方法。

#### 多分类指标（multiple classification index）

在面对多分类问题时，常用的指标包括准确率（Accuracy）、**混淆矩阵（Confusion Matrix）**以及宏平均（Macro-average）和微平均（Micro-average）。

1. 准确率：准确率是最简单直观的评估指标，表示模型正确预测的样本比例。对于多分类问题，准确率被定义为所有正确分类的样本数除以总样本数。

2. 混淆矩阵：混淆矩阵可以提供**更详细的多类别分类性能信息**。它是一个二维表格，行代表真实类别，列代表预测类别。每个单元格记录了属于特定真实类别和预测类别组合的样本数量。

   例如，在3个类别A、B、C下进行分类时，可能有以下情况：
   
   - 类A中有10个样本被正确地预测为A。
   - 类B中有5个样本被错误地预测为A。
   - 类C中有3个样本被错误地预测为A。
   - ...

   这些信息都可以通过混淆矩阵得到，并进一步计算其他指标如精确度、召回率等。

3. 宏平均与微平均：在处理多分类问题时，我们通常**需要将各种指标汇总成一个统一的度量**（即拆分成多个二分类子问题，最后求平均得到结果）。宏平均和微平均是两种常用的方法。

   - 宏平均：对**每个类别单独计算指标**（如`精确度、召回率`等），然后求取其算术平均值。它将**所有类别视为同等重要，适用于各个类别都具有相似重要性的情况**。
   
   - 微平均：将多分类问题视为二分类问题，在**所有样本上**进行计算指标（如精确度、召回率等）。这意味着每个预测都被认为是同等重要的，并且**更加关注少数类别**（如果不平均值会很低） 。适用于不同类别之间存在**明显不平衡**时使用。
   

无论是准确率、混淆矩阵还是宏/微平均，这些指标可以帮助我们评估模型在多分类任务中的整体性能以及对每个特定类别的预测能力。根据具体需求和问题背景，选择合适的评估指标来解读和分析结果非常重要。

##### `classification_report`（分类报告）

`classification_report`函数用于生成分类模型的性能报告，该报告提供了模型在每个类别上的精确度（precision）、召回率（recall）、F1-score和支持度（support）等指标。f		ws

具体来说，`classification_report`函数的输入是真实的目标标签（y_true）和模型预测的标签（y_pred），它会根据这些标签计算并显示每个类别的以下指标：

- 精确度（Precision）：分类正确的正样本数量与所有被预测为正样本的数量的比值。表示模型预测为正样本的触发的真实正样本的概率。
- 召回率（Recall）：分类正确的正样本数量与所有真实正样本的数量的比值。表示模型能够正确找到的真实正样本的比例。
- F1-score：精确度和召回率的加权调和平均值，用于综合考虑两者的性能。F1-score的取值范围是0到1，值越高表示模型的性能越好。
- 支持度（Support）：每个类别在真实标签中的样本数量。

`classification_report`的输出类似于下面的示例：

```
              precision    recall  f1-score   support

    class 0       0.80      0.90      0.85        30
    class 1       0.75      0.60      0.67        20
    class 2       0.92      0.97      0.94        50

    accuracy                          0.86       100
   macro avg      0.82      0.82      0.82       100
weighted avg      0.85      0.86      0.85       100
```

在这个示例中，有三个类别（class 0、class 1和class 2），模型的平均精确度、召回率和F1-score等指标都会被报告。

##### `confusion_matrix`（混淆矩阵）
`confusion_matrix`函数用于创建分类模型的混淆矩阵。混淆矩阵是一种以矩阵形式显示模型分类结果的方法，它可以帮助我们了解模型在每个类别上的预测情况。

混淆矩阵的行表示真实标签，列表示预测标签。矩阵的每个元素表示模型将样本预测为某个类别的数量。通过观察混淆矩阵，我们可以分析模型在不同类别上的预测准确性、错误分类等情况。

以下是一个二分类问题的混淆矩阵示例：

```
[[85 15]
 [20 80]]
```

在这个示例中，真实标签包含两个类别，模型的预测结果将样本划分为四个区域：真正例（True Positive，TP）、真反例（True Negative，TN）、假正例（False Positive，FP）和假反例（False Negative，FN）。

- TP：模型将正样本正确地预测为正样本的数量。
- TN：模型将负样本正确地预测为负样本的数量。
- FP：模型将负样本错误地预测为正样本的数量。
- FN：模型将正样本错误地预测为负样本的数量。

混淆矩阵提供了对模型性能的更详细的了解，例如通过计算准确率（accuracy）、精确度、召回率和F1-score等指标。

### 回归

回归模型的评估指标有很多种，以下是其中常见的几种：

1. 均方误差（Mean Squared Error, MSE）：预测值与真实值之间的平均误差的平方和。
2. 均方根误差（Root Mean Squared Error, RMSE）：均方误差的平方根，它展示了实际数据与拟合数据之间的误差程度。
3. 平均绝对误差（Mean Absolute Error, MAE）：预测值与真实值之间的平均绝对误差。
4. R方系数（R-squared）：又称决定系数，用于说明自变量对因变量的解释程度，其取值范围为0-1之间。
5. 重要性分析（Feature Importance）：用于衡量特征在建立模型时的重要性。

不同的评估指标适用于不同的应用场景，需要根据具体情况选择合适的评估指标。

#### 均方误差（Mean Squared Error, MSE）

均方误差是预测值与真实值之间的平均误差的平方和，它表示预测值与真实值之间的离散程度。

均方误差的公式如下：

$$ MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y_i})^2 $$

其中 $y_i$ 是第 $i$ 个样本的真实值，$\hat{y_i}$ 是第 $i$ 个样本的预测值，$n$ 是样本数量。

####  均方根误差（Root Mean Squared Error, RMSE）

均方根误差是均方误差的平方根，它展示了实际数据与拟合数据之间的误差程度，并且与原始数据的单位相同。

均方根误差的公式如下：

$$ RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y_i})^2} $$

其中 $y_i$ 是第 $i$ 个样本的真实值，$\hat{y_i}$ 是第 $i$ 个样本的预测值，$n$ 是样本数量。

#### 平均绝对误差（Mean Absolute Error, MAE）

平均绝对误差是预测值与真实值之间的平均绝对误差，它表示预测值与真实值之间的平均距离。

平均绝对误差的公式如下：

$$ MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y_i}| $$

其中 $y_i$ 是第 $i$ 个样本的真实值，$\hat{y_i}$ 是第 $i$ 个样本的预测值，$n$ 是样本数量。

####  R方系数（R-squared）

R方系数也称决定系数，用于说明自变量对因变量的解释程度，其取值范围为0-1之间。当R方系数越接近1时，模型的拟合效果越好。

R方系数的公式如下：

$$ R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y_i})^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} $$

其中 $y_i$ 是第 $i$ 个样本的真实值，$\hat{y_i}$ 是第 $i$ 个样本的预测值，$\bar{y}$ 是所有样本的平均值，$n$ 是样本数量。

#### 重要性分析（Feature Importance）

在建立模型的过程中，特征的选择对模型的影响至关重要。因此，对特征的重要性进行分析可以帮助了解模型的建立过程。

特征的重要性可以通过不同的方法进行计算，如树模型中的信息增益、线性模型中的系数大小等。



