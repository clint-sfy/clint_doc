# 机器学习基础

## 1.  线性回归原理推导

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/线性回归1.png)

## 2.  线性回归代码实现

### 通用代码

```python
import numpy as np
from utils.features import prepare_for_training

class LinearRegression:

    def __init__(self,data,labels,polynomial_degree = 0,sinusoid_degree = 0,normalize_data=True):
        """
        1.对数据进行预处理操作
        2.先得到所有的特征个数
        3.初始化参数矩阵
        """
        (data_processed,
         features_mean, 
         features_deviation) = prepare_for_training(data, polynomial_degree, sinusoid_degree,normalize_data=True)
        # print(data_processed)
        self.data = data_processed  # 预处理完的数据
        self.labels = labels  # 标签
        self.features_mean = features_mean  #
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data
        
        num_features = self.data.shape[1]
        self.theta = np.zeros((num_features,1))

    # train(学习率  迭代次数)
    def train(self,alpha,num_iterations = 500):
        """
                    训练模块，执行梯度下降
        """
        cost_history = self.gradient_descent(alpha,num_iterations)
        return self.theta,cost_history

    # 实现小批量梯度下降法
    def gradient_descent(self,alpha,num_iterations):
        """
                    实际迭代模块，会迭代num_iterations次
        """
        cost_history = []
        for _ in range(num_iterations):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data,self.labels))
        return cost_history

    # 实现小批量梯度下降法 具体步骤
    def gradient_step(self,alpha):    
        """
                    梯度下降参数更新计算方法，注意是矩阵运算
        """
        # 样本的个数
        num_examples = self.data.shape[0]
        # 预测值 predictions = np.dot(data,theta)
        prediction = LinearRegression.hypothesis(self.data,self.theta)
        # 残差 = 预测值 - 真实值
        delta = prediction - self.labels
        theta = self.theta
        # 更新theta 小批量梯度下降法
        # 公式直接带进去
        theta = theta - alpha*(1/num_examples)*(np.dot(delta.T,self.data)).T
        self.theta = theta

    def cost_function(self,data,labels):
        """
                    损失计算方法
        """
        # 该函数会计算模型预测值和实际值之间的平均误差的平方和，然后将其除以样本数量得到平均误差值作为损失值。
        num_examples = data.shape[0]
        delta = LinearRegression.hypothesis(self.data,self.theta) - labels
        cost = (1/2)*np.dot(delta.T,delta)/num_examples
        return cost[0][0]
        

    @staticmethod
    def hypothesis(data,theta):   
        predictions = np.dot(data,theta)
        return predictions

    # 得到当前的损失
    def get_cost(self,data,labels):  
        data_processed = prepare_for_training(data,
         self.polynomial_degree,
         self.sinusoid_degree,
         self.normalize_data
         )[0]
        
        return self.cost_function(data_processed,labels)

    def predict(self,data):
        """
                    用训练的参数模型，与预测得到回归值结果
        """
        data_processed = prepare_for_training(data,
         self.polynomial_degree,
         self.sinusoid_degree,
         self.normalize_data
         )[0]
         
        predictions = LinearRegression.hypothesis(data_processed,self.theta)
        
        return predictions
```

### 线性回归

 ```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linear_regression import LinearRegression

# 根据gdp 去预测幸福指数
data = pd.read_csv('../data/world-happiness-report-2017.csv')

# 得到训练和测试数据
train_data = data.sample(frac = 0.8)
test_data = data.drop(train_data.index)

input_param_name = 'Economy..GDP.per.Capita.'
output_param_name = 'Happiness.Score'

# 这里要注意！传数组进去！二维矩阵
x_train = train_data[[input_param_name]].values
y_train = train_data[[output_param_name]].values

# 这里是一维的
x_test = test_data[input_param_name].values
y_test = test_data[output_param_name].values

plt.scatter(x_train,y_train,label='Train data')
plt.scatter(x_test,y_test,label='test data')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Happy')
plt.legend()
plt.show()

# 开始训练回归模型
num_iterations = 500
learning_rate = 0.01

linear_regression = LinearRegression(x_train,y_train)
(theta,cost_history) = linear_regression.train(learning_rate,num_iterations)
# cost_history 是一个损失函数的列表
print ('开始时的损失：',cost_history[0])
print ('训练后的损失：',cost_history[-1])

plt.plot(range(num_iterations),cost_history)
plt.xlabel('Iter')
plt.ylabel('cost')
plt.title('GD')
plt.show()

predictions_num = 100
# 生成一个指定间隔序列 记得要变成一个二维数组 （，100）->（100,1）
x_predictions = np.linspace(x_train.min(),x_train.max(),predictions_num).reshape(predictions_num,1)
y_predictions = linear_regression.predict(x_predictions)

plt.scatter(x_train,y_train,label='Train data')
plt.scatter(x_test,y_test,label='test data')
plt.plot(x_predictions,y_predictions,'r',label = 'Prediction')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Happy')
plt.legend()
plt.show()
 ```

### 多特征回归模型

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go

plotly.offline.init_notebook_mode()
from linear_regression import LinearRegression
# 多特征回归模型

data = pd.read_csv('../data/world-happiness-report-2017.csv')

train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)

input_param_name_1 = 'Economy..GDP.per.Capita.'
input_param_name_2 = 'Freedom'
output_param_name = 'Happiness.Score'

x_train = train_data[[input_param_name_1, input_param_name_2]].values
y_train = train_data[[output_param_name]].values

x_test = test_data[[input_param_name_1, input_param_name_2]].values
y_test = test_data[[output_param_name]].values

# Configure the plot with training dataset.
plot_training_trace = go.Scatter3d(
    x=x_train[:, 0].flatten(),
    y=x_train[:, 1].flatten(),
    z=y_train.flatten(),
    name='Training Set',
    mode='markers',
    marker={
        'size': 10,
        'opacity': 1,
        'line': {
            'color': 'rgb(255, 255, 255)',
            'width': 1
        },
    }
)

plot_test_trace = go.Scatter3d(
    x=x_test[:, 0].flatten(),
    y=x_test[:, 1].flatten(),
    z=y_test.flatten(),
    name='Test Set',
    mode='markers',
    marker={
        'size': 10,
        'opacity': 1,
        'line': {
            'color': 'rgb(255, 255, 255)',
            'width': 1
        },
    }
)

plot_layout = go.Layout(
    title='Date Sets',
    scene={
        'xaxis': {'title': input_param_name_1},
        'yaxis': {'title': input_param_name_2},
        'zaxis': {'title': output_param_name} 
    },
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
)

plot_data = [plot_training_trace, plot_test_trace]

plot_figure = go.Figure(data=plot_data, layout=plot_layout)

plotly.offline.plot(plot_figure) # 如果要用jupter 改成plotly.offline.iplot

num_iterations = 500  
learning_rate = 0.01  
polynomial_degree = 0  
sinusoid_degree = 0  

linear_regression = LinearRegression(x_train, y_train, polynomial_degree, sinusoid_degree)

(theta, cost_history) = linear_regression.train(
    learning_rate,
    num_iterations
)

print('开始损失',cost_history[0])
print('结束损失',cost_history[-1])

plt.plot(range(num_iterations), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Gradient Descent Progress')
plt.show()

predictions_num = 10

x_min = x_train[:, 0].min();
x_max = x_train[:, 0].max();

y_min = x_train[:, 1].min();
y_max = x_train[:, 1].max();


x_axis = np.linspace(x_min, x_max, predictions_num)
y_axis = np.linspace(y_min, y_max, predictions_num)


x_predictions = np.zeros((predictions_num * predictions_num, 1))
y_predictions = np.zeros((predictions_num * predictions_num, 1))

x_y_index = 0
for x_index, x_value in enumerate(x_axis):
    for y_index, y_value in enumerate(y_axis):
        x_predictions[x_y_index] = x_value
        y_predictions[x_y_index] = y_value
        x_y_index += 1

z_predictions = linear_regression.predict(np.hstack((x_predictions, y_predictions)))

plot_predictions_trace = go.Scatter3d(
    x=x_predictions.flatten(),
    y=y_predictions.flatten(),
    z=z_predictions.flatten(),
    name='Prediction Plane',
    mode='markers',
    marker={
        'size': 1,
    },
    opacity=0.8,
    surfaceaxis=2, 
)

plot_data = [plot_training_trace, plot_test_trace, plot_predictions_trace]
plot_figure = go.Figure(data=plot_data, layout=plot_layout)
plotly.offline.plot(plot_figure)
```

### 非线性回归

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linear_regression import LinearRegression

# 非线性回归
data = pd.read_csv('../data/non-linear-regression-x-y.csv')

x = data['x'].values.reshape((data.shape[0], 1))
y = data['y'].values.reshape((data.shape[0], 1))

data.head(10)

plt.plot(x, y)
plt.show()

num_iterations = 50000  
learning_rate = 0.02
# 特征变换
polynomial_degree = 15
sinusoid_degree = 15
normalize_data = True  

linear_regression = LinearRegression(x, y, polynomial_degree, sinusoid_degree, normalize_data)

(theta, cost_history) = linear_regression.train(
    learning_rate,
    num_iterations
)

print('开始损失: {:.2f}'.format(cost_history[0]))
print('结束损失: {:.2f}'.format(cost_history[-1]))

theta_table = pd.DataFrame({'Model Parameters': theta.flatten()})


plt.plot(range(num_iterations), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Gradient Descent Progress')
plt.show()

predictions_num = 1000
x_predictions = np.linspace(x.min(), x.max(), predictions_num).reshape(predictions_num, 1);
y_predictions = linear_regression.predict(x_predictions)

plt.scatter(x, y, label='Training Dataset')
plt.plot(x_predictions, y_predictions, 'r', label='Prediction')
plt.show()
```


## 3. 模型评估方法

### 数据集切分

```python
import numpy as np
import os
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# Mnist数据是图像数据：(28,28,1)的灰度图
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
mnist


X.shape
X, y = mnist["data"], mnist["target"]
X.shape
(70000, 784)
y.shape
(70000,)
```

```python
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

import numpy as np
# 洗牌操作
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
```

### 交叉验证

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/模型评估1.png)

```python
y_train_5 = (y_train==5)
y_test_5 = (y_test==5)

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(max_iter=5,random_state=42)
sgd_clf.fit(X_train,y_train_5)


from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf,X_train,y_train_5,cv=3,scoring='accuracy')
array([0.9502 , 0.96565, 0.96495])

X_train.shape
(60000, 784)
y_train_5.shape
(60000,)
```

```python

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
​
skflods = StratifiedKFold(n_splits=3,random_state=42)
for train_index,test_index in skflods.split(X_train,y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_folds = X_train[test_index]
    y_test_folds = y_train_5[test_index]
    
    clone_clf.fit(X_train_folds,y_train_folds)
    y_pred = clone_clf.predict(X_test_folds)
    n_correct = sum(y_pred == y_test_folds)
    print(n_correct/len(y_pred))
0.9502
0.96565
0.96495
```



### 混淆矩阵

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/模型评估2.png)

```python
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf,X_train,y_train_5,cv=3)


from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5,y_train_pred)
array([[53272,  1307],
       [ 1077,  4344]], dtype=int64)
```

```
negative class [[ true negatives , false positives ],
positive class [ false negatives , true positives ]]

true negatives: 53,272个数据被正确的分为非5类别
false positives：1307张被错误的分为5类别
false negatives：1077张错误的分为非5类别
true positives： 4344张被正确的分为5类别
```

```python
# 准确率： TP/(TP+FP)
# 召回率： TP/(TP+FN)

from sklearn.metrics import precision_score,recall_score
precision_score(y_train_5,y_train_pred)
0.7687135020350381

recall_score(y_train_5,y_train_pred)
0.801328168234643


from sklearn.metrics import f1_score
f1_score(y_train_5,y_train_pred)
0.7846820809248555
```



### 评估指标对比分析



### 阈值对结果的影响

```PYTHON
Scikit-Learn不允许直接设置阈值，但它可以得到决策分数，调用其decision_function（）方法，而不是调用分类器的predict（）方法，该方法返回每个实例的分数，然后使用想要的阈值根据这些分数进行预测

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

def plot_precision_recall_vs_threshold(precisions,recalls,thresholds):
    plt.plot(thresholds,
             precisions[:-1],
            "b--",
            label="Precision")
    
    plt.plot(thresholds,
             recalls[:-1],
            "g-",
            label="Recall")
    plt.xlabel("Threshold",fontsize=16)
    plt.legend(loc="upper left",fontsize=16)
    plt.ylim([0,1])
    
plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions,recalls,thresholds)
plt.xlim([-700000, 700000])
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/模型评估3.png)

```python
def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, 
             precisions, 
             "b-", 
             linewidth=2)
    
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])

plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/模型评估4.png)

### ROC曲线

```
receiver operating characteristic (ROC) 曲线是二元分类中的常用评估方法

它与精确度/召回曲线非常相似，但ROC曲线不是绘制精确度与召回率，而是绘制true positive rate(TPR) 与false positive rate(FPR)

要绘制ROC曲线，首先需要使用roc_curve（）函数计算各种阈值的TPR和FPR：

TPR = TP / (TP + FN) (Recall)
FPR = FP / (FP + TN)
```

```python
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)

plt.figure(figsize=(8, 6))
plot_roc_curve(fpr, tpr)
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/模型评估5.png)

```
虚线表示纯随机分类器的ROC曲线; 一个好的分类器尽可能远离该线（朝左上角）。

比较分类器的一种方法是测量曲线下面积（AUC）。完美分类器的ROC AUC等于1，而纯随机分类器的ROC AUC等于0.5。 Scikit-Learn提供了计算ROC AUC的函数：
```

```python
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)
0.9624496555967156
```

## 4. 线性回归实验分析

### 回归方程

```python
import numpy as np
X = 2*np.random.rand(100,1)
y = 4+ 3*X +np.random.randn(100,1)

X_b = np.c_[np.ones((100,1)),X]
# array([[1.        , 0.74908024],

# linalg.inv 1求逆
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
theta_best
array([[4.21509616],
       [2.77011339]])

X_new = np.array([[0],[2]])
X_new_b = np.c_[np.ones((2,1)),X_new]
y_predict = X_new_b.dot(theta_best)
y_predict
array([[4.21509616],
       [9.75532293]])
```

### 梯度下降

#### 批量梯度下降计算公式

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/回归分析2.png)

```python
eta = 0.1 # 学习率
n_iterations = 1000
m = 100 # 样本个数
theta = np.random.randn(2,1)
for iteration in range(n_iterations):
    gradients = 2/m* X_b.T.dot(X_b.dot(theta)-y)
    theta = theta - eta*gradients


theta
array([[4.21509616],
       [2.77011339]])
theta_path_bgd = []
def plot_gradient_descent(theta,eta,theta_path = None):
    m = len(X_b)
    plt.plot(X,y,'b.')
    n_iterations = 1000
    for iteration in range(n_iterations): 
        y_predict = X_new_b.dot(theta)
        plt.plot(X_new,y_predict,'b-')
        gradients = 2/m* X_b.T.dot(X_b.dot(theta)-y)
        theta = theta - eta*gradients
        if theta_path is not None:
            theta_path.append(theta)
    plt.xlabel('X_1')
    plt.axis([0,2,0,15])
    plt.title('eta = {}'.format(eta))

theta = np.random.randn(2,1)

plt.figure(figsize=(10,4))
plt.subplot(131)
plot_gradient_descent(theta,eta = 0.02)
plt.subplot(132)
plot_gradient_descent(theta,eta = 0.1,theta_path=theta_path_bgd)
plt.subplot(133)
plot_gradient_descent(theta,eta = 0.5)
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/回归分析3.png)

#### 随机梯度下降

```python
theta_path_sgd=[]
m = len(X_b)
np.random.seed(42)
n_epochs = 50

t0 = 5
t1 = 50

def learning_schedule(t):
    return t0/(t1+t)

theta = np.random.randn(2,1)

for epoch in range(n_epochs):
    for i in range(m):
        if epoch < 10 and i<10:
            y_predict = X_new_b.dot(theta)
            plt.plot(X_new,y_predict,'r-')
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2* xi.T.dot(xi.dot(theta)-yi)
        eta = learning_schedule(epoch*m+i)
        theta = theta-eta*gradients
        theta_path_sgd.append(theta)
        
plt.plot(X,y,'b.')
plt.axis([0,2,0,15])   
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/回归分析4.png)

#### MiniBatch梯度下降

```python
theta_path_mgd=[]
n_epochs = 50
minibatch = 16
theta = np.random.randn(2,1)
t0, t1 = 200, 1000
def learning_schedule(t):
    return t0 / (t + t1)
np.random.seed(42)
t = 0
for epoch in range(n_epochs):
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for i in range(0,m,minibatch):
        t+=1
        xi = X_b_shuffled[i:i+minibatch]
        yi = y_shuffled[i:i+minibatch]
        gradients = 2/minibatch* xi.T.dot(xi.dot(theta)-yi)
        eta = learning_schedule(t)
        theta = theta-eta*gradients
        theta_path_mgd.append(theta)
theta
array([[4.25490684],
       [2.80388785]])
```

#### 3种策略的对比实验

```python
# 实际当中用minibatch比较多，一般情况下选择batch数量应当越大越好。
# 批量梯度下降
theta_path_bgd = np.array(theta_path_bgd)
# 随机梯度下降
theta_path_sgd = np.array(theta_path_sgd)
# MiniBatch梯度下降
theta_path_mgd = np.array(theta_path_mgd)
plt.figure(figsize=(12,6))
plt.plot(theta_path_sgd[:,0],theta_path_sgd[:,1],'r-s',linewidth=1,label='SGD')
plt.plot(theta_path_mgd[:,0],theta_path_mgd[:,1],'g-+',linewidth=2,label='MINIGD')
plt.plot(theta_path_bgd[:,0],theta_path_bgd[:,1],'b-o',linewidth=3,label='BGD')
plt.legend(loc='upper left')
plt.axis([3.5,4.5,2.0,4.0])
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/回归分析5.png)

### 多项式回归

```python
m = 100
X = 6*np.random.rand(m,1) - 3
y = 0.5*X**2+X+np.random.randn(m,1)

from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree = 2,include_bias = False)
X_poly = poly_features.fit_transform(X)

X_poly[0]
array([2.82919615, 8.00435083]) # x x^2



from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_poly,y)
print (lin_reg.coef_)
print (lin_reg.intercept_)
[[1.10879671 0.53435287]]
[-0.03765461]

X_new = np.linspace(-3,3,100).reshape(100,1)
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)
plt.plot(X,y,'b.')
plt.plot(X_new,y_new,'r--',label='prediction')
plt.axis([-3,3,-5,10])
plt.legend()
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/回归分析6.png)

```python
# 特征变换的越复杂，得到的结果过拟合风险越高，不建议做的特别复杂。
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
plt.figure(figsize=(12,6))
for style,width,degree in (('g-',1,100),('b--',1,2),('r-+',1,1)):
    poly_features = PolynomialFeatures(degree = degree,include_bias = False)
    std = StandardScaler()
    lin_reg = LinearRegression()
    polynomial_reg = Pipeline([('poly_features',poly_features),
             ('StandardScaler',std),
             ('lin_reg',lin_reg)])
    polynomial_reg.fit(X,y)
    y_new_2 = polynomial_reg.predict(X_new)
    plt.plot(X_new,y_new_2,style,label = 'degree   '+str(degree),linewidth = width)
plt.plot(X,y,'b.')
plt.axis([-3,3,-5,10])
plt.legend()
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/回归分析7.png)

### 数据样本数量对结果的影响

```python
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model,X,y):
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size = 0.2,random_state=100)
    train_errors,val_errors = [],[]
    for m in range(1,len(X_train)):
        model.fit(X_train[:m],y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m],y_train_predict[:m]))
        val_errors.append(mean_squared_error(y_val,y_val_predict))
    plt.plot(np.sqrt(train_errors),'r-+',linewidth = 2,label = 'train_error')
    plt.plot(np.sqrt(val_errors),'b-',linewidth = 3,label = 'val_error')
    plt.xlabel('Trainsing set size')
    plt.ylabel('RMSE')
    plt.legend()
    
# 数据量越少，训练集的效果会越好，但是实际测试效果很一般。实际做模型的时候需要参考测试集和验证集的效果。
lin_reg = LinearRegression()
plot_learning_curves(lin_reg,X,y)
plt.axis([0,80,0,3.3])
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/回归分析8.png)

### 多项式回归的过拟合风险

```python
# 越复杂 越拟合
polynomial_reg = Pipeline([('poly_features',PolynomialFeatures(degree = 25,include_bias = False)),
             ('lin_reg',LinearRegression())])
plot_learning_curves(polynomial_reg,X,y)
plt.axis([0,80,0,5])
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/回归分析9.png)

### 正则化

对权重参数进行惩罚，让权重参数尽可能平滑一些，有两种不同的方法来进行正则化惩罚:

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/回归分析10.png)

```python
from sklearn.linear_model import Ridge
np.random.seed(42)
m = 20
X = 3*np.random.rand(m,1)
y = 0.5 * X +np.random.randn(m,1)/1.5 +1
X_new = np.linspace(0,3,100).reshape(100,1)

def plot_model(model_calss,polynomial,alphas,**model_kargs):
    for alpha,style in zip(alphas,('b-','g--','r:')):
        model = model_calss(alpha,**model_kargs)
        if polynomial:
            model = Pipeline([('poly_features',PolynomialFeatures(degree =10,include_bias = False)),
             ('StandardScaler',StandardScaler()),
             ('lin_reg',model)])
        model.fit(X,y)
        y_new_regul = model.predict(X_new)
        lw = 2 if alpha > 0 else 1
        plt.plot(X_new,y_new_regul,style,linewidth = lw,label = 'alpha = {}'.format(alpha))
    plt.plot(X,y,'b.',linewidth =3)
    plt.legend()
 
plt.figure(figsize=(14,6))
plt.subplot(121)
plot_model(Ridge,polynomial=False,alphas = (0,10,100))
plt.subplot(122)
plot_model(Ridge,polynomial=True,alphas = (0,10**-5,1))
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/回归分析11.png)

惩罚力度越大，alpha值越大的时候，得到的决策方程越平稳。

```python
from sklearn.linear_model import Lasso

plt.figure(figsize=(14,6))
plt.subplot(121)
plot_model(Lasso,polynomial=False,alphas = (0,0.1,1))
plt.subplot(122)
plot_model(Lasso,polynomial=True,alphas = (0,10**-1,1))
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/回归分析12.png)



## 5.  逻辑回归



## 6. 逻辑回归代码实现



## 7.  逻辑回归实验分析

```
回归是如何变成分类的呢？
之前在线性回归问题中，得到了具体的回归值，如果此时任务要做一个二分类该怎么办呢？

如果可以将连续的数值转换成对应的区间，这样就可以完成分类任务了，逻辑回归中借助sigmoid函数完成了数值映射，通过概率值比较来完成分类任务
```

```python
import numpy as np
import os
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

t = np.linspace(-10, 10, 100)
sig = 1 / (1 + np.exp(-t))
plt.figure(figsize=(9, 3))
plt.plot([-10, 10], [0, 0], "k-")
plt.plot([-10, 10], [0.5, 0.5], "k:")
plt.plot([-10, 10], [1, 1], "k:")
plt.plot([0, 0], [-1.1, 1.1], "k-")
plt.plot(t, sig, "b-", linewidth=2, label=r"$\sigma(t) = \frac{1}{1 + e^{-t}}$")
plt.xlabel("t")
plt.legend(loc="upper left", fontsize=20)
plt.axis([-10, 10, -0.1, 1.1])
plt.title('Figure 4-21. Logistic function')
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/7.1.png)

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/7.2.png)

### 数据集

```python
from sklearn import datasets
iris = datasets.load_iris()
list(iris.keys())
['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename']

X = iris['data'][:,3:]
y = (iris['target'] == 2).astype(np.int)


LogisticRegression()
log_res.fit(X,y)
from sklearn.linear_model import LogisticRegression
log_res = LogisticRegression()

log_res.fit(X,y)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='warn',
          n_jobs=None, penalty='l2', random_state=None, solver='warn',
          tol=0.0001, verbose=0, warm_start=False)

X_new = np.linspace(0,3,1000).reshape(-1,1)
y_proba = log_res.predict_proba(X_new)
y_proba
array([[0.98554411, 0.01445589],
       
 # 随着输入特征数值的变化，结果概率值也会随之变化
       plt.figure(figsize=(12,5))
decision_boundary = X_new[y_proba[:,1]>=0.5][0]
plt.plot([decision_boundary,decision_boundary],[-1,2],'k:',linewidth = 2)
plt.plot(X_new,y_proba[:,1],'g-',label = 'Iris-Virginica')
plt.plot(X_new,y_proba[:,0],'b--',label = 'Not Iris-Virginica')
plt.arrow(decision_boundary,0.08,-0.3,0,head_width = 0.05,head_length=0.1,fc='b',ec='b')
plt.arrow(decision_boundary,0.92,0.3,0,head_width = 0.05,head_length=0.1,fc='g',ec='g')
plt.text(decision_boundary+0.02,0.15,'Decision Boundary',fontsize = 16,color = 'k',ha='center')
plt.xlabel('Peta width(cm)',fontsize = 16)
plt.ylabel('y_proba',fontsize = 16)
plt.axis([0,3,-0.02,1.02])
plt.legend(loc = 'center left',fontsize = 16)
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/7.3.png)

```python
X = iris['data'][:,(2,3)]
y = (iris['target']==2).astype(np.int)

from sklearn.linear_model import LogisticRegression
log_res = LogisticRegression(C = 10000)
log_res.fit(X,y)
LogisticRegression(C=10000, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='warn',
          n_jobs=None, penalty='l2', random_state=None, solver='warn',
          tol=0.0001, verbose=0, warm_start=False)

X[:,0].min(),X[:,0].max()
(1.0, 6.9)
X[:,1].min(),X[:,1].max()
(0.1, 2.5)
```

### 决策边界的绘制

```
构建坐标数据，合理的范围当中，根据实际训练时输入数据来决定
整合坐标点，得到所有测试输入数据坐标点
预测，得到所有点的概率值
绘制等高线，完成决策边界
```

```python
x0,x1 = np.meshgrid(np.linspace(1,2,2).reshape(-1,1),np.linspace(10,20,3).reshape(-1,1))


x0
array([[1., 2.],
       [1., 2.],
       [1., 2.]])

array([[10., 10.],
       [15., 15.],
       [20., 20.]])

np.c_[x0.ravel(),x1.ravel()]
array([[ 1., 10.],
       [ 2., 10.],
       [ 1., 15.],
       [ 2., 15.],
       [ 1., 20.],
       [ 2., 20.]])
x0,x1 = np.meshgrid(np.linspace(2.9,7,500).reshape(-1,1),np.linspace(0.8,2.7,200).reshape(-1,1))
X_new = np.c_[x0.ravel(),x1.ravel()]
X_new
y_proba = log_res.predict_proba(X_new)

plt.figure(figsize=(10,4))
plt.plot(X[y==0,0],X[y==0,1],'bs')
plt.plot(X[y==1,0],X[y==1,1],'g^')

zz = y_proba[:,1].reshape(x0.shape)
contour = plt.contour(x0,x1,zz,cmap=plt.cm.brg)
plt.clabel(contour,inline = 1)
plt.axis([2.9,7,0.8,2.7])
plt.text(3.5,1.5,'NOT Vir',fontsize = 16,color = 'b')
plt.text(6.5,2.3,'Vir',fontsize = 16,color = 'g')
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/7.4.png)

### 多类别分类

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/7.5.png)

```python
X = iris['data'][:,(2,3)]
y = iris['target']


softmax_reg = LogisticRegression(multi_class = 'multinomial',solver='lbfgs')
softmax_reg.fit(X,y)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='multinomial',
          n_jobs=None, penalty='l2', random_state=None, solver='lbfgs',
          tol=0.0001, verbose=0, warm_start=False)
softmax_reg.predict([[5,2]])
softmax_reg.predict_proba([[5,2]])

x0, x1 = np.meshgrid(
        np.linspace(0, 8, 500).reshape(-1, 1),
        np.linspace(0, 3.5, 200).reshape(-1, 1),
    )
X_new = np.c_[x0.ravel(), x1.ravel()]


y_proba = softmax_reg.predict_proba(X_new)
y_predict = softmax_reg.predict(X_new)

zz1 = y_proba[:, 1].reshape(x0.shape)
zz = y_predict.reshape(x0.shape)

plt.figure(figsize=(10, 4))
plt.plot(X[y==2, 0], X[y==2, 1], "g^", label="Iris-Virginica")
plt.plot(X[y==1, 0], X[y==1, 1], "bs", label="Iris-Versicolor")
plt.plot(X[y==0, 0], X[y==0, 1], "yo", label="Iris-Setosa")

from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])

plt.contourf(x0, x1, zz, cmap=custom_cmap)
contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
plt.clabel(contour, inline=1, fontsize=12)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 7, 0, 3.5])
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/7.6.png)

## 8. 聚类算法原理

kmeans

优势：
简单，快速，适合常规数据集
K值难确定
劣势：
复杂度与样本呈线性关系
很难发现任意形状的簇  

db聚类

优势：
不需要指定簇个数
擅长找到离群点（检测任务）
可以发现任意形状的簇
两个参数就够了

劣势

高维数据有些困难（可以做降维）
Sklearn中效率很慢（数据削减策略）
参数难以选择（参数对结果的影响非常

## 9. Kmeans代码实现



```python

```

## 10. 聚类算法实验分析

### Kmeans

```python
import numpy as np
import os
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

from sklearn.datasets import make_blobs

blob_centers = np.array(
    [[0.2,2.3],
     [-1.5,2.3],
     [-2.8,1.8],
     [-2.8,2.8],
     [-2.8,1.3]])

blob_std =np.array([0.4,0.3,0.1,0.1,0.1]) 
X,y = make_blobs(n_samples=2000,centers=blob_centers,
                     cluster_std = blob_std,random_state=7)

def plot_clusters(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)
plt.figure(figsize=(8, 4))
plot_clusters(X)
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/10.1.png)

### 决策边界

```python
from sklearn.cluster import KMeans
k = 5
kmeans = KMeans(n_clusters = k,random_state=42)
y_pred =  kmeans.fit_predict(X)

# fit_predict(X)与kmeans.labels_ 得到预测结果是一致的
# 分类标签
y_pred
kmeans.labels_ 
array([4, 0, 1, ..., 2, 1, 0])
# 中心点
kmeans.cluster_centers_
array([[-2.80389616,  1.80117999],
       [ 0.20876306,  2.25551336],
       [-2.79290307,  2.79641063],
       [-1.46679593,  2.28585348],
       [-2.80037642,  1.30082566]])
```

```python
X_new = np.array([[0,2],[3,2],[-3,3],[-3,2.5]])
kmeans.predict(X_new)
array([1, 1, 2, 2])

kmeans.transform(X_new)
array([[2.81093633, 0.32995317, 2.9042344 , 1.49439034, 2.88633901],
       [5.80730058, 2.80290755, 5.84739223, 4.4759332 , 5.84236351],
       [1.21475352, 3.29399768, 0.29040966, 1.69136631, 1.71086031],
       [0.72581411, 3.21806371, 0.36159148, 1.54808703, 1.21567622]])

def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=30, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=50, linewidths=50,
                color=cross_color, zorder=11, alpha=1)

def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom='off')
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft='off')
```

```python
plt.figure(figsize=(8, 4))
plot_decision_boundaries(kmeans, X)
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/10.2.png)

### 算法流程

```python
# 展示了迭代的过程
kmeans_iter1 = KMeans(n_clusters = 5,init = 'random',n_init = 1,max_iter=1,random_state=1)
kmeans_iter2 = KMeans(n_clusters = 5,init = 'random',n_init = 1,max_iter=2,random_state=1)
kmeans_iter3 = KMeans(n_clusters = 5,init = 'random',n_init = 1,max_iter=3,random_state=1)

kmeans_iter1.fit(X)
kmeans_iter2.fit(X)
kmeans_iter3.fit(X)

plt.figure(figsize=(12,8))
plt.subplot(321)
plot_data(X)
plot_centroids(kmeans_iter1.cluster_centers_, circle_color='r', cross_color='k')
plt.title('Update cluster_centers')

plt.subplot(322)
plot_decision_boundaries(kmeans_iter1, X,show_xlabels=False, show_ylabels=False)
plt.title('Label')

plt.subplot(323)
plot_decision_boundaries(kmeans_iter1, X,show_xlabels=False, show_ylabels=False)
plot_centroids(kmeans_iter2.cluster_centers_,)

plt.subplot(324)
plot_decision_boundaries(kmeans_iter2, X,show_xlabels=False, show_ylabels=False)

plt.subplot(325)
plot_decision_boundaries(kmeans_iter2, X,show_xlabels=False, show_ylabels=False)
plot_centroids(kmeans_iter3.cluster_centers_,)

plt.subplot(326)
plot_decision_boundaries(kmeans_iter3, X,show_xlabels=False, show_ylabels=False)

plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/10.3.png)

### 不稳定的结果

初始的随机种子不一样，得到的结果也不同

```python
def plot_clusterer_comparison(c1,c2,X):
    c1.fit(X)
    c2.fit(X)
    
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plot_decision_boundaries(c1,X)
    plt.subplot(122)
    plot_decision_boundaries(c2,X)
    
c1 = KMeans(n_clusters = 5,init='random',n_init = 1,random_state=11) # 随机种子 初始中心点
c2 = KMeans(n_clusters = 5,init='random',n_init = 1,random_state=19)
plot_clusterer_comparison(c1,c2,X)
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/10.4.png)

### 评估方法

```python
# Inertia指标：每个样本与其质心的距离

kmeans.inertia_ # 希望越小越好
211.5985372581684

#transform得到的是当前样本到每个簇中心距离
kmeans.transform(X)
array([[0.46779778, 3.04611916, 1.45402521, 1.54944305, 0.11146795],
       [0.07122059, 3.11541584, 0.99002955, 1.48612753, 0.51431557],
       [3.81713488, 1.32016676, 4.09069201, 2.67154781, 3.76340605],
       ...,
       [0.92830156, 3.04886464, 0.06769209, 1.40795651, 1.42865797],
       [3.10300136, 0.14895409, 3.05913478, 1.71125   , 3.23385668],
       [0.22700281, 2.8625311 , 0.85434589, 1.21678483, 0.67518173]])

kmeans.labels_
array([4, 0, 1, ..., 2, 1, 0])
X_dist[np.arange(len(X_dist)),kmeans.labels_]
array([0.11146795, 0.07122059, 1.32016676, ..., 0.06769209, 0.14895409,
       0.22700281])

np.sum(X_dist[np.arange(len(X_dist)),kmeans.labels_]**2)
211.59853725816856
kmeans.score(X)
-211.59853725816856
```

### 找到最佳簇数

```python
# 如果k值越大，得到的结果肯定会越来越小！！！
kmeans_per_k = [KMeans(n_clusters = k).fit(X) for k in range(1,10)]
inertias = [model.inertia_ for model in kmeans_per_k]
plt.figure(figsize=(8,4))
plt.plot(range(1,10),inertias,'bo-')
plt.axis([1,8.5,0,1300])
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/10.5.png)

### 轮廓系数

另一种判断k值的 方法

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/10.6.png)

```python
from sklearn.metrics import silhouette_score 
# 轮廓系数
silhouette_score(X,kmeans.labels_)
0.655517642572828

kmeans_per_k
[KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
     n_clusters=1, n_init=10, n_jobs=None, precompute_distances='auto',
     random_state=None, tol=0.0001, verbose=0),
 KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
     n_clusters=2, n_init=10, n_jobs=None, precompute_distances='auto',
     random_state=None, tol=0.0001, verbose=0)
 
silhouette_scores = [silhouette_score(X,model.labels_) for model in kmeans_per_k[1:]]
# 轮廓系数越接近1 说明越合理  但不一定最合理 只是参考
plt.figure(figsize=(8,4))
plt.plot(range(2,10),silhouette_scores,'bo-')
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/10.7.png)

### Kmeans存在的问题

```python
# 评估只能当做参考
X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
X2 = X2 + [6, -8]
X = np.r_[X1, X2]
y = np.r_[y1, y2]

plot_data(X)
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/10.8.png)

```python
# 玩赖的 初始值规定了比较合理三个中心
# 事实上 即使inertia比较小 看出来分类也比较不合理
kmeans_good = KMeans(n_clusters=3,init=np.array([[-1.5,2.5],[0.5,0],[4,0]]),n_init=1,random_state=42)
kmeans_bad = KMeans(n_clusters=3,random_state=42)
kmeans_good.fit(X)
kmeans_bad.fit(X)

plt.figure(figsize = (10,4))
plt.subplot(121)
plot_decision_boundaries(kmeans_good,X)
plt.title('Good - inertia = {}'.format(kmeans_good.inertia_))
 
plt.subplot(122)
plot_decision_boundaries(kmeans_bad,X)
plt.title('Bad - inertia = {}'.format(kmeans_bad.inertia_))
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/10.9.png)

### 图像分割小例子

```python
#ladybug.png 把向日葵从图像中分割出来
from matplotlib.image import imread
image = imread('ladybug.png')
image.shape
(533, 800, 3)


X = image.reshape(-1,3)
X.shape
(426400, 3)

kmeans = KMeans(n_clusters = 8,random_state=42).fit(X)
# 拿到中心位置
kmeans.cluster_centers_

# 需要还原成三维  图片
segmented_img = kmeans.cluster_centers_[kmeans.labels_].reshape(533, 800, 3)

segmented_imgs = []
n_colors = (10,8,6,4,2)
# 不同聚类中心进行展示
for n_cluster in n_colors:
    kmeans = KMeans(n_clusters = n_cluster,random_state=42).fit(X)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    segmented_imgs.append(segmented_img.reshape(image.shape))
    
plt.figure(figsize=(10,5))
plt.subplot(231)
plt.imshow(image)
plt.title('Original image')

for idx,n_clusters in enumerate(n_colors):
    plt.subplot(232+idx)
    plt.imshow(segmented_imgs[idx])
    plt.title('{}colors'.format(n_clusters))
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/10.10.png)

### 半监督学习

用逻辑回归识别手写体数字，可以用无监督聚类提高准确率

```python
# 首先，让我们将训练集聚类为50个集群， 然后对于每个聚类，让我们找到最靠近质心的图像。 我们将这些图像称为代表性图像
from sklearn.datasets import load_digits

X_digits,y_digits = load_digits(return_X_y = True)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X_digits,y_digits,random_state=42)
# 有1347个手写体，8*8的像素点
X_train.shape 
(1347, 64)
y_train.shape
(1347,)
```

```python
#直接用逻辑回归的话 一般
#训练集只选了前50个
from sklearn.linear_model import LogisticRegression
n_labeled = 50

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])
log_reg.score(X_test, y_test)
0.8266666666666667

# 对这前50个进行聚类
k = 50
kmeans = KMeans(n_clusters=k, random_state=42)
X_digits_dist = kmeans.fit_transform(X_train)
X_digits_dist.shape
(1347, 50)
# 它用于返回数组中最小元素的索引
representative_digits_idx = np.argmin(X_digits_dist,axis=0)
representative_digits_idx.shape
(50,)

X_representative_digits = X_train[representative_digits_idx]

# 现在让我们绘制这些代表性图像并手动标记它们：
plt.figure(figsize=(8, 2))
for index, X_representative_digit in enumerate(X_representative_digits):
    plt.subplot(k // 10, 10, index + 1)
    plt.imshow(X_representative_digit.reshape(8, 8), cmap="binary", interpolation="bilinear")
    plt.axis('off')

plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/10.11.png)

```python
y_representative_digits = np.array([
    4, 8, 0, 6, 8, 3, 7, 7, 9, 2,
    5, 5, 8, 5, 2, 1, 2, 9, 6, 1,
    1, 6, 9, 0, 8, 3, 0, 7, 4, 1,
    6, 5, 2, 4, 1, 8, 6, 3, 9, 2,
    4, 2, 9, 4, 7, 6, 2, 3, 1, 1])
# 现在我们有一个只有50个标记实例的数据集，它们中的每一个都是其集群的代表性图像，而不是完全随机的实例。 让我们看看性能是否更好：

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_representative_digits, y_representative_digits)
log_reg.score(X_test, y_test)
0.9244444444444444
```

```python
# 但也许我们可以更进一步：如果我们将标签传播到同一群集中的所有其他实例，该怎么办？
y_train_propagated = np.empty(len(X_train), dtype=np.int32)
for i in range(k):
    y_train_propagated[kmeans.labels_==i] = y_representative_digits[i]
    
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train_propagated)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='warn',
          n_jobs=None, penalty='l2', random_state=42, solver='warn',
          tol=0.0001, verbose=0, warm_start=False)

log_reg.score(X_test, y_test)
0.9288888888888889
```

```python
#只选择前20个来试试
percentile_closest = 20

X_cluster_dist = X_digits_dist[np.arange(len(X_train)), kmeans.labels_]
for i in range(k):
    in_cluster = (kmeans.labels_ == i)
    cluster_dist = X_cluster_dist[in_cluster] #选择属于当前簇的所有样本
    cutoff_distance = np.percentile(cluster_dist, percentile_closest) #排序找到前20个
    above_cutoff = (X_cluster_dist > cutoff_distance) # False True结果
    X_cluster_dist[in_cluster & above_cutoff] = -1

partially_propagated = (X_cluster_dist != -1)
X_train_partially_propagated = X_train[partially_propagated]
y_train_partially_propagated = y_train_propagated[partially_propagated]

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)

log_reg.score(X_test, y_test)
0.9422222222222222
```

### DBSCAN

```python
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)

plt.plot(X[:,0],X[:,1],'b.')
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/10.12.png)

```python
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps = 0.05,min_samples=5)
dbscan.fit(X)
DBSCAN(algorithm='auto', eps=0.05, leaf_size=30, metric='euclidean',
    metric_params=None, min_samples=5, n_jobs=None, p=None)


dbscan.labels_[:10]
array([ 0,  2, -1, -1,  1,  0,  0,  0,  2,  5], dtype=int64)
dbscan.core_sample_indices_[:10]
array([ 0,  4,  5,  6,  7,  8, 10, 11, 12, 13], dtype=int64)
np.unique(dbscan.labels_) #相当于看集合了
array([-1,  0,  1,  2,  3,  4,  5,  6], dtype=int64)

dbscan2 = DBSCAN(eps = 0.2,min_samples=5)
dbscan2.fit(X)
DBSCAN(algorithm='auto', eps=0.2, leaf_size=30, metric='euclidean',
    metric_params=None, min_samples=5, n_jobs=None, p=None)

```

```python
def plot_dbscan(dbscan, X, size, show_xlabels=True, show_ylabels=True):
    core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_mask[dbscan.core_sample_indices_] = True
    anomalies_mask = dbscan.labels_ == -1
    non_core_mask = ~(core_mask | anomalies_mask)

    cores = dbscan.components_
    anomalies = X[anomalies_mask]
    non_cores = X[non_core_mask]
    
    plt.scatter(cores[:, 0], cores[:, 1],
                c=dbscan.labels_[core_mask], marker='o', s=size, cmap="Paired")
    plt.scatter(cores[:, 0], cores[:, 1], marker='*', s=20, c=dbscan.labels_[core_mask])
    plt.scatter(anomalies[:, 0], anomalies[:, 1],
                c="r", marker="x", s=100)
    plt.scatter(non_cores[:, 0], non_cores[:, 1], c=dbscan.labels_[non_core_mask], marker=".")
    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom='off')
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft='off')
    plt.title("eps={:.2f}, min_samples={}".format(dbscan.eps, dbscan.min_samples), fontsize=14)

plt.figure(figsize=(9, 3.2))

plt.subplot(121)
plot_dbscan(dbscan, X, size=100)

plt.subplot(122)
plot_dbscan(dbscan2, X, size=600, show_ylabels=False)
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/10.13.png)

## 11. 决策树原理

分类和回归都可以解决

## 12. 决策树手写代码实现



## 13. 决策树算法实验分析

```pythono
import numpy as np
import os
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
import warnings
warnings.filterwarnings('ignore')
```

- 下载安装包：https://graphviz.gitlab.io/_pages/Download/Download_windows.html
- 环境变量配置：https://jingyan.baidu.com/article/020278115032461bcc9ce598.html

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data[:,2:] # petal length and width
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X,y)
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')
```

```python
from sklearn.tree import export_graphviz

export_graphviz(
    tree_clf,
    out_file="iris_tree.dot",
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)
```

然后，你可以使用graphviz包中的dot命令行工具将此**.dot**文件转换为各种格式，如PDF或PNG。下面这条命令行将.dot文件转换为.png图像文件：

$ dot -Tpng iris_tree.dot -o iris_tree.png

```python
from IPython.display import Image
Image(filename='iris_tree.png',width=400,height=400)
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/13.1.png)

### 决策边界展示

```python
from matplotlib.colors import ListedColormap

def plot_decision_boundary(clf, X, y, axes=[0, 7.5, 0, 3], iris=True, legend=False, plot_training=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if not iris:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    if plot_training:
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris-Setosa")
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris-Versicolor")
        plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="Iris-Virginica")
        plt.axis(axes)
    if iris:
        plt.xlabel("Petal length", fontsize=14)
        plt.ylabel("Petal width", fontsize=14)
    else:
        plt.xlabel(r"$x_1$", fontsize=18)
        plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    if legend:
        plt.legend(loc="lower right", fontsize=14)

plt.figure(figsize=(8, 4))
plot_decision_boundary(tree_clf, X, y)
plt.plot([2.45, 2.45], [0, 3], "k-", linewidth=2)
plt.plot([2.45, 7.5], [1.75, 1.75], "k--", linewidth=2)
plt.plot([4.95, 4.95], [0, 1.75], "k:", linewidth=2)
plt.plot([4.85, 4.85], [1.75, 3], "k:", linewidth=2)
plt.text(1.40, 1.0, "Depth=0", fontsize=15)
plt.text(3.2, 1.80, "Depth=1", fontsize=13)
plt.text(4.05, 0.5, "(Depth=2)", fontsize=11)
plt.title('Decision Tree decision boundaries')

plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/13.2.png)

### 概率估计

```
估计类概率 输入数据为：花瓣长5厘米，宽1.5厘米的花。 相应的叶节点是深度为2的左节点，因此决策树应输出以下概率：

Iris-Setosa 为 0％（0/54），
Iris-Versicolor 为 90.7％（49/54），
Iris-Virginica 为 9.3％（5/54）。
```

```python
tree_clf.predict_proba([[5,1.5]])
array([[0.        , 0.90740741, 0.09259259]])
tree_clf.predict([[5,1.5]])
array([1])
```

### 决策树中的正则化

```
DecisionTreeClassifier类还有一些其他参数类似地限制了决策树的形状：
min_samples_split（节点在分割之前必须具有的最小样本数），
min_samples_leaf（叶子节点必须具有的最小样本数），
max_leaf_nodes（叶子节点的最大数量），
max_features（在每个节点处评估用于拆分的最大特征数）。
max_depth(树最大的深度)
```

```python
from sklearn.datasets import make_moons
X,y = make_moons(n_samples=100,noise=0.25,random_state=53)
tree_clf1 = DecisionTreeClassifier(random_state=42)
tree_clf2 = DecisionTreeClassifier(min_samples_leaf=4,random_state=42)
tree_clf1.fit(X,y)
tree_clf2.fit(X,y)

plt.figure(figsize=(12,4))
plt.subplot(121)
plot_decision_boundary(tree_clf1,X,y,axes=[-1.5,2.5,-1,1.5],iris=False)
plt.title('No restrictions')

plt.subplot(122)
plot_decision_boundary(tree_clf2,X,y,axes=[-1.5,2.5,-1,1.5],iris=False)
plt.title('min_samples_leaf=4')
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/13.3.png)

### 对数据的敏感

```python
np.random.seed(6)
Xs = np.random.rand(100, 2) - 0.5
ys = (Xs[:, 0] > 0).astype(np.float32) * 2

angle = np.pi / 4
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
Xsr = Xs.dot(rotation_matrix)

tree_clf_s = DecisionTreeClassifier(random_state=42)
tree_clf_s.fit(Xs, ys)
tree_clf_sr = DecisionTreeClassifier(random_state=42)
tree_clf_sr.fit(Xsr, ys)

plt.figure(figsize=(11, 4))
plt.subplot(121)
plot_decision_boundary(tree_clf_s, Xs, ys, axes=[-0.7, 0.7, -0.7, 0.7], iris=False)
plt.title('Sensitivity to training set rotation')

plt.subplot(122)
plot_decision_boundary(tree_clf_sr, Xsr, ys, axes=[-0.7, 0.7, -0.7, 0.7], iris=False)
plt.title('Sensitivity to training set rotation')

plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/13.4.png)

### 回归任务

```python
np.random.seed(42)
m=200
X=np.random.rand(m,1)
y = 4*(X-0.5)**2
y = y + np.random.randn(m,1)/10

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X,y)

export_graphviz(
        tree_reg,
        out_file=("regression_tree.dot"),
        feature_names=["x1"],
        rounded=True,
        filled=True
    )
# 你的第二个决策树长这样
from IPython.display import Image
Image(filename="regression_tree.png",width=400,height=400,)
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/13.5.png)

```python
from sklearn.tree import DecisionTreeRegressor
v
tree_reg1 = DecisionTreeRegressor(random_state=42, max_depth=2)
tree_reg2 = DecisionTreeRegressor(random_state=42, max_depth=3)
tree_reg1.fit(X, y)
tree_reg2.fit(X, y)

def plot_regression_predictions(tree_reg, X, y, axes=[0, 1, -0.2, 1], ylabel="$y$"):
    x1 = np.linspace(axes[0], axes[1], 500).reshape(-1, 1)
    y_pred = tree_reg.predict(x1)
    plt.axis(axes)
    plt.xlabel("$x_1$", fontsize=18)
    if ylabel:
        plt.ylabel(ylabel, fontsize=18, rotation=0)
    plt.plot(X, y, "b.")
    plt.plot(x1, y_pred, "r.-", linewidth=2, label=r"$\hat{y}$")

plt.figure(figsize=(11, 4))
plt.subplot(121)

plot_regression_predictions(tree_reg1, X, y)
for split, style in ((0.1973, "k-"), (0.0917, "k--"), (0.7718, "k--")):
    plt.plot([split, split], [-0.2, 1], style, linewidth=2)
plt.text(0.21, 0.65, "Depth=0", fontsize=15)
plt.text(0.01, 0.2, "Depth=1", fontsize=13)
plt.text(0.65, 0.8, "Depth=1", fontsize=13)
plt.legend(loc="upper center", fontsize=18)
plt.title("max_depth=2", fontsize=14)

plt.subplot(122)

plot_regression_predictions(tree_reg2, X, y, ylabel=None)
for split, style in ((0.1973, "k-"), (0.0917, "k--"), (0.7718, "k--")):
    plt.plot([split, split], [-0.2, 1], style, linewidth=2)
for split in (0.0458, 0.1298, 0.2873, 0.9040):
    plt.plot([split, split], [-0.2, 1], "k:", linewidth=1)
plt.text(0.3, 0.5, "Depth=2", fontsize=13)
plt.title("max_depth=3", fontsize=14)

plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/13.6.png)

```python
tree_reg1 = DecisionTreeRegressor(random_state=42)
tree_reg2 = DecisionTreeRegressor(random_state=42, min_samples_leaf=10)
tree_reg1.fit(X, y)
tree_reg2.fit(X, y)

x1 = np.linspace(0, 1, 500).reshape(-1, 1)
y_pred1 = tree_reg1.predict(x1)
y_pred2 = tree_reg2.predict(x1)

plt.figure(figsize=(11, 4))

plt.subplot(121)
plt.plot(X, y, "b.")
plt.plot(x1, y_pred1, "r.-", linewidth=2, label=r"$\hat{y}$")
plt.axis([0, 1, -0.2, 1.1])
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", fontsize=18, rotation=0)
plt.legend(loc="upper center", fontsize=18)
plt.title("No restrictions", fontsize=14)

plt.subplot(122)
plt.plot(X, y, "b.")
plt.plot(x1, y_pred2, "r.-", linewidth=2, label=r"$\hat{y}$")
plt.axis([0, 1, -0.2, 1.1])
plt.xlabel("$x_1$", fontsize=18)
plt.title("min_samples_leaf={}".format(tree_reg2.min_samples_leaf), fontsize=14)

plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/13.7.png)

## 14. 集成算法原理

## 15. 集成算法实验分析

```python
import numpy as np
import os
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/15.1.png)

```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

X,y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
plt.plot(X[:,0][y==0],X[:,1][y==0],'yo',alpha = 0.6)
plt.plot(X[:,0][y==0],X[:,1][y==1],'bs',alpha = 0.6)
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/15.2.png)

### 软投票与硬投票

```
硬投票：直接用类别值，少数服从多数
软投票：各自分类器的概率值进行加权平均
```

```python
# 硬投票实验
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

log_clf = LogisticRegression(random_state=42)
rnd_clf = RandomForestClassifier(random_state=42)
svm_clf = SVC(random_state=42)

voting_clf = VotingClassifier(estimators =[('lr',log_clf),('rf',rnd_clf),('svc',svm_clf)],voting='hard')

voting_clf.fit(X_train,y_train)

# 牺牲一些时间点  提升准确率  但是提高的不高
from sklearn.metrics import accuracy_score
for clf in (log_clf,rnd_clf,svm_clf,voting_clf):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print (clf.__class__.__name__,accuracy_score(y_test,y_pred))
LogisticRegression 0.864
RandomForestClassifier 0.872
SVC 0.888
VotingClassifier 0.896
```

```python
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

log_clf = LogisticRegression(random_state=42)
rnd_clf = RandomForestClassifier(random_state=42)
# 要把概率值打开 
svm_clf = SVC(probability = True,random_state=42)

voting_clf = VotingClassifier(estimators =[('lr',log_clf),('rf',rnd_clf),('svc',svm_clf)],voting='soft')

voting_clf.fit(X_train,y_train)

# 软投票比硬投票要靠谱点
from sklearn.metrics import accuracy_score
for clf in (log_clf,rnd_clf,svm_clf,voting_clf):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print (clf.__class__.__name__,accuracy_score(y_test,y_pred))
LogisticRegression 0.864
RandomForestClassifier 0.872
SVC 0.888
VotingClassifier 0.912
```

### Bagging策略

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/15.3.png)

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(DecisionTreeClassifier(),
                  n_estimators = 500,
                  max_samples = 100,
                  bootstrap = True,
                  n_jobs = -1,
                  random_state = 42
)
bag_clf.fit(X_train,y_train)
y_pred = bag_clf.predict(X_test)


accuracy_score(y_test,y_pred)
0.904

tree_clf = DecisionTreeClassifier(random_state = 42)
tree_clf.fit(X_train,y_train)
y_pred_tree = tree_clf.predict(X_test)
accuracy_score(y_test,y_pred_tree)
0.856
```

### 决策边界

```python
# 集成与传统方法对比
from matplotlib.colors import ListedColormap
def plot_decision_boundary(clf,X,y,axes=[-1.5,2.5,-1,1.5],alpha=0.5,contour =True):
    x1s=np.linspace(axes[0],axes[1],100)
    x2s=np.linspace(axes[2],axes[3],100)
    x1,x2 = np.meshgrid(x1s,x2s)
    X_new = np.c_[x1.ravel(),x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1,x2,y_pred,cmap = custom_cmap,alpha=0.3)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1,x2,y_pred,cmap = custom_cmap2,alpha=0.8)
    plt.plot(X[:,0][y==0],X[:,1][y==0],'yo',alpha = 0.6)
    plt.plot(X[:,0][y==0],X[:,1][y==1],'bs',alpha = 0.6)
    plt.axis(axes)
    plt.xlabel('x1')
    plt.xlabel('x2')

plt.figure(figsize = (12,5))
plt.subplot(121)
plot_decision_boundary(tree_clf,X,y)
plt.title('Decision Tree') # 决策树的边界
plt.subplot(122)
plot_decision_boundary(bag_clf,X,y)
plt.title('Decision Tree With Bagging') # 决策树集成的模型 
# 越平稳的边界 越是我们想要的
# Colormap颜色：https://blog.csdn.net/zhaogeng111/article/details/78419015
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/15.4.png)

### OOB策略

```python
bag_clf = BaggingClassifier(DecisionTreeClassifier(),
                  n_estimators = 500,
                  max_samples = 100,
                  bootstrap = True,
                  n_jobs = -1,
                  random_state = 42,
                  oob_score = True
)
bag_clf.fit(X_train,y_train)
bag_clf.oob_score_
0.9253333333333333

y_pred = bag_clf.predict(X_test)
accuracy_score(y_test,y_pred)
0.904
bag_clf.oob_decision_function_
```

### 随机森林

```python
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train,y_train)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
```

```python
# 特征重要性： sklearn中是看每个特征的平均深度
from sklearn.datasets import load_iris
iris = load_iris()
rf_clf = RandomForestClassifier(n_estimators=500,n_jobs=-1)
rf_clf.fit(iris['data'],iris['target'])
for name,score in zip(iris['feature_names'],rf_clf.feature_importances_):
    print (name,score)
sepal length (cm) 0.11105536416721994
sepal width (cm) 0.02319505364393038
petal length (cm) 0.44036215067701534
petal width (cm) 0.42538743151183406
```

```python
# Mnist中哪些特征比较重要呢？
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
# 28*28的正方形  数字图像
rf_clf = RandomForestClassifier(n_estimators=500,n_jobs=-1)
rf_clf.fit(mnist['data'],mnist['target'])

rf_clf.feature_importances_.shape 
(784,)
def plot_digit(data):
    image = data.reshape(28,28)
    plt.imshow(image,cmap=matplotlib.cm.hot)
    plt.axis('off')

plot_digit(rf_clf.feature_importances_)
char = plt.colorbar(ticks=[rf_clf.feature_importances_.min(),rf_clf.feature_importances_.max()])
char.ax.set_yticklabels(['Not important','Very important'])
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/15.5.png)

### Boosting-提升策略 AdaBoost

```python
# 跟上学时的考试一样，这次做错的题，是不是得额外注意，下次的时候就和别错了！
# 以SVM分类器为例来演示AdaBoost的基本策略
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/15.6.png)

```python
from sklearn.svm import SVC

m = len(X_train)

plt.figure(figsize=(14,5))
for subplot,learning_rate in ((121,1),(122,0.5)):
    sample_weights = np.ones(m)
    plt.subplot(subplot)
    for i in range(5):
        svm_clf = SVC(kernel='rbf',C=0.05,random_state=42)
        svm_clf.fit(X_train,y_train,sample_weight = sample_weights)
        y_pred = svm_clf.predict(X_train)
        sample_weights[y_pred != y_train] *= (1+learning_rate)
        plot_decision_boundary(svm_clf,X,y,alpha=0.2)
        plt.title('learning_rate = {}'.format(learning_rate))
    if subplot == 121:
        plt.text(-0.7, -0.65, "1", fontsize=14)
        plt.text(-0.6, -0.10, "2", fontsize=14)
        plt.text(-0.5,  0.10, "3", fontsize=14)
        plt.text(-0.4,  0.55, "4", fontsize=14)
        plt.text(-0.3,  0.90, "5", fontsize=14)
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/15.7.png)

```python
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                   n_estimators = 200,
                   learning_rate = 0.5,
                   random_state = 42
)
ada_clf.fit(X_train,y_train)
plot_decision_boundary(ada_clf,X,y)
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/15.8.png)

### Gradient Boosting

```python
# 梯度提升决策树
np.random.seed(42)
X = np.random.rand(100,1) - 0.5
y = 3*X[:,0]**2 + 0.05*np.random.randn(100)

y.shape
(100,)
from sklearn.tree import DecisionTreeRegressor
tree_reg1 = DecisionTreeRegressor(max_depth = 2)
tree_reg1.fit(X,y)

y2 = y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth = 2)
tree_reg2.fit(X,y2)

y3 = y2 - tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max_depth = 2)
tree_reg3.fit(X,y3)


X_new = np.array([[0.8]])
y_pred = sum(tree.predict(X_new) for tree in (tree_reg1,tree_reg2,tree_reg3))
y_pred
array([0.75026781])
```

```python
def plot_predictions(regressors, X, y, axes, label=None, style="r-", data_style="b.", data_label=None):
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pred = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in regressors)
    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pred, style, linewidth=2, label=label)
    if label or data_label:
        plt.legend(loc="upper center", fontsize=16)
    plt.axis(axes)

plt.figure(figsize=(11,11))

plt.subplot(321)
plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h_1(x_1)$", style="g-", data_label="Training set")
plt.ylabel("$y$", fontsize=16, rotation=0)
plt.title("Residuals and tree predictions", fontsize=16)

plt.subplot(322)
plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1)$", data_label="Training set")
plt.ylabel("$y$", fontsize=16, rotation=0)
plt.title("Ensemble predictions", fontsize=16)

plt.subplot(323)
plot_predictions([tree_reg2], X, y2, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_2(x_1)$", style="g-", data_style="k+", data_label="Residuals")
plt.ylabel("$y - h_1(x_1)$", fontsize=16)

plt.subplot(324)
plot_predictions([tree_reg1, tree_reg2], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1)$")
plt.ylabel("$y$", fontsize=16, rotation=0)

plt.subplot(325)
plot_predictions([tree_reg3], X, y3, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_3(x_1)$", style="g-", data_style="k+")
plt.ylabel("$y - h_1(x_1) - h_2(x_1)$", fontsize=16)
plt.xlabel("$x_1$", fontsize=16)

plt.subplot(326)
plot_predictions([tree_reg1, tree_reg2, tree_reg3], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1) + h_3(x_1)$")
plt.xlabel("$x_1$", fontsize=16)
plt.ylabel("$y$", fontsize=16, rotation=0)

plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/15.9.png)

```python
from sklearn.ensemble import GradientBoostingRegressor
gbrt = GradientBoostingRegressor(max_depth = 2,
                          n_estimators = 3,
                          learning_rate = 1.0,
                          random_state = 41
)
gbrt.fit(X,y)

gbrt_slow_1 = GradientBoostingRegressor(max_depth = 2,
                          n_estimators = 3,
                          learning_rate = 0.1,
                          random_state = 41
)
gbrt_slow_1.fit(X,y) 

gbrt_slow_2 = GradientBoostingRegressor(max_depth = 2,
                          n_estimators = 200,
                          learning_rate = 0.1,
                          random_state = 41
)
gbrt_slow_2.fit(X,y)

plt.figure(figsize = (11,4))
plt.subplot(121)
plot_predictions([gbrt],X,y,axes=[-0.5,0.5,-0.1,0.8],label = 'Ensemble predictions')
plt.title('learning_rate={},n_estimators={}'.format(gbrt.learning_rate,gbrt.n_estimators))

plt.subplot(122)
plot_predictions([gbrt_slow_1],X,y,axes=[-0.5,0.5,-0.1,0.8],label = 'Ensemble predictions')
plt.title('learning_rate={},n_estimators={}'.format(gbrt_slow_1.learning_rate,gbrt_slow_1.n_estimators))
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/15.10.png)

```python
plt.figure(figsize = (11,4))
plt.subplot(121)
plot_predictions([gbrt_slow_2],X,y,axes=[-0.5,0.5,-0.1,0.8],label = 'Ensemble predictions')
plt.title('learning_rate={},n_estimators={}'.format(gbrt_slow_2.learning_rate,gbrt_slow_2.n_estimators))

plt.subplot(122)
plot_predictions([gbrt_slow_1],X,y,axes=[-0.5,0.5,-0.1,0.8],label = 'Ensemble predictions')
plt.title('learning_rate={},n_estimators={}'.format(gbrt_slow_1.learning_rate,gbrt_slow_1.n_estimators))
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/15.11.png)

### 提前停止策略

```python
from sklearn.metrics import mean_squared_error
X_train,X_val,y_train,y_val = train_test_split(X,y,random_state=49)
gbrt = GradientBoostingRegressor(max_depth = 2,
                          n_estimators = 120,
                          random_state = 42
)
gbrt.fit(X_train,y_train)

errors = [mean_squared_error(y_val,y_pred) for y_pred in gbrt.staged_predict(X_val)]
bst_n_estimators = np.argmin(errors)

gbrt_best = GradientBoostingRegressor(max_depth = 2,
                          n_estimators = bst_n_estimators,
                          random_state = 42
)
gbrt_best.fit(X_train,y_train)
```

```python
min_error = np.min(errors)
min_error
0.002712853325235463

plt.figure(figsize = (11,4))
plt.subplot(121)
plt.plot(errors,'b.-')
plt.plot([bst_n_estimators,bst_n_estimators],[0,min_error],'k--')
plt.plot([0,120],[min_error,min_error],'k--')
plt.axis([0,120,0,0.01])
plt.title('Val Error')

plt.subplot(122)
plot_predictions([gbrt_best],X,y,axes=[-0.5,0.5,-0.1,0.8])
plt.title('Best Model(%d trees)'%bst_n_estimators)
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/15.12.png)

```python
gbrt = GradientBoostingRegressor(max_depth = 2,
                             random_state = 42,
                                 warm_start =True
)
error_going_up = 0
min_val_error = float('inf')

for n_estimators in range(1,120):
    gbrt.n_estimators = n_estimators
    gbrt.fit(X_train,y_train)
    y_pred = gbrt.predict(X_val)
    val_error = mean_squared_error(y_val,y_pred)
    if val_error < min_val_error:
        min_val_error = val_error
        error_going_up = 0
    else:
        error_going_up +=1
        if error_going_up == 5:
            break

print (gbrt.n_estimators)
61
```

### Stacking（堆叠集成）

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/15.13.png)

```python
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

mnist = fetch_mldata('MNIST original')

X_train_val, X_test, y_train_val, y_test = train_test_split(
    mnist.data, mnist.target, test_size=10000, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=10000, random_state=42)
```

```python
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
random_forest_clf = RandomForestClassifier(random_state=42)
extra_trees_clf = ExtraTreesClassifier(random_state=42)
svm_clf = LinearSVC(random_state=42)
mlp_clf = MLPClassifier(random_state=42)

estimators = [random_forest_clf, extra_trees_clf, svm_clf, mlp_clf]
for estimator in estimators:
    print("Training the", estimator)
    estimator.fit(X_train, y_train)
 
X_val_predictions = np.empty((len(X_val), len(estimators)), dtype=np.float32)

for index, estimator in enumerate(estimators):
    X_val_predictions[:, index] = estimator.predict(X_val)
    
X_val_predictions

rnd_forest_blender = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42)
rnd_forest_blender.fit(X_val_predictions, y_val)
rnd_forest_blender.oob_score_
0.9642
```



## 16. 支持向量机原理



## 17. 支持向量机实验分析

```python
# 与传统算法进行对比，看看SVM究竟能带来什么样的效果
# 软间隔的作用，这么复杂的算法肯定会导致过拟合现象，如何来进行解决呢？
# 核函数的作用，如果只是做线性分类，好像轮不到SVM登场了，核函数才是它的强大之处！
import numpy as np
import os
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
import warnings
warnings.filterwarnings('ignore')
```

### 支持向量机带来的效果

```python
from sklearn.svm import SVC
from sklearn import datasets

iris = datasets.load_iris()
X = iris['data'][:,(2,3)]
y = iris['target']

setosa_or_versicolor = (y==0)|(y==1)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]

svm_clf = SVC(kernel='linear',C=float('inf'))
svm_clf.fit(X,y)
SVC(C=inf, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
  kernel='linear', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False)

```

```python
# 一般的模型
x0 = np.linspace(0, 5.5, 200)
pred_1 = 5*x0 - 20
pred_2 = x0 - 1.8
pred_3 = 0.1 * x0 + 0.5

def plot_svc_decision_boundary(svm_clf, xmin, xmax,sv=True):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]
    print (w)
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = - w[0]/w[1] * x0 - b/w[1]
    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin
    if sv:
        svs = svm_clf.support_vectors_
        plt.scatter(svs[:,0],svs[:,1],s=180,facecolors='#FFAAAA')
    plt.plot(x0,decision_boundary,'k-',linewidth=2)
    plt.plot(x0,gutter_up,'k--',linewidth=2)
    plt.plot(x0,gutter_down,'k--',linewidth=2)
plt.figure(figsize=(14,4))
plt.subplot(121)
plt.plot(X[:,0][y==1],X[:,1][y==1],'bs')
plt.plot(X[:,0][y==0],X[:,1][y==0],'ys')
plt.plot(x0,pred_1,'g--',linewidth=2)
plt.plot(x0,pred_2,'m-',linewidth=2)
plt.plot(x0,pred_3,'r-',linewidth=2)
plt.axis([0,5.5,0,2])

plt.subplot(122)
plot_svc_decision_boundary(svm_clf, 0, 5.5)
plt.plot(X[:,0][y==1],X[:,1][y==1],'bs')
plt.plot(X[:,0][y==0],X[:,1][y==0],'ys')
plt.axis([0,5.5,0,2])
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/15.14.png)

### 数据标准化的影响

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/15.15.png)



## 18. 神经网络原理



## 19. 神经网络代码实现



## 20. 贝叶斯算法原理



## 21. 贝叶斯代码实现



## 22. 关联规则实战分析

```
关联规则
在美国，一些年轻的父亲下班后经常要到超市去买婴儿尿布，超市也因此发现了一个规律，在购买婴儿尿布的年轻父亲们中，有30%～40%的人同时要买一些啤酒。超市随后调整了货架的摆放，把尿布和啤酒放在一起，明显增加了销售额。

若两个或多个变量的取值之间存在某种规律性，就称为关联
关联规则是寻找在同一个事件中出现的不同项的相关性，比如在一次购买活动中所买不同商品的相关性。
“在购买计算机的顾客中，有30％的人也同时购买了打印机”

一个样本称为一个“事务”
每个事务由多个属性来确定，这里的属性称为“项”
多个项组成的集合称为“项集”

由k个项构成的集合
{牛奶}、{啤酒}都是1-项集；
{牛奶，果冻}是2-项集；
{啤酒，面包，牛奶}是3-项集
X==>Y含义：
X和Y是项集
X称为规则前项（antecedent）
Y称为规则后项（consequent）
```

```
事务仅包含其涉及到的项目，而不包含项目的具体信息。
在超级市场的关联规则挖掘问题中事务是顾客一次购物所购买的商品，但事务中并不包含这些商品的具体信息，如商品的数量、价格等。

支持度（support）：一个项集或者规则在所有事务中出现的频率，σ(X):表示项集X的支持度计数
项集X的支持度：s(X)=σ(X)/N
规则X==>Y表示物品集X对物品集Y的支持度，也就是物品集X和物品集Y同时出现的概率
某天共有100个顾客到商场购买物品，其中有30个顾客同时购买了啤酒和尿布，那么上述的关联规则的支持度就是30％

置信度（confidence）：确定Y在包含X的事务中出现的频繁程度。c(X → Y) = σ(X∪Y)/σ(X)
p（Y│X）＝p（XY）/p(X)。
置信度反应了关联规则的可信度—购买了项目集X中的商品的顾客同时也购买了Y中商品的可能性有多大
购买薯片的顾客中有50％的人购买了可乐,则置信度为50％
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/22.1.png)

```
设最小支持度为50%, 最小可信度为 50%, 则可得到 :
A==>C (50%, 66.6%)
C==>A (50%, 100%)
若关联规则X->Y的支持度和置信度分别大于或等于用户指定的最小支持率minsupport和最小置信度minconfidence，则称关联规则X->Y为强关联规则，否则称关联规则X->Y为弱关联规则。

提升度（lift）：物品集A的出现对物品集B的出现概率发生了多大的变化
lift（A==>B）=confidence（A==>B）/support(B)=p(B|A)/p(B)
现在有** 1000 ** 个消费者，有** 500** 人购买了茶叶，其中有** 450人同时** 购买了咖啡，另** 50人** 没有。由于** confidence(茶叶=>咖啡)=450/500=90%** ，由此可能会认为喜欢喝茶的人往往喜欢喝咖啡。但如果另外没有购买茶叶的** 500人** ，其中同样有** 450人** 购买了咖啡，同样是很高的** 置信度90%** ,由此，得到不爱喝茶的也爱喝咖啡。这样看来，其实是否购买咖啡，与有没有购买茶叶并没有关联，两者是相互独立的，其** 提升度90%/[(450+450)/1000]=1** 。
由此可见，lift正是弥补了confidence的这一缺陷，if lift=1,X与Y独立，X对Y出现的可能性没有提升作用，其值越大(lift>1),则表明X对Y的提升程度越大，也表明关联性越强。

Leverage 与 Conviction的作用和lift类似，都是值越大代表越关联
Leverage :P(A,B)-P(A)P(B)
Conviction:P(A)P(!B)/P(A,!B)
```

 ### 使用mlxtend工具包得出频繁项集与规则

```python
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

data = {'ID':[1,2,3,4,5,6],
       'Onion':[1,0,0,1,1,1],
       'Potato':[1,1,0,1,1,1],
       'Burger':[1,1,0,0,1,1],
       'Milk':[0,1,1,1,0,1],
       'Beer':[0,0,1,0,1,0]}
df = pd.DataFrame(data)
df = df[['ID', 'Onion', 'Potato', 'Burger', 'Milk', 'Beer' ]]

# 选择最小支持度为50%
# apriori(df, min_support=0.5, use_colnames=True)
frequent_itemsets = apriori(df[['Onion', 'Potato', 'Burger', 'Milk', 'Beer' ]], min_support=0.50, use_colnames=True)
frequent_itemsets
support	itemsets
0	0.666667	(Onion)
1	0.833333	(Potato)
2	0.666667	(Burger)
3	0.666667	(Milk)
4	0.666667	(Potato, Onion)
```

### 计算规则

```python
# association_rules(df, metric='lift', min_threshold=1)
# 可以指定不同的衡量标准与最小阈值
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)
rules
antecedents	consequents	antecedent support	consequent support	support	confidence	lift	leverage	conviction
0	(Potato)	(Onion)	0.833333	0.666667	0.666667	0.80	1.200	0.111111	1.666667
1	(Onion)	(Potato)	0.666667	0.833333	0.666667	1.00	1.200	0.111111	inf
2	(Burger)	(Onion)	0.666667	0.666667	0.500000	0.75	1.125	0.055556	1.333333

# 返回的是各个的指标的数值，可以按照感兴趣的指标排序观察,但具体解释还得参考实际数据的含义。
rules [ (rules['lift'] >1.125)  & (rules['confidence']> 0.8)  ]

antecedents	consequents	antecedent support	consequent support	support	confidence	lift	leverage	conviction
1	(Onion)	(Potato)	0.666667	0.833333	0.666667	1.0	1.2	0.111111	inf
4	(Burger)	(Potato)	0.666667	0.833333	0.666667	1.0	1.2	0.111111	inf
7	(Burger, Onion)	(Potato)	0.500000	0.833333	0.500000	1.0	1.2	0.083333	inf

# 这几条结果就比较有价值了：
#（洋葱和马铃薯）（汉堡和马铃薯）可以搭配着来卖
# 如果洋葱和汉堡都在购物篮中, 顾客买马铃薯的可能性也比较高，如果他篮子里面没有，可以推荐一下.
```

### 所有指标的计算公式

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/22.2.png)

### 数据需转换成one-hot编码

```python
retail_shopping_basket = {'ID':[1,2,3,4,5,6],
                         'Basket':[['Beer', 'Diaper', 'Pretzels', 'Chips', 'Aspirin'],
                                   ['Diaper', 'Beer', 'Chips', 'Lotion', 'Juice', 'BabyFood', 'Milk'],
                                   ['Soda', 'Chips', 'Milk'],
                                   ['Soup', 'Beer', 'Diaper', 'Milk', 'IceCream'],
                                   ['Soda', 'Coffee', 'Milk', 'Bread'],
                                   ['Beer', 'Chips']
                                  ]
                         }
retail = pd.DataFrame(retail_shopping_basket)
retail = retail[['ID', 'Basket']]
pd.options.display.max_colwidth=100

retail
ID	Basket
0	1	[Beer, Diaper, Pretzels, Chips, Aspirin]

retail_id = retail.drop('Basket' ,1)
retail_id
retail_Basket = retail.Basket.str.join(',')
retail_Basket

retail_Basket = retail_Basket.str.get_dummies(',')
retail_Basket
retail_Basket = retail_Basket.str.get_dummies(',')
retail_Basket
Aspirin	BabyFood	Beer	Bread	Chips	Coffee	Diaper	IceCream	Juice	Lotion	Milk	Pretzels	Soda	Soup
0	1	0	1	0	1	0	1	0	0	0	0	1	0	0

retail = retail_id.join(retail_Basket)
frequent_itemsets_2 = apriori(retail.drop('ID',1), use_colnames=True)
# 如果光考虑支持度support(X>Y), [Beer, Chips] 和 [Beer, Diaper] 都是很频繁的，哪一种组合更相关呢？
association_rules(frequent_itemsets_2, metric='lift')


```

### 电影相关联

```python
movies = pd.read_csv('ml-latest-small/movies.csv')
movies.head(10)
movieId	     title	              genres
0	1	Toy Story (1995)	Adventure|Animation|Children|Comedy|Fantasy
1	2	Jumanji (1995)	Adventure|Children|Fantasy

movies_ohe = movies.drop('genres',1).join(movies.genres.str.get_dummies())
pd.options.display.max_columns=100
movies_ohe.head()

# 数据集包括9125部电影，一共有20种不同类型。
movies_ohe.shape
(9125, 22)

movies_ohe.set_index(['movieId','title'],inplace=True)
frequent_itemsets_movies = apriori(movies_ohe,use_colnames=True, min_support=0.025)
rules_movies =  association_rules(frequent_itemsets_movies, metric='lift', min_threshold=1.25)
rules_movies[(rules_movies.lift>4)].sort_values(by=['lift'], ascending=False)
# Children和Animation 这俩题材是最相关的了，常识也可以分辨出来。
movies[(movies.genres.str.contains('Children')) & (~movies.genres.str.contains('Animation'))]

```



## 23. 关联规则代码实现

## 24. 词向量word2vec

## 25. 代码实现词向量word2vec

## 26. 线性判别分析降维 LDA

```python
# 适用于有监督 分类 的数据初始降维
# 必须是有监督问题
feature_dict = {i:label for i,label in zip(
                range(4),
                  ('sepal length in cm',
                  'sepal width in cm',
                  'petal length in cm',
                  'petal width in cm', ))}

import pandas as pd

df = pd.io.parsers.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None,
    sep=',',
    )
df.columns = [l for i,l in sorted(feature_dict.items())] + ['class label']
df.dropna(how="all", inplace=True) # to drop the empty line at file-end

df.tail()
sepal length in cm	sepal width in cm	petal length in cm	petal width in cm	class label
145	6.7	3.0	5.2	2.3	Iris-virginica
146	6.3	2.5	5.0	1.9	Iris-virginica

from sklearn.preprocessing import LabelEncoder

X = df[['sepal length in cm','sepal width in cm','petal length in cm','petal width in cm']].values
y = df['class label'].values

# 把标签变成数字快捷写法
enc = LabelEncoder()
label_encoder = enc.fit(y)
y = label_encoder.transform(y) + 1

#label_dict = {1: 'Setosa', 2: 'Versicolor', 3:'Virginica'}
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/26.1.png)

```python
import numpy as np
np.set_printoptions(precision=4)

mean_vectors = []
for cl in range(1,4):
    mean_vectors.append(np.mean(X[y==cl], axis=0))
    print('Mean Vector class %s: %s\n' %(cl, mean_vectors[cl-1]))
    

Mean Vector class 1: [ 5.006  3.418  1.464  0.244]
Mean Vector class 2: [ 5.936  2.77   4.26   1.326]
Mean Vector class 3: [ 6.588  2.974  5.552  2.026]
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/26.2.png)

```python
S_W = np.zeros((4,4))
for cl,mv in zip(range(1,4), mean_vectors):
    class_sc_mat = np.zeros((4,4))                  # scatter matrix for every class
    for row in X[y == cl]:
        row, mv = row.reshape(4,1), mv.reshape(4,1) # make column vectors
        class_sc_mat += (row-mv).dot((row-mv).T)
    S_W += class_sc_mat                             # sum class scatter matrices
print('within-class Scatter Matrix:\n', S_W)
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/26.3.png)

```python
overall_mean = np.mean(X, axis=0)

S_B = np.zeros((4,4))
for i,mean_vec in enumerate(mean_vectors):  
    n = X[y==i+1,:].shape[0]
    mean_vec = mean_vec.reshape(4,1) # make column vector
    overall_mean = overall_mean.reshape(4,1) # make column vector
    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

print('between-class Scatter Matrix:\n', S_B)
```

```python
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
# 算每个特征的特征值和特征向量
for i in range(len(eig_vals)):
    eigvec_sc = eig_vecs[:,i].reshape(4,1)   
    print('\nEigenvector {}: \n{}'.format(i+1, eigvec_sc.real))
    print('Eigenvalue {:}: {:.2e}'.format(i+1, eig_vals[i].real))
```

```python
#特征向量：表示映射方向
#特征值：特征向量的重要程度
#Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in decreasing order:\n')
for i in eig_pairs:
    print(i[0])
Eigenvalues in decreasing order:
32.2719577997
0.27756686384
7.7995841654e-15
5.76433252705e-15

print('Variance explained:\n')
eigv_sum = sum(eig_vals)
for i,j in enumerate(eig_pairs):
    print('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))
Variance explained:

eigenvalue 1: 99.15%
eigenvalue 2: 0.85%
eigenvalue 3: 0.00%
eigenvalue 4: 0.00%
```

```python
# 选择前两维特征
W = np.hstack((eig_pairs[0][1].reshape(4,1), eig_pairs[1][1].reshape(4,1)))
print('Matrix W:\n', W.real)
Matrix W:
 [[ 0.2049 -0.009 ]
 [ 0.3871 -0.589 ]
 [-0.5465  0.2543]
 [-0.7138 -0.767 ]]
X_lda = X.dot(W)
assert X_lda.shape == (150,2), "The matrix is not 150x2 dimensional."
```

```python
from matplotlib import pyplot as plt

def plot_step_lda():

    ax = plt.subplot(111)
    for label,marker,color in zip(
        range(1,4),('^', 's', 'o'),('blue', 'red', 'green')):

        plt.scatter(x=X_lda[:,0].real[y == label],
                y=X_lda[:,1].real[y == label],
                marker=marker,
                color=color,
                alpha=0.5,
                label=label_dict[label]
                )

    plt.xlabel('LD1')
    plt.ylabel('LD2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title('LDA: Iris projection onto the first 2 linear discriminants')

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)    

    plt.grid()
    plt.tight_layout
    plt.show()

plot_step_lda()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/26.4.png)

### 官方的包

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# LDA
sklearn_lda = LDA(n_components=2)
X_lda_sklearn = sklearn_lda.fit_transform(X, y)

def plot_scikit_lda(X, title):

    ax = plt.subplot(111)
    for label,marker,color in zip(
        range(1,4),('^', 's', 'o'),('blue', 'red', 'green')):

        plt.scatter(x=X[:,0][y == label],
                    y=X[:,1][y == label] * -1, # flip the figure
                    marker=marker,
                    color=color,
                    alpha=0.5,
                    label=label_dict[label])

    plt.xlabel('LD1')
    plt.ylabel('LD2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title(title)

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)    

    plt.grid()
    plt.tight_layout
    plt.show()
    
plot_step_lda()
plot_scikit_lda(X_lda_sklearn, title='Default LDA via scikit-learn')
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/26.5.png)

## 27.主成分分析降维 PCA

```python
# 无监督 的  可以降维
import numpy as np
import pandas as pd
df = pd.read_csv('iris.data')
df.head()
df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.head()
sepal_len	sepal_wid	petal_len	petal_wid	class
0	4.9	3.0	1.4	0.2	Iris-setosa
1	4.7	3.2	1.3	0.2	Iris-setosa
2	4.6	3.1	1.5	0.2	Iris-setos
```

```python
# split data table into data X and class labels y
X = df.ix[:,0:4].values
y = df.ix[:,4].values

from matplotlib import pyplot as plt
import math

label_dict = {1: 'Iris-Setosa',
              2: 'Iris-Versicolor',
              3: 'Iris-Virgnica'}

feature_dict = {0: 'sepal length [cm]',
                1: 'sepal width [cm]',
                2: 'petal length [cm]',
                3: 'petal width [cm]'}


plt.figure(figsize=(8, 6))
for cnt in range(4):
    plt.subplot(2, 2, cnt+1)
    for lab in ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'):
        plt.hist(X[y==lab, cnt],
                     label=lab,
                     bins=10,
                     alpha=0.3,)
    plt.xlabel(feature_dict[cnt])
    plt.legend(loc='upper right', fancybox=True, fontsize=8)

plt.tight_layout()
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/26.6.png)

```python
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)
print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))

cov_mat = np.cov(X_std.T)
# 来计算给定矩阵cov_mat的特征值和特征向量
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)
Eigenvectors 
[[ 0.52308496 -0.36956962 -0.72154279  0.26301409]
 [-0.25956935 -0.92681168  0.2411952  -0.12437342]
 [ 0.58184289 -0.01912775  0.13962963 -0.80099722]
 [ 0.56609604 -0.06381646  0.63380158  0.52321917]]

Eigenvalues 
[ 2.92442837  0.93215233  0.14946373  0.02098259]

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
print (eig_pairs)
print ('----------')
# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)
# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])
    
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
print (var_exp)
# 计算输入数组的累积和
cum_var_exp = np.cumsum(var_exp)
cum_var_exp
[72.620033326920336, 23.147406858644135, 3.7115155645845164, 0.52104424985101538]
array([  72.62003333,   95.76744019,   99.47895575,  100.        ])

plt.figure(figsize=(6, 4))

plt.bar(range(4), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
plt.step(range(4), cum_var_exp, where='mid',
             label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/26.7.png)

```python
matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1),
                      eig_pairs[1][1].reshape(4,1)))

print('Matrix W:\n', matrix_w)

Y = X_std.dot(matrix_w)  # 降维后的数据
array([[-2.10795032,  0.64427554],
       [-2.38797131,  0.30583307],
```

```python
plt.figure(figsize=(6, 4))
for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                        ('blue', 'red', 'green')):
     plt.scatter(X[y==lab, 0],
                X[y==lab, 1],
                label=lab,
                c=col)
plt.xlabel('sepal_len')
plt.ylabel('sepal_wid')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/26.8.png)

```python
plt.figure(figsize=(6, 4))
for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                        ('blue', 'red', 'green')):
     plt.scatter(Y[y==lab, 0],
                Y[y==lab, 1],
                label=lab,
                c=col)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='lower center')
plt.tight_layout()
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/ML/26.9.png)

## 28. HMM

