# 数学基础

## 1. 高等数学基础

### 1.函数

![](http://img.clint-sfy.cn/python/math/高等数学1.png)

### 2. 极限

![](http://img.clint-sfy.cn/python/math/高等数学2.png)

![](http://img.clint-sfy.cn/python/math/高等数学3.png)

### 3. 无穷小与无穷大

![](http://img.clint-sfy.cn/python/math/高等数学4.png)

![](http://img.clint-sfy.cn/python/math/高等数学5.png)

### 4. 连续性和导数

![](http://img.clint-sfy.cn/python/math/高等数学6.png)

### 5. 偏导数

![](http://img.clint-sfy.cn/python/math/高等数学7.png)

### 6. 方向导数

![](http://img.clint-sfy.cn/python/math/高等数学8.png)

![](http://img.clint-sfy.cn/python/math/高等数学9.png)

### 7. 梯度

![](http://img.clint-sfy.cn/python/math/高等数学10.png)

![](http://img.clint-sfy.cn/python/math/高等数学11.png)

![](http://img.clint-sfy.cn/python/math/高等数学12.png)

## 2. 微积分

### 1. 微积分解释

微积分诞生于17世纪，主要帮助人们解决各种速度，面积等实际问题  

![](http://img.clint-sfy.cn/python/math/微积分1.png)

### 2. 定积分

![](http://img.clint-sfy.cn/python/math/微积分2.png)

### 3. 定积分性质

![](http://img.clint-sfy.cn/python/math/微积分3.png)

![](http://img.clint-sfy.cn/python/math/微积分4.png)

### 4. 牛顿莱布尼茨

![](http://img.clint-sfy.cn/python/math/微积分5.png)

## 3. 泰勒公式和拉格朗日

### 1. 泰勒公式

用简单的熟悉的多项式来近似代替复杂的函数
易计算函数值，导数与积分仍是多项式
多项式由它的系数完全确定，其系数又由它在一点的函数值及其导数所确定。  

![](http://img.clint-sfy.cn/python/math/泰勒1.png)

### 2. 阶数和阶乘

![](http://img.clint-sfy.cn/python/math/泰勒2.png)

如果把9次的和2次的直接放在一起，那2次的就不用玩了。
但是在开始的时候应该是2次的效果更好，之后才是慢慢轮到9次的呀！  

![](http://img.clint-sfy.cn/python/math/泰勒3.png)

### 3. 拉格朗日乘子法

![](http://img.clint-sfy.cn/python/math/泰勒4.png)

### 4. 求解拉格朗日

![](http://img.clint-sfy.cn/python/math/泰勒5.png)

## 4. 线性代数基础

### 1. 行列式概述

![](http://img.clint-sfy.cn/python/math/线代1.png)

### 2. 矩阵与数据的关系

![](http://img.clint-sfy.cn/python/math/线代2.png)

### 3. 矩阵的基本操作



### 4. 矩阵的几种变换



### 5. 矩阵的秩



### 6. 内积和正交

![](http://img.clint-sfy.cn/python/math/线代3.png)

![](http://img.clint-sfy.cn/python/math/线代4.png)

## 5. 特征值与矩阵分解

### 1. 特征值和特征向量

![](http://img.clint-sfy.cn/python/math/特征1.png)

既然特征值表达了重要程度且和特征向量所对应，那么特征值大的就是
主要信息了，基于这点我们可以提取各种有价值的信息了！  

### 2. 特征空间和应用



### 3. SVD解决的问题



### 4. 特征值分解

特征值分解
矩阵里面的信息有很多呀？来分一分吧！
当矩阵是N*N的方阵且有N个线性无关的特征向量时就可以来玩啦！
这时候我们就可以在对角阵当中找比较大的啦，他们就是代表了  

### 5.SVD矩阵分解

前提：对于一个二维矩阵M可以找到一组标准正交基v1和v2使得Mv1和Mv2是正交的。  

![](http://img.clint-sfy.cn/python/math/特征2.png)

## 6. 随机变量

### 1. 离散型随机变量

概率函数（概率质量函数）
专为离散型随机变量定义的：
本身就是一个概率值，X是随机变量的取值，P就是概率了。
比如我们来投掷骰子  

![](http://img.clint-sfy.cn/python/math/随机变量1.png)

### 2. 连续型随机变量

概率密度：对于连续型随机变量X，我们不能给出其取每一个值的概率也就是画不出那个分布表，这里我们选择使用密度来表示其概率分布！  

![](http://img.clint-sfy.cn/python/math/随机变量2.png)

### 3. 简单随机抽样

![](http://img.clint-sfy.cn/python/math/随机变量3.png)

### 4. 似然函数

![](http://img.clint-sfy.cn/python/math/似然函数1.png)

![](http://img.clint-sfy.cn/python/math/似然函数2.png)

### 5. 极大似然估计

在一次吃鸡比赛中，有两位选手，一个是职业选手，一个是菜鸟路人。
比赛结束后，公布结果有一位选手完成20杀，请问是哪个选手呢？
估计大家都选的是职业选手吧！
因为我们会普遍认为概率最大的事件最有可能发生！
极大似然估计：在一次抽样中，得到观测值x1,x2...xn。
选取θ'(x1,x2...xn)作为θ的估计值，使得θ=θ'(x1,x2...xn)时样本出现的概率最大  

![](http://img.clint-sfy.cn/python/math/似然函数3.png)



## 7. 概率论基础

### 1. 频率与概率 

研究随机现象数量规矩的数学分支  

![](http://img.clint-sfy.cn/python/math/概率论1.png)

### 2. 古典概型

![](http://img.clint-sfy.cn/python/math/概率论2.png)

### 3. 条件概率

![](http://img.clint-sfy.cn/python/math/概率论3.png)

![](http://img.clint-sfy.cn/python/math/概率论4.png)

### 4. 独立性

![](http://img.clint-sfy.cn/python/math/概率论5.png)



### 5. 二维离散型随机变量

![](http://img.clint-sfy.cn/python/math/概率论6.png)

### 6. 二维连续型随机变量

![](http://img.clint-sfy.cn/python/math/概率论7.png)

### 7. 边缘分布

![](http://img.clint-sfy.cn/python/math/概率论8.png)

### 8. 期望

![](http://img.clint-sfy.cn/python/math/概率论9.png)

![](http://img.clint-sfy.cn/python/math/概率论10.png)

![](http://img.clint-sfy.cn/python/math/概率论11.png)

数学期望反映了随机变量的取值水平，衡量随机变量相对于数学期望
的分散程度则的另一个数字特征。
X为随机变量，如果 存在，则称其为X的方差，记作D(X)  

### 9. 马尔科夫不等式

![](http://img.clint-sfy.cn/python/math/概率论12.png)

### 10. 切比雪夫不等式

![](http://img.clint-sfy.cn/python/math/概率论13.png)

### 11. 后验概率估计

![](http://img.clint-sfy.cn/python/math/后验概率1.png)

![](http://img.clint-sfy.cn/python/math/后验概率2.png)

### 12. 案例：贝叶斯拼写纠错



### 13 案例：垃圾邮件过滤



## 8. 几种分布

### 1. 正太分布

​      正态分布代表了宇宙中大多数情况的运转状态。大量的随机变量被证明是正态分布的。

​      若随机变量X服从一个数学期望为μ、方差为σ^2的正态分布，记为N(μ，σ^2)。其概率密度函数为正态分布的期望值μ决定了其位置，其标准差σ决定了分布的幅度。当μ = 0,σ = 1时的正态分布是标准正态分布。

![](http://img.clint-sfy.cn/python/math/分布1.png)

![](http://img.clint-sfy.cn/python/math/分布2.png)

![](http://img.clint-sfy.cn/python/math/分布3.png)

```python
plt.figure(dpi=100)

##### COMPUTATION #####
# DECLARING THE "TRUE" PARAMETERS UNDERLYING THE SAMPLE
mu_real = 10
sigma_real = 2

# DRAW A SAMPLE OF N=1000
np.random.seed(42)
sample = stats.norm.rvs(loc=mu_real, scale=sigma_real, size=1000)

# ESTIMATE MU AND SIGMA
mu_est = np.mean(sample)
sigma_est = np.std(sample)
print("Estimated MU: {}\nEstimated SIGMA: {}".format(mu_est, sigma_est))

##### PLOTTING #####
# SAMPLE DISTRIBUTION
plt.hist(sample, bins=50,normed=True, alpha=.25)

# TRUE CURVE
plt.plot(np.linspace(2, 18, 1000), norm.pdf(np.linspace(2, 18, 1000),loc=mu_real, scale=sigma_real))

# ESTIMATED CURVE
plt.plot(np.linspace(2, 18, 1000), norm.pdf(np.linspace(2, 18, 1000),loc=np.mean(sample), scale=np.std(sample)))

# LEGEND
plt.text(x=9.5, y=.1, s="sample", alpha=.75, weight="bold", color="#008fd5")
plt.text(x=7, y=.2, s="true distrubtion", rotation=55, alpha=.75, weight="bold", color="#fc4f30")
plt.text(x=5, y=.12, s="estimated distribution", rotation=55, alpha=.75, weight="bold", color="#e5ae38")

# TICKS
plt.tick_params(axis = 'both', which = 'major', labelsize = 18)
plt.axhline(y = 0, color = 'black', linewidth = 1.3, alpha = .7)

# TITLE
plt.text(x = 0, y = 0.3, s = "Normal Distribution",
               fontsize = 26, weight = 'bold', alpha = .75)
```

![](http://img.clint-sfy.cn/python/math/分布4.png)

### 2. 二项分布

```
让我们来看看玩板球这个例子。假设你今天赢了一场比赛，这表示一个成功的事件。你再比了一场，但你输了。如果你今天赢了一场比赛，但这并不表示你明天肯定会赢。我们来分配一个随机变量X，用于表示赢得的次数。 X可能的值是多少呢？它可以是任意值，这取决于你掷硬币的次数。

只有两种可能的结果，成功和失败。因此，成功的概率 = 0.5，失败的概率可以很容易地计算得到：q = p – 1 = 0.5。

二项式分布就是只有两个可能结果的分布，比如成功或失败、得到或者丢失、赢或败，每一次尝试成功和失败的概率相等。

结果有可能不一定相等。如果在实验中成功的概率为0.2，则失败的概率可以很容易地计算得到 q = 1 - 0.2 = 0.8。

每一次尝试都是独立的，因为前一次投掷的结果不能决定或影响当前投掷的结果。只有两个可能的结果并且重复n次的实验叫做二项式。二项分布的参数是n和p，其中n是试验的总数，p是每次试验成功的概率。

在上述说明的基础上，二项式分布的属性包括：

- 每个试验都是独立的。
- 在试验中只有两个可能的结果：成功或失败。
- 总共进行了n次相同的试验。
- 所有试验成功和失败的概率是相同的。 （试验是一样的）
```

![](http://img.clint-sfy.cn/python/math/分布5.png)

- PMF( 概率质量函数 ): 是对 离散随机变量 的定义. 是 离散随机变量 在各个特定取值的概率. 该函数通俗来说,就是 对于一个离散型概率事件来说, 使用这个函数来求它的各个成功事件结果的概率.

- PDF ( 概率密度函数 ): 是对 连续性随机变量 的定义. 与PMF不同的是 PDF 在特定点上的值并不是该点的概率, 连续随机概率事件只能求一段区域内发生事件的概率, 通过对这段区间进行积分来求. 通俗来说, 使用这个概率密度函数 将 想要求概率的区间的临界点( 最大值和最小值)带入求积分. 就是该区间的概率.

  ![](http://img.clint-sfy.cn/python/math/分布6.png)

  ![](http://img.clint-sfy.cn/python/math/分布7.png)

### 3. 泊松分布

现实生活多数服从于泊松分布

假设你在一个呼叫中心工作，一天里你大概会接到多少个电话？它可以是任何一个数字。现在，呼叫中心一天的呼叫总数可以用泊松分布来建模。这里有一些例子：

- 医院在一天内录制的紧急电话的数量。
- 某个地区在一天内报告的失窃的数量。
- 在一小时内抵达沙龙的客户人数。
- 在特定城市上报的自杀人数。
- 书中每一页打印错误的数量。
泊松分布适用于在随机时间和空间上发生事件的情况，其中，我们只关注事件发生的次数。

当以下假设有效时，则称为泊松分布

- 任何一个成功的事件都不应该影响另一个成功的事件。
- 在短时间内成功的概率必须等于在更长的间内成功的概率。
- 时间间隔很小时，在给间隔时间内成功的概率趋向于零。

泊松分布中使用了这些符号：

- λ是事件发生的速率
- t是时间间隔的长
- X是该时间间隔内的事件数。
- 其中，X称为泊松随机变量，X的概率分布称为泊松分布。

- 令μ表示长度为t的间隔中的平均事件数。那么，µ = λ*t。


例如说一个医院中，每个病人来看病都是随机并独立的概率，则该医院一天（或者其他特定时间段，一小时，一周等等）接纳的病人总数可以看做是一个服从poisson分布的随机变量。但是为什么可以这样处理呢？
通俗定义：假定一个事件在一段时间内随机发生，且符合以下条件：
- （1）将该时间段无限分隔成若干个小的时间段，在这个接近于零的小时间段里，该事件发生一次的概率与这个极小时间段的长度成正比。
- （2）在每一个极小时间段内，该事件发生两次及以上的概率恒等于零。
- （3）该事件在不同的小时间段里，发生与否相互独立。

则该事件称为poisson process。这个第二定义就更加利于大家理解了，回到医院的例子之中，如果我们把一天分成24个小时，或者24x60分钟，或者24x3600秒。时间分的越短，这个时间段里来病人的概率就越小（比如说医院在正午12点到正午12点又一毫秒之间来病人的概率是不是很接近于零？）。 条件一符合。另外如果我们把时间分的很细很细，是不是同时来两个病人（或者两个以上的病人）就是不可能的事件？即使两个病人同时来，也总有一个人先迈步子跨进医院大门吧。条件二也符合。倒是条件三的要求比较苛刻。应用到实际例子中就是说病人们来医院的概率必须是相互独立的，如果不是，则不能看作是poisson分布。

![](http://img.clint-sfy.cn/python/math/分布8.png)

![](http://img.clint-sfy.cn/python/math/分布9.png)



### 4. 均匀分布

对于投骰子来说，结果是1到6。得到任何一个结果的概率是相等的，这就是均匀分布的基础。与伯努利分布不同，均匀分布的所有可能结果的n个数也是相等的。

如果变量X是均匀分布的，则密度函数可以表示为：

![](http://img.clint-sfy.cn/python/math/分布10.png)

### 5. 卡方分布

```

通俗的说就是通过小数量的样本容量去预估总体容量的分布情况

卡方检验就是统计样本的实际观测值与理论推断值之间的偏离程度

若n个相互独立的随机变量ξ₁，ξ₂，...,ξn ，均服从标准正态分布（也称独立同分布于标准正态分布），则这n个服从标准正态分布的随机变量的平方和构成一新的随机变量，其分布规律称为卡方分布（chi-square distribution）

自由度：假设你现在手头有 3 个样本，。因为样本具有随机性，所以它们取值不定。但是假设出于某种原因，我们需要让样本均值固定，比如说，  ， 那么这时真正取值自由，”有随机性“ 的样本只有 2 个。 试想，如果  ,那么每选取一组  的取值，  将不得不等于  对于第三个样本来说，这种 “不得不” 就可以理解为被剥夺了一个自由度。所以就这个例子而言，3 个样本最终"自由"的只有其中的 2 个。不失一般性，  个样本， 留出一个自由度给固定的均值，剩下的自由度即为  。

卡方检验的基本思想是根据样本数据推断总体的频次与期望频次是否有显著性差异
```

![](http://img.clint-sfy.cn/python/math/分布11.png)

![](http://img.clint-sfy.cn/python/math/分布12.png)

### 6. beta分布

```
beta分布可以看作一个概率的概率分布，当你不知道一个东西的具体概率是多少时，它可以给出了所有概率出现的可能性大小

举一个简单的例子，熟悉棒球运动的都知道有一个指标就是棒球击球率(batting average)，就是用一个运动员击中的球数除以击球的总数，我们一般认为0.266是正常水平的击球率，而如果击球率高达0.3就被认为是非常优秀的。现在有一个棒球运动员，我们希望能够预测他在这一赛季中的棒球击球率是多少。你可能就会直接计算棒球击球率，用击中的数除以击球数，但是如果这个棒球运动员只打了一次，而且还命中了，那么他就击球率就是100%了，这显然是不合理的，因为根据棒球的历史信息，我们知道这个击球率应该是0.215到0.36之间才对啊。对于这个问题一个最好的方法就是用beta分布，这表示在我们没有看到这个运动员打球之前，我们就有了一个大概的范围。beta分布的定义域是(0,1)这就跟概率的范围是一样的。接下来我们将这些先验信息转换为beta分布的参数，我们知道一个击球率应该是平均0.27左右，而他的范围是0.21到0.35，那么根据这个信息，我们可以取α=81,β=219（击中了81次，未击中219次）

之所以取这两个参数是因为：

beta分布的均值是从图中可以看到这个分布主要落在了(0.2,0.35)间，这是从经验中得出的合理的范围。
在这个例子里，我们的x轴就表示各个击球率的取值，x对应的y值就是这个击球率所对应的概率。也就是说beta分布可以看作一个概率的概率分布。
```

![](http://img.clint-sfy.cn/python/math/分布13.png)

## 9. 核函数变换

### 1. 核函数的目的

如果我的数据有足够多的可利用的信息，那么我可以直接做我喜欢的事了，但是现在如果没有那么多的信息，我可不可以在数学上进行一些投机呢？

低维（比如我只知道一个人的年龄，性别，那我能对她多了解吗？）
高维（比如我知道他从出生开始，做过哪些事，赚过哪些钱等）

如果我们对数据更好的了解（是机器去了解他们，我们不需要认识啦）得到的结果不也会更好嘛。  

### 2. 线性核函数

![](http://img.clint-sfy.cn/python/math/核函数1.png)

### 3. 多项式核函数

![](http://img.clint-sfy.cn/python/math/核函数2.png)

### 4. 核函数实例

![](http://img.clint-sfy.cn/python/math/核函数3.png)

### 5. 高斯核函数

![](http://img.clint-sfy.cn/python/math/核函数4.png)

### 6. 参数的影响

![](http://img.clint-sfy.cn/python/math/核函数5.png)

## 10. 熵与激活函数

### 1. 熵的概念

![](http://img.clint-sfy.cn/python/math/熵1.png)

### 2. 熵的大小

![](http://img.clint-sfy.cn/python/math/熵2.png)

### 3. 激活函数

![](http://img.clint-sfy.cn/python/math/激活函数1.png)

### 4. 激活函数的问题

![](http://img.clint-sfy.cn/python/math/激活函数2.png)

输出值全为整数会导致梯度全为正或者全为负
优化更新会产生阶梯式情况  

![](http://img.clint-sfy.cn/python/math/激活函数3.png)

![](http://img.clint-sfy.cn/python/math/激活函数4.png)

![](http://img.clint-sfy.cn/python/math/激活函数5.png)

## 11、 回归分析

### 1. 回归分析概述

相关分析是研究两个或两个以上的变量之间相关程度及大小的一种统计方法

回归分析是寻找存在相关关系的变量间的数学表达式，并进行统计推断的一种统计方法

在对回归分析进行分类时，主要有两种分类方式：

- 根据变量的数目，可以分类一元回归、多元回归

- 根据自变量与因变量的表现形式，分为线性与非线性

所以，回归分析包括四个方向：一元线性回归分析、多元线性回归分析、一元非线性回归分析、多元非线性回归分析。

![](http://img.clint-sfy.cn/python/math/回归分析1.png)

![](http://img.clint-sfy.cn/python/math/回归分析5.png)

### 2. 回归方程定义

一元线性回归分析

- 因变量(dependent variable)：被预测或被解释的变量，用y表示

- 自变量(independent variable)：预测或解释因变量的一个或多个变量，用x表示 

- 对于具有线性关系的两个变量，可以用一个方程来表示它们之间的线性关系

- 描述因变量y如何依赖于自变量x和误差项ε的方程称为回归模型。对于只涉及一个自变量的一元线性回归模型可表示为：

  ```
  y = β0 + β1*x + ε
  - y叫做因变量或被解释变量
  - x叫做自变量或解释变量
  - β0 表示截距
  - β1 表示斜率
  - ε表示误差项，反映除x和y之间的线性关系之外的随机因素对y的影响
  ```

![](http://img.clint-sfy.cn/python/math/回归分析2.png)

### 3. 误差项的定义

![](http://img.clint-sfy.cn/python/math/回归分析3.png)

![](http://img.clint-sfy.cn/python/math/回归分析4.png)

### 4. 最小二乘法的推导和求解

![](http://img.clint-sfy.cn/python/math/回归分析6.png)

![](http://img.clint-sfy.cn/python/math/回归分析7.png)

![](http://img.clint-sfy.cn/python/math/回归分析8.png)

- 如果回归方程中的参数已知，对于一个给定的x值，利用回归方程就能计算出y的期望值

- 用样本统计量代替回归方程中的未知参数 ，就得到估计的回归方程，简称回归直线

  对于回归直线，关键在于求解参数，常用高斯提出的最小二乘法，它是使因变量的观察值y与估计值之间的离差平方和达到最小来求解

  ![](http://img.clint-sfy.cn/python/math/回归分析9.png)
  
  ![](http://img.clint-sfy.cn/python/math/回归分析10.png)

### 5. 回归方程求解最小例子

![](http://img.clint-sfy.cn/python/math/回归分析11.png)

- 点估计：利用估计的回归方程，对于x的某一个特定的值 ，求出y的一个估计值 就是点估计

- 区间估计：利用估计的回归方程，对于x的一个特定值 ，求出y的一个估计值的区间就是区间估计

  ![](http://img.clint-sfy.cn/python/math/回归分析12.png)

  ![](http://img.clint-sfy.cn/python/math/回归分析13.png)

  影响区间宽度的因素：

  - 置信水平 (1 - α)，区间宽度随置信水平的增大而增大

  - 数据的离散程度Se，区间宽度随离程度的增大而增大

  - 样本容量，区间宽度随样本容量的增大而减小

  - X0与X均值之间的差异，随着差异程度的增大而增大

### 6. 回归直线拟合优度

回归直线与各观测点的接近程度称为回归直线对数据的拟合优度

![](http://img.clint-sfy.cn/python/math/回归分析14.png)

总平方和可以分解为回归平方和、残差平方和两部分：SST＝SSR+SSE

- 总平方和(SST)，反映因变量的 n 个观察值与其均值的总离差

- 回归平方和SSR反映了y的总变差中，由于x与y之间的线性关系引起的y的变化部分

- 残差平方和SSE反映了除了x对y的线性影响之外的其他因素对y变差的作用，是不能由回归直线来解释的y的变差部分

#### 判定系数

回归平方和占总平方和的比例，用R^2表示,其值在0到1之间。

- R^2 == 0：说明y的变化与x无关，x完全无助于解释y的变差

- R^2 == 1：说明残差平方和为0，拟合是完全的，y的变化只与x有关

  ![](http://img.clint-sfy.cn/python/math/回归分析15.png)

#### 显著性检验

显著性检验的主要目的是根据所建立的估计方程用自变量x来估计或预测因变量y的取值。当建立了估计方程后，还不能马上进行估计或预测，因为该估计方程是根据样本数据得到的，它是否真实的反映了变量x和y之间的关系，则需要通过检验后才能证实。

根据样本数据拟合回归方程时，实际上就已经假定变量x与y之间存在着线性关系，并假定误差项是一个服从正态分布的随机变量，且具有相同的方差。但这些假设是否成立需要检验

显著性检验包括两方面：

- 线性关系检验
- 回归系数检验

```
线性关系检验
线性关系检验是检验自变量x和因变量y之间的线性关系是否显著，或者说，它们之间能否用一个线性模型来表示。

将均方回归 (MSR)同均方残差 (MSE)加以比较，应用F检验来分析二者之间的差别是否显著。

均方回归：回归平方和SSR除以相应的自由度(自变量的个数K)
均方残差：残差平方和SSE除以相应的自由度(n-k-1)
H0：β1=0 所有回归系数与零无显著差异，y与全体x的线性关系不显著
计算检验统计量F：
```

![](http://img.clint-sfy.cn/python/math/回归分析16.png)

```
     回归系数显著性检验的目的是通过检验回归系数β的值与0是否有显著性差异，来判断Y与X之间是否有显著的线性关系.若β=0,则总体回归方程中不含X项(即Y不随X变动而变动),因此,变量Y与X之间并不存在线性关系;若β≠0,说明变量Y与X之间存在显著的线性关系。 
```

![](http://img.clint-sfy.cn/python/math/回归分析17.png)

线性关系的检验是检验自变量与因变量是否可以用线性来表达，而回归系数的检验是对样本数据计算的回归系数检验总体中回归系数是否为0

- 在一元线性回归中，自变量只有一个，线性关系检验与回归系数检验是等价的

- 多元回归分析中，这两种检验的意义是不同的。线性关系检验只能用来检验总体回归关系的显著性，而回归系数检验可以对各个回归系数分别进行检验

### 7. 多元与曲线回归问题

经常会遇到某一现象的发展和变化取决于几个影响因素的情况，也就是一个因变量和几个自变量有依存关系的情况，这时需用多元线性回归分析。

- 多元线性回归分析预测法，是指通过对两上或两个以上的自变量与一个因变量的相关分析，建立预测模型进行预测和控制的方法
- 多元线性回归预测模型一般式为：

![](http://img.clint-sfy.cn/python/math/回归分析18.png)

![](http://img.clint-sfy.cn/python/math/回归分析19.png)

#### 曲线回归分析

直线关系是两变量间最简单的一种关系，曲线回归分析的基本任务是通过两个相关变量x与y的实际观测数据建立曲线回归方程，以揭示x与y间的曲线联系的形式。

曲线回归分析最困难和首要的工作是确定自变量与因变量间的曲线关系的类型，曲线回归分析的基本过程：

- 先将x或y进行变量转换
- 对新变量进行直线回归分析、建立直线回归方程并进行显著性检验和区间估计
- 将新变量还原为原变量，由新变量的直线回归方程和置信区间得出原变量的曲线回归方程和置信区间

由于曲线回归模型种类繁多，所以没有通用的回归方程可直接使用。但是对于某些特殊的回归模型，可以通过变量代换、取对数等方法将其线性化，然后使用标准方程求解参数，再将参数带回原方程就是所求。

#### 多重共线性

回归模型中两个或两个以上的自变量彼此相关的现象

多重共线性带来的问题有:

- 回归系数估计值的不稳定性增强

- 回归系数假设检验的结果不显著等

多重共线性检验的主要方法:
- 容忍度
- 方差膨胀因子（VIF）

![](http://img.clint-sfy.cn/python/math/回归分析20.png)

### 8. python工具包



### 9. statsmodels回归分析

```python
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

nsample = 20
x = np.linspace(0, 10, nsample)
x

# 一元线性回归
# 它可以将一个n行、k列的矩阵X变为一个n行、k+1列的矩阵，其中第一列都是1，代表常数项
X = sm.add_constant(x)
#β0,β1分别设置成2,5
beta = np.array([2, 5])
#误差项
e = np.random.normal(size=nsample)
#实际值y
y = np.dot(X, beta) + e

#最小二乘法
model = sm.OLS(y,X)

#拟合数据
res = model.fit()
#回归系数
res.params
array([ 1.49524076,  5.08701837])

#全部结果
res.summary()
# R-squared:	0.995 R^2值  R-squared:	0.995   调整过的R^2值
# F-statistic:	3668.  F检验 Prob (F-statistic):	2.94e-22 F检验的概率值 比较小相关性强
# const 常数项 x1斜率项
```

```python
#拟合的估计值
y_ = res.fittedvalues
y_
array([  1.49524076,   4.17261885,   6.84999693,   9.52737502,
        12.20475311,  14.8821312 ,  17.55950928,  20.23688737,
        22.91426546,  25.59164354,  28.26902163,  30.94639972,
        33.62377781,  36.30115589,  38.97853398,  41.65591207,
        44.33329015,  47.01066824,  49.68804633,  52.36542442])
```

```python
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(x, y, 'o', label='data')#原始数据
ax.plot(x, y_, 'r--.',label='test')#拟合数据
ax.legend(loc='best')
plt.show()
```

![](http://img.clint-sfy.cn/python/math/回归分析21.png)

### 10. 高阶与分类变量实例

#### 高阶回归

```PYTHON
#Y=5+2⋅X+3⋅X^2
 nsample = 50
x = np.linspace(0, 10, nsample)
X = np.column_stack((x, x**2))
X = sm.add_constant(X)
array([[1.00000000e+00, 0.00000000e+00, 0.00000000e+00],
```

```PYTHON
beta = np.array([5, 2, 3])
e = np.random.normal(size=nsample)
y = np.dot(X, beta) + e
model = sm.OLS(y,X)
results = model.fit()
results.params
array([ 4.93210623,  2.16604081,  2.97682135])

results.summary()
```

```PYTHON
y_fitted = results.fittedvalues
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(x, y, 'o', label='data')
ax.plot(x, y_fitted, 'r--.',label='OLS')
ax.legend(loc='best')
plt.show()
```

![](http://img.clint-sfy.cn/python/math/回归分析22.png)

#### 分类变量

假设分类变量有4个取值（a,b,c）,比如考试成绩有3个等级。那么a就是（1,0,0），b（0,1,0），c(0,0,1),这个时候就需要3个系数β0,β1,β2，也就是β0x0+β1x1+β2x2

```python
nsample = 50
groups = np.zeros(nsample,int)

groups[20:40] = 1
groups[40:] = 2
# 用于将离散变量转化为其它可以用于回归模型的变量
# 参数drop可以设置为True或False，表示是否删除一列
dummy = sm.categorical(groups, drop=True)
[1 2 1 2 3 1 2 1 0 3]
[[0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 1. 0. 0.]
 [1. 0. 0. 0.]
 [0. 0. 0. 1.]]
```

```python
#Y=5+2X+3Z1+6⋅Z2+9⋅Z3.
x = np.linspace(0, 20, nsample)
X = np.column_stack((x, dummy))
X = sm.add_constant(X)
beta = [5, 2, 3, 6, 9]
e = np.random.normal(size=nsample)
y = np.dot(X, beta) + e
result = sm.OLS(y,X).fit()
result.summary()
```

```python
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(x, y, 'o', label="data")
ax.plot(x, result.fittedvalues, 'r--.', label="OLS")
ax.legend(loc='best')
plt.show()
```

![](http://img.clint-sfy.cn/python/math/回归分析23.png)

### 11. 案例: 汽车价格预测任务

#### 任务概述

主要包括3类指标: 

- 汽车的各种特性.
- 保险风险评级：(-3, -2, -1, 0, 1, 2, 3).
- 每辆保险车辆年平均相对损失支付.

类别属性

- make: 汽车的商标（奥迪，宝马。。。）
- fuel-type: 汽油还是天然气
- aspiration: 涡轮
- num-of-doors: 两门还是四门 
- body-style: 硬顶车、轿车、掀背车、敞篷车
- drive-wheels: 驱动轮
- engine-location: 发动机位置
- engine-type: 发动机类型
- num-of-cylinders: 几个气缸
- fuel-system: 燃油系统

连续指标

- bore:                     continuous from 2.54 to 3.94.
- stroke:                   continuous from 2.07 to 4.17.
- compression-ratio:        continuous from 7 to 23.
- horsepower:               continuous from 48 to 288.
- peak-rpm:                 continuous from 4150 to 6600.
- city-mpg:                 continuous from 13 to 49.
- highway-mpg:              continuous from 16 to 54.
- price:                    continuous from 5118 to 45400.

```python
# loading packages
import numpy as np
import pandas as pd
from pandas import datetime

# data visualization and missing values
import matplotlib.pyplot as plt
import seaborn as sns # advanced vizs
import missingno as msno # missing values
%matplotlib inline

# stats
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.metrics import mean_squared_error, r2_score

# machine learning
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV 
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.ensemble import RandomForestRegressor
seed = 123

# importing data ( ? = missing values)
data = pd.read_csv("Auto-Data.csv", na_values = '?')
data.columns
```

```python
data.dtypes
```

```python
print("In total: ",data.shape)
data.head(5)
In total:  (205, 26)
    
data.describe()
```

#### 缺失值填充

```python
# missing values?
sns.set(style = "ticks")

msno.matrix(data)
#https://github.com/ResidentMario/missingno
# 缺失值占比比较小 可以去掉
# 用均值去填充
# 用线性回归去预测缺失值
```

![](http://img.clint-sfy.cn/python/math/汽车价格1.png)

```python
# 查看缺失值比较多的
data[pd.isnull(data['normalized-losses'])].head()

sns.set(style = "ticks")
plt.figure(figsize = (12, 5)) 
c = '#366DE8'

# ECDF 指经验累积分布函数
# 它是在统计学和概率论中，用来描述样本的累积分布情况的一种非参数方法。具体而言，ECDF是对样本的经验分布函数进行逐点估计，它表示所有数值小于某个特定值的样本比例。
plt.subplot(121)
cdf = ECDF(data['normalized-losses'])
plt.plot(cdf.x, cdf.y, label = "statmodels", color = c);
plt.xlabel('normalized losses'); plt.ylabel('ECDF');

# overall distribution
plt.subplot(122)
plt.hist(data['normalized-losses'].dropna(), 
         bins = int(np.sqrt(len(data['normalized-losses']))),
         color = c);
```

![](http://img.clint-sfy.cn/python/math/汽车价格2.png)

```
可以发现 80% 的 normalized losses 是低于200 并且绝大多数低于125. 

一个基本的想法就是用中位数来进行填充，但是我们得来想一想，这个特征跟哪些因素可能有关呢？应该是保险的情况吧，所以我们可以分组来进行填充这样会更精确一些。
首先来看一下对于不同保险情况的统计指标:

data.groupby('symboling')['normalized-losses'].describe()
```

```python
# replacing
# 直接删掉缺失值所在的行
data = data.dropna(subset = ['price', 'bore', 'stroke', 'peak-rpm', 'horsepower', 'num-of-doors'])
# 根据分组来填充均值
data['normalized-losses'] = data.groupby('symboling')['normalized-losses'].transform(lambda x: x.fillna(x.mean()))

print('In total:', data.shape)
data.head()
In total: (193, 26)
```

#### 特征相关性

```python
cormatrix = data.corr()
cormatrix *= np.tri(*cormatrix.values.shape, k=-1).T  #返回函数的上三角矩阵，把对角线上的置0，让他们不是最高的。
cormatrix = cormatrix.stack() # 当前指标和其他指标的关系
cormatrix = cormatrix.reindex(cormatrix.abs().sort_values(ascending=False).index).reset_index() # 两两之间的关系
cormatrix.columns = ["FirstVariable", "SecondVariable", "Correlation"]

cormatrix
	FirstVariable	SecondVariable	Correlation
0	city-mpg	highway-mpg	0.971975
1	engine-size	price	0.888778
```

```python
# city_mpg 和 highway-mpg 这哥俩差不多是一个意思了. 对于这个长宽高，他们应该存在某种配对关系，给他们合体吧！
# 相关性差不多的  合体  
data['volume'] = data.length * data.width * data.height
# 删掉不要的列
data.drop(['width', 'length', 'height', 
           'curb-weight', 'city-mpg'], 
          axis = 1, # 1 for columns
          inplace = True) 
# new variables
data.columns
```

```python
# Compute the correlation matrix 
corr_all = data.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr_all, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize = (11, 9))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_all, mask = mask,
            square = True, linewidths = .5, ax = ax, cmap = "BuPu")      
plt.show()
# 看起来 price 跟这几个的相关程度比较大 wheel-base,enginine-size, bore,horsepower.
```

![](http://img.clint-sfy.cn/python/math/汽车价格3.png)

```python
sns.pairplot(data, hue = 'fuel-type', palette = 'plasma') # 变量两两之间关系
```

```python
# 让我们仔细看看价格和马力变量之间的关系
# 在fuel-type和num-of-doors条件下，price和horsepower的关系
sns.lmplot('price', 'horsepower', data, 
           hue = 'fuel-type', col = 'fuel-type',  row = 'num-of-doors', 
           palette = 'plasma', 
           fit_reg = True);
# 事实上，对于燃料的类型和数门变量，我们看到，在一辆汽车马力的增加与价格成比例的增加相关的各个层面
```

![](http://img.clint-sfy.cn/python/math/汽车价格4.png)

#### 预处理问题

如果一个特性的方差比其他的要大得多，那么它可能支配目标函数，使估计者不能像预期的那样正确地从其他特性中学习。这就是为什么我们需要首先对数据进行缩放。

对连续值进行标准化

```python
# target and features
target = data.price

regressors = [x for x in data.columns if x not in ['price']]
features = data.loc[:, regressors]
# 只对数字型的进行标准化 其他除外
num = ['symboling', 'normalized-losses', 'volume', 'horsepower', 'wheel-base',
       'bore', 'stroke','compression-ratio', 'peak-rpm']

# scale the data
standard_scaler = StandardScaler() # 标准化
features[num] = standard_scaler.fit_transform(features[num])

# glimpse
features.head()
```

```python
# 对分类属性就行one-hot编码
# categorical vars 对字符串类型的 进行热编码 
classes = ['make', 'fuel-type', 'aspiration', 'num-of-doors', 
           'body-style', 'drive-wheels', 'engine-location',
           'engine-type', 'num-of-cylinders', 'fuel-system']

# create new dataset with only continios vars 
dummies = pd.get_dummies(features[classes])
features = features.join(dummies).drop(classes, 
                                       axis = 1)

# new dataset
print('In total:', features.shape)
features.head()
```

```python
# 划分数据集
# split the data into train/test set
X_train, X_test, y_train, y_test = train_test_split(features, target, 
                                                    test_size = 0.3,
                                                    random_state = seed)
print("Train", X_train.shape, "and test", X_test.shape)
Train (135, 66) and test (58, 66)
```

#### 回归问题

```python
# Lasso回归
# 多加了一个绝对值项来惩罚过大的系数，alphas=0那就是最小二乘了

# logarithmic scale: log base 2
# high values to zero-out more variables
alphas = 2. ** np.arange(2, 12)
scores = np.empty_like(alphas)

for i, a in enumerate(alphas):
    lasso = Lasso(random_state = seed)
    lasso.set_params(alpha = a)
    lasso.fit(X_train, y_train)
    scores[i] = lasso.score(X_test, y_test)

# 交叉验证 平均切成10份
lassocv = LassoCV(cv = 10, random_state = seed)
lassocv.fit(features, target)
lassocv_score = lassocv.score(features, target)
lassocv_alpha = lassocv.alpha_

plt.figure(figsize = (10, 4))
plt.plot(alphas, scores, '-ko')
plt.axhline(lassocv_score, color = c)
plt.xlabel(r'$\alpha$')
plt.ylabel('CV Score')
plt.xscale('log', basex = 2)
sns.despine(offset = 15)
# 选择一个最高点 为惩罚系数
print('CV results:', lassocv_score, lassocv_alpha)
```

![](http://img.clint-sfy.cn/python/math/汽车价格5.png)

```python
# lassocv coefficients
coefs = pd.Series(lassocv.coef_, index = features.columns)

# prints out the number of picked/eliminated features
print("Lasso picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  \
      str(sum(coefs == 0)) + " features.")

# takes first and last 10
coefs = pd.concat([coefs.sort_values().head(5), coefs.sort_values().tail(5)])

plt.figure(figsize = (10, 4))
coefs.plot(kind = "barh", color = c)
plt.title("Coefficients in the Lasso Model")
plt.show()
```

![](http://img.clint-sfy.cn/python/math/汽车价格6.png)

```python
# 用得到的最好的alpha去建模
model_l1 = LassoCV(alphas = alphas, cv = 10, random_state = seed).fit(X_train, y_train)
y_pred_l1 = model_l1.predict(X_test)

model_l1.score(X_test, y_test)
0.83307445226244159
```

```python
# residual plot 残差点画出来  实际值和预测值之间的差距
# 期望在0左右浮动
plt.rcParams['figure.figsize'] = (6.0, 6.0)

preds = pd.DataFrame({"preds": model_l1.predict(X_train), "true": y_train})
preds["residuals"] = preds["true"] - preds["preds"]
preds.plot(x = "preds", y = "residuals", kind = "scatter", color = c)
```

![](http://img.clint-sfy.cn/python/math/汽车价格7.png)

```python
def MSE(y_true,y_pred):
    mse = mean_squared_error(y_true, y_pred)
    print('MSE: %2.3f' % mse)
    return mse

def R2(y_true,y_pred):    
    r2 = r2_score(y_true, y_pred)
    print('R2: %2.3f' % r2)     
    return r2
# 均方误差MSE    r^2  还可以 
MSE(y_test, y_pred_l1); R2(y_test, y_pred_l1);

MSE: 3870543.789
R2: 0.833
```

```python
# predictions
d = {'true' : list(y_test),
     'predicted' : pd.Series(y_pred_l1)
    }

pd.DataFrame(d).head()
```

## 12 . 假设检验

### 1. 基本思想

- 什么是假设：对总体参数（均值，比例等）的具体数值所作的陈述。比如，我认为新的配方的药效要比原来的更好。
- 什么是假设检验：先对总体的参数提出某种假设，然后利用样本的信息判断假设是否成立的过程。比如，上面的假设我是要接受还是拒绝呢。

- 推广新的教育方案后，教学效果是否有所提高
- 醉驾判定为刑事犯罪后是否会使得交通事故减少
- 男生和女生在选文理科时是否存在性别因素影响

![](http://img.clint-sfy.cn/python/math/假设检验1.png)



```
显著性水平：

一个概率值，原假设为真时，拒绝原假设的概率，表示为 alpha 常用取值为0.01, 0.05, 0.10
一个公司要来招聘了，本来实际有200个人准备混一混，但是公司希望只有5%的人是浑水摸鱼进来的，所以可能会有200*0.05=4个人混进来，所谓显著性水平α，就是你允许最多有多大比例浑水摸鱼的通过你的测试。
```

```
假设检验的步骤：

提出假设
确定适当的检验统计量
规定显著性水平
计算检验统计量的值
做出统计决策
```

```
原假设与备择建设：

待检验的假设又叫原假设，也可以叫零假设，表示为H0。（零假设其实就是表示原假设一般都是说没有差异，没有改变。。。）
与原假设对比的假设叫做备择假设，表示为H1
一般在比较的时候，主要有等于，大于，小于
```

```
检验统计量：

计算检验的统计量
根据给定的显著性水平，查表得出相应的临界值
将检验统计量的值与显著性水平的临界值进行比较
得出拒绝或不拒绝原假设的结论
```

```
检验中常说的小概率：
在一次试验中，一个几乎不可能发生的事件发生的概率
在一次试验中小概率事件一旦发生，我们就有理由拒绝原假设
小概率由我们事先确定
```

### 2. 左右侧检验与双侧检验

```
P值：
是一个概率值
如果原假设为真，P-值是抽样分布中大于或小于样本统计量的概率
左侧检验时，P-值为曲线上方小于等于检验统计量部分的面积
右侧检验时，P-值为曲线上方大于等于检验统计量部分的面积
```

![](http://img.clint-sfy.cn/python/math/假设检验2.png)

```
当关键词有不得少于/低于的时候用左侧，比如灯泡的使用寿命不得少于/低于700小时时
当关键词有不得多于/高于的时候用右侧，比如次品率不得多于/高于5%时
```

![](http://img.clint-sfy.cn/python/math/假设检验3.png)

```
    单侧检验指按分布的一侧计算显著性水平概率的检验。用于检验大于、小于、高于、低于、优于、劣于等有确定性大小关系的假设检验问题。这类问题的确定是有一定的理论依据的。假设检验写作：μ1<μ2或μ1>μ2。

    双侧检验指按分布两端计算显著性水平概率的检验， 应用于理论上不能确定两个总体一个一定比另一个大或小的假设检验。一般假设检验写作H1：μ1≠μ2。
```

```
检验结果：
单侧检验
若p值 > α, 不拒绝 H0
若p值 < α, 拒绝 H0

双侧检验
若p-值 > α/2, 不拒绝 H0
若p-值 < α/2, 拒绝 H0
```

![](http://img.clint-sfy.cn/python/math/假设检验4.png)

### 3. Z检验基本原理

- 当总体标准差已知,样本量较大时用标准正态分布的理论来推断差异发生的概率，从而比较两个平均数的差异是否显著
- 标准正态变换后Ｚ的界值

![](http://img.clint-sfy.cn/python/math/假设检验5.png)

### 4. Z检验实例

研究正常人与高血压患者胆固醇含量(mg%)的资料如下,试比较两组血清胆固醇含量有无差别。

![](http://img.clint-sfy.cn/python/math/假设检验6.png)

   某机床厂加工一种零件，根据经验知道，该厂加工零件的椭圆度近似服从正态分布，其总体均值为μ=0.081mm，总体标准差为σ= 0.025 。今换一种新机床进行加工，抽取n=200个零件进行检验，得到的椭圆度为0.076mm。试问新机床加工零件的椭圆度的均值与以前有无显著差异？（α＝0.05）

![](http://img.clint-sfy.cn/python/math/假设检验7.png)

​      根据过去大量资料，某厂生产的灯泡的使用寿命服从正态分布N~(1020，100^2)。现从最近生产的一批产品中随机抽取16只，测得样本平均寿命为1080小时。试在0.05的显著性水平下判断这批产品的使用寿命是否有显著提高？(α＝0.05)

![](http://img.clint-sfy.cn/python/math/假设检验8.png)

### 5. T检验基本原理

```
根据研究设计,t检验有三种形式：
单个样本的t检验：
用来比较一组数据的平均值和一个数值有无差异。例如，你选取了5个人，测定了他们的身高，要看这五个人的身高平均值是否高于、低于还是等于1.70m，就需要用这个检验方法。

配对样本均数t检验(非独立两样本均数t检验)
用来看一组样本在处理前后的平均值有无差异。比如，你选取了5个人，分别在饭前和饭后测量了他们的体重，想检测吃饭对他们的体重有无影响，就需要用这个t检验。

两个独立样本均数t检验
用来看两组数据的平均值有无差异。比如，你选取了5男5女，想看男女之间身高有无差异，这样，男的一组，女的一组，这两个组之间的身高平均值的大小比较可用这种方法。
```

```
单个样本t检验

又称单样本均数t检验(one sample t test),适用于样本均数与已知总体均数μ0的比较,目的是检验样本均数所代表的总体均数μ是否与已知总体均数μ0有差别。
已知总体均数μ0一般为标准值、理论值或经大量观察得到的较稳定的指标值。
应用条件，总体标准α未知的小样本资料,且服从正态分布。
```

```
配对样本均数t检验：
简称配对t检验(paired t test),又称非独立两样本均数t检验,适用于配对设计计量资料均数的比较。
配对设计(paired design)是将受试对象按某些特征相近的原则配成对子，每对中的两个个体随机地给予两种处理。
```

```
配对样本均数t检验原理：
配对设计的资料具有对子内数据一一对应的特征,研究者应关心是对子的效应差值而不是各自的效应值。

进行配对t检验时，首选应计算各对数据间的差值d，将d作为变量计算均数。

配对样本t检验的基本原理是假设两种处理的效应相同，理论上差值d的总体均数μd 为0，现有的不等于0差值样本均数可以来自μd = 0的总体,也可以来μd ≠ 0的总体。

可将该检验理解为差值样本均数与已知总体均数μd（μd = 0）比较的单样本t检验,其检验统计量为：
```

### 6. T检验实例

有12名接种卡介苗的儿童，8周后用两批不同的结核菌素，一批是标准结核菌素，一批是新制结核菌素，分别注射在儿童的前臂，两种结核菌素的皮肤浸润反应平均直径(mm)如表所示，问两种结核菌素的反应性有无差别。

![](http://img.clint-sfy.cn/python/math/假设检验9.png)

![](http://img.clint-sfy.cn/python/math/假设检验10.png)

```
两独立样本t检验
两独立样本t 检验(two independent sample t-test)，又称成组 t 检验。

适用于完全随机设计的两样本均数的比较,其目的是检验两样本所来自总体的均数是否相等。

完全随机设计是将受试对象随机地分配到两组中，每组患者分别接受不同的处理，分析比较处理的效应。

两独立样本t检验要求两样本所代表的总体服从正态分布N(μ1，σ^2)和N(μ2，σ^2)，且两总体方差σ1^2、σ2^2相等,即方差齐性。若两总体方差不等需要先进行变换

两独立样本t检验原理
两独立样本t检验的检验假设是两总体均数相等,即H0：μ1=μ2，也可表述为μ1－μ2=0,这里可将两样本均数的差值看成一个变量样本,则在H0条件下两独立样本均数t检验可视为样本与已知总体均数μ1－μ2=0的单样本t检验, 统计量计算公式为：
```

![](http://img.clint-sfy.cn/python/math/假设检验11.png)



### 7. T检验应用条件

- 两组计量资料小样本比较
- 样本对总体有较好代表性，对比组间有较好组间均衡性——随机抽样和随机分组
- 样本来自正态分布总体，配对t检验要求差值服从正态分布，大样本时，用z检验，且正态性要求可以放宽
- 两独立样本均数t检验要求方差齐性——两组总体方差相等或两样本方差间无显著性

### 8. 卡方检验

用于检验两个（或多个）率或构成比之间差别是否有统计学意义，配对卡方检验检验配对计数资料的差异是否有统计学意义。

## 卡方检验(Chi-square test)

用于检验两个（或多个）率或构成比之间差别是否有统计学意义，配对卡方检验检验配对计数资料的差异是否有统计学意义。

检验实际频数(A)和理论频数(T)的差别是否由抽样误差所引起的。也就是由样本率（或样本构成比）来推断总体率或构成比。

![](http://img.clint-sfy.cn/python/math/假设检验12.png)

ARC是位于R行C列交叉处的实际频数， TRC是位于R行C列交叉处的理论频数。 （ ARC - TRC ）反映实际频数与理论频数的差距，除以TRC 为的是考虑相对差距。所以，χ^2 值反映了实际频数与理论频数的吻合程度， χ^2 值大，说明实际频数与理论频数的差距大。 χ^2 值的大小除了与实际频数和理论频数的差的大小有关外，还与它们的行、列数有关。即自由度的大小。

![](http://img.clint-sfy.cn/python/math/假设检验13.png)

某药品检验所随机抽取574名成年人，研究抗生素的耐药性（资料如表8-11）。问两种人群的耐药率是否一致？

![](http://img.clint-sfy.cn/python/math/假设检验14.png)

### 9. 假设检验中的两类错误

```
第一类错误（弃真错误）：
原假设为真时拒绝原假设
第一类错误的概率为α

第二类错误（取伪错误）：
原假设为假时接受原假设
第二类错误的概率为β
```

![](http://img.clint-sfy.cn/python/math/假设检验15.png)

```
一个公司有员工3000 人（研究的总体） ，为了检验公司员工工资统计报表的真实性，研究者作了 50 人的大样本随机抽样调查，人均收入的调查结果是： X （样本均值）=871 元；S（标准差）=21 元 问能否认为统计报表中人均收入μ0=880 元的数据是真实的？（显著性水平α=0.05 ）

原假设 H0：调查数据 871 元与报表数据 880 元之间没有显著性差异，公司员工工资均值的真实情况为880 元；
假设 H1：调查数据和报表数据之间有显著性的差异，公司员工工资均值的真实情况不是880 元。
α 错误出现原因：
我们只抽了一个样本，而个别的样本可能是特殊的，不管你的抽样多么符合科学抽样的要求。理论上讲，在 3000 个员工中随机抽取 50 人作为调查样本，有很多种构成样本的可能性，相当于 3000 选 50，这个数目是很大的。这样，在理论上就有存在很多个样本平均数。也就是说，由于小概率事件的出现，我们把本来真实的原假设拒绝了。这就是 α 错误出现的原因。

β 错误出现原因：
第二个问题是，统计检验的逻辑犯了从结论推断前提的错误。命题 B 是由命题 A 经演绎推论出来的，或写作符号 A→B，命题 C 是我们在检验中所依据操作法则。如果A 是真的，且我们从 A 到 B 的演绎推论如果也是正确的，那么B 可能是真实的。相反，如果结果 B是真实的，那么就不能得出A 必定是真实的结论。这就是 β错误出现的原因。

α 错误概率计算：
由实际推断原理引起的，即“小概率事件不会发生”的假定所引起的，所以有理由将所有小概率事件发生的概率之和或者即显著性水平（α=0.05）看作α错误发生的概率，换言之，α错误发生的概率为检验所选择的显著性水平。如果是单侧检验，弃真错误的概率则为 α/2。

β错误的概率计算：
犯β错误的概率的计算是比较复杂的，由于β错误的出现原因是属于逻辑上的，所以在总体参数不知道的情况下是无法计算它出现概率的大小的。 我们在以上例子的基础上进一步设计：这个公司职员的实际工资不是880 元，而是是 870 元，原假设为伪，仍然假设实际工资是880元。这样我们就可以在总体均值为 870 元和 880元两种情况下， 分别作出两条正态分布曲线 （A线和 B 线）

犯 β错误的概率大小就是相对正态曲线A 而言，图 1 中阴影部分的面积： ZX1=1.41 ；ZX2=5.59
查标准正态分布表可知，β=Φ（ZX2）-Φ（ZX1）=0.0793 结果表明，如果总体的真值为 870 元，而虚无假设为880元的话，那么，平均而言每100 次抽样中，将约有8次把真实情况当作880 元被接受，即犯β错误的概率大小是0.0793。

犯第一类错误的危害较大，由于报告了本来不存在的现象，则因此现象而衍生出的后续研究、应用的危害将是不可估量的。想对而言，第二类错误的危害则相对较小，因为研究者如果对自己的假设很有信心，可能会重新设计实验，再次来过，直到得到自己满意的结果（但是如果对本就错误的观点坚持的话，可能会演变成第一类错误）。
```

### 10. 案例：假设检验

```python
import pandas as pd
import pylab
import math
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
from scipy.stats import norm
import scipy.stats
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('normtemp.txt',sep='   ',names = ['Temperature','Gender','Heart Rate'])
```

要保证数据为正态分布

```python
observed_temperatures = df['Temperature'].sort_values()
bin_val = np.arange(start= observed_temperatures.min(), stop= observed_temperatures.max(), step = .05)
mu, std = np.mean(observed_temperatures), np.std(observed_temperatures)


p = norm.pdf(observed_temperatures, mu, std)


plt.hist(observed_temperatures,bins = bin_val, normed=True, stacked=True)
plt.plot(observed_temperatures, p, color = 'red')
plt.xticks(np.arange(95.75,101.25,0.25),rotation=90)
plt.xlabel('Human Body Temperature Distributions')
plt.xlabel('human body temperature')
plt.show()

print('Average (Mu): '+ str(mu) + ' / ' 'Standard Deviation: '+str(std))
```

![](http://img.clint-sfy.cn/python/math/假设检验16.png)

```python
x = observed_temperatures

#Shapiro-Wilk Test: https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test
shapiro_test, shapiro_p = scipy.stats.shapiro(x)
print("Shapiro-Wilk Stat:",shapiro_test, " Shapiro-Wilk p-Value:", shapiro_p)

k2, p = scipy.stats.normaltest(observed_temperatures)
print('p:',p) # p值大于0.05 可以接受假设 符合正态分布


#Another method to determining normality is through Quantile-Quantile Plots.
scipy.stats.probplot(observed_temperatures, dist="norm", plot=pylab)
pylab.show()
```

![](http://img.clint-sfy.cn/python/math/假设检验17.png)

```python
def ecdf(data):
    #Compute ECDF
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1) / n
    return x, y

# Compute empirical mean and standard deviation

# Number of samples
n = len(df['Temperature']) 

# Sample mean
mu = np.mean(df['Temperature']) 

# Sample standard deviation
std = np.std(df['Temperature']) 

print('Mean temperature: ', mu, 'with standard deviation of +/-', std)

#Random sampling of the data based off of the mean of the data.
normalized_sample = np.random.normal(mu, std, size=10000)
x_temperature, y_temperature = ecdf(df['Temperature'])
normalized_x, normalized_y = ecdf(normalized_sample)

# Plot the ECDFs
fig = plt.figure(figsize=(8, 5))
plt.plot(normalized_x, normalized_y)
plt.plot(x_temperature, y_temperature, marker='.', linestyle='none')
plt.ylabel('ECDF')
plt.xlabel('Temperature')
plt.legend(('Normal Distribution', 'Sample data'))
```

![](http://img.clint-sfy.cn/python/math/假设检验18.png)

有学者提出98.6是人类的平均体温，我们该这样认为吗？
在这里我们选择t检验，因为我们只能计算样本的标准差

```python

from scipy import stats
# T检验
CW_mu = 98.6
stats.ttest_1samp(df['Temperature'], CW_mu, axis=0)
Ttest_1sampResult(statistic=-5.4548232923640771, pvalue=2.4106320415610081e-07)
# T-Stat -5.454 p-value近乎0了. 我们该拒绝这样的假设
```

男性和女性的体温有明显差异吗
两独立样本t检验 H0: 没有明显差异 H1: 有明显差异

```python

female_temp = df.Temperature[df.Gender == 2]
male_temp = df.Temperature[df.Gender == 1]
mean_female_temp = np.mean(female_temp)
mean_male_temp = np.mean(male_temp)
print('Average female body temperature = ' + str(mean_female_temp))
print('Average male body temperature = ' + str(mean_male_temp))

# Compute independent t-test  独立的
stats.ttest_ind(female_temp, male_temp, axis=0)
Average female body temperature = 98.39384615384616
Average male body temperature = 98.1046153846154
Ttest_indResult(statistic=2.2854345381654984, pvalue=0.02393188312240236)
# 由于P值=0.024 < 0.05，我们需要拒绝原假设，我们有%95的自信认为是有差异的！
```

### 11. 案例： 卡方检验

白人和黑人在求职路上会有种族的歧视吗？

```python
data = pd.io.stata.read_stata('us_job_market_discrimination.dta')
blacks = data[data.race == 'b']
whites = data[data.race == 'w']
```

```
卡方检验
白人获得职位
白人被拒绝
黑人获得职位
黑人被拒绝

假设检验
H0：种族对求职结果没有显著影响
H1：种族对求职结果有影响
```

```python
blacks_called = len(blacks[blacks['call'] == True])
blacks_not_called = len(blacks[blacks['call'] == False])
whites_called = len(whites[whites['call'] == True])
whites_not_called = len(whites[whites['call'] == False])
observed = pd.DataFrame({'blacks': {'called': blacks_called, 'not_called': blacks_not_called},
                         'whites': {'called' : whites_called, 'not_called' : whites_not_called}})

num_called_back = blacks_called + whites_called
num_not_called = blacks_not_called + whites_not_called

# 得到期望的比率
rate_of_callbacks = num_called_back / (num_not_called + num_called_back)
expected_called = len(data)  * rate_of_callbacks
expected_not_called = len(data)  * (1 - rate_of_callbacks)

import scipy.stats as stats
observed_frequencies = [blacks_not_called, whites_not_called, whites_called, blacks_called]
expected_frequencies = [expected_not_called/2, expected_not_called/2, expected_called/2, expected_called/2]

stats.chisquare(f_obs = observed_frequencies,
                f_exp = expected_frequencies)
Power_divergenceResult(statistic=16.879050414270221, pvalue=0.00074839594410972638)
# 看起来种族歧视是存在的！
```

## 13. 相关分析

### 1. 概述

```
相关分析：

衡量事物之间或称变量之间线性相关程度的强弱，并用适当的统计指标表示出来的过程。
比如，家庭收入和支出、一个人所受教育程度与其收入、子女身高和父母身高等
```

```
相关系数：

- 衡量变量之间相关程度的一个量值
- 相关系数r的数值范围是在一1到十1之间
- 相关系数r的正负号表示变化方向。“+”号表示变化方向一致，即正相关；“-”号表示变化方向相反，即负相关
- r的绝对值表示变量之间的密切程度(即强度)。绝对值越接近1，表示两个变量之间关系越密切；越接近0，表示两个变量之间关系越不密切
- 相关系数的值，仅仅是一个比值。它不是由相等单位度量而来(即不等距)，也不是百分比，因此，不能直接作加、减、乘、除运算
- 相关系数只能描述两个变量之间的变化方向及密切程度，并不能揭示两者之间的内在本质联系，即存在相关的两个变量，不一定存在因果关系
```

### 2. 皮尔森相关系数

```
连续变量的相关分析

连续变量即数据变量，它的取值之间可以比较大小，可以用加减法计算出差异的大小。如“年龄”、“收入”、“成绩”等变量。
当两个变量都是正态连续变量，而且两者之间呈线性关系时，通常用Pearson相关系数来衡量
```

![](http://img.clint-sfy.cn/python/math/相关系数1.png)

```
虽然协方差能反映两个随机变量的相关程度（协方差大于0的时候表示两者正相关，小于0的时候表示两者负相关），但是协方差值的大小并不能很好地度量两个随机变量的关联程度。

在二维空间中分布着一些数据，我们想知道数据点坐标X轴和Y轴的相关程度，如果X与Y的相关程度较小但是数据分布的比较离散，这样会导致求出的协方差值较大，用这个值来度量相关程度是不合理的

为了更好的度量两个随机变量的相关程度，引入了Pearson相关系数，其在协方差的基础上除以了两个随机变量的标准差
```

![](http://img.clint-sfy.cn/python/math/相关系数2.png)

```
pearson是一个介于-1和1之间的值，当两个变量的线性关系增强时，相关系数趋于1或-1；当一个变量增大，另一个变量也增大时，表明它们之间是正相关的，相关系数大于0；如果一个变量增大，另一个变量却减小，表明它们之间是负相关的，相关系数小于0；如果相关系数等于0，表明它们之间不存在线性相关关系。
```

```python
# np.corrcoef(a)可计算行与行之间的相关系数，np.corrcoef(a,rowvar=0)用于计算各列之间的相关系数
import numpy as np
tang = np.array([[10, 10, 8, 9, 7],  
       [4, 5, 4, 3, 3],  
       [3, 3, 1, 1, 1]])

np.corrcoef(tang)
array([[ 1.        ,  0.64168895,  0.84016805],
       [ 0.64168895,  1.        ,  0.76376262],
       [ 0.84016805,  0.76376262,  1.        ]])


np.corrcoef(tang,rowvar=0) 
array([[ 1.        ,  0.98898224,  0.9526832 ,  0.9939441 ,  0.97986371],
       [ 0.98898224,  1.        ,  0.98718399,  0.99926008,  0.99862543],
       [ 0.9526832 ,  0.98718399,  1.        ,  0.98031562,  0.99419163],
       [ 0.9939441 ,  0.99926008,  0.98031562,  1.        ,  0.99587059],
       [ 0.97986371,  0.99862543,  0.99419163,  0.99587059,  1.        ]])
```

![](http://img.clint-sfy.cn/python/math/相关系数3.png)

### 3. 计算与检验

相关系数的假设性检验

![](http://img.clint-sfy.cn/python/math/相关系数4.png)

![](http://img.clint-sfy.cn/python/math/相关系数5.png)

```python

import numpy as np
import scipy.stats as stats  
import scipy
#https://docs.scipy.org/doc/scipy-0.19.1/reference/stats.html#module-scipy.stats

x = [10.35, 6.24, 3.18, 8.46, 3.21, 7.65, 4.32, 8.66, 9.12, 10.31]  
y = [5.1, 3.15, 1.67, 4.33, 1.76, 4.11, 2.11, 4.88, 4.99, 5.12]  
correlation,pvalue = stats.stats.pearsonr(x,y) 
print ('correlation',correlation)
print ('pvalue',pvalue)
correlation 0.989176319869
pvalue 5.92687594648e-08

import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.show()
```



### 4. 斯皮尔曼等级相关

```
等级变量的相关分析

当测量得到的数据不是等距或等比数据，而是具有等级顺序的数据；或者得到的数据是等距或等比数据，但其所来自的总体分布不是正态的，不满足求皮尔森相关系数（积差相关）的要求。这时就要运用等级相关系数。
```

![](http://img.clint-sfy.cn/python/math/相关系数6.png)

```
计算得出，他们的皮尔森相关系数r=1，P-vlaue≈0，从以上可以直观看出，如果两个基因的表达量呈线性关系，则具有显著的皮尔森相关性。

以上是两个基因呈线性关系的结果。如果两者呈非线性关系，例如幂函数关系（曲线关系），那又如何呢？ 我们再试试。

两个基因A、D，他们的关系是D=A^10，在8个样本中的表达量值如下：
```

![](http://img.clint-sfy.cn/python/math/相关系数7.png)

```python

import numpy as np
import scipy.stats as stats  
import scipy

x = [0.6,0.7,1,2.1,2.9,3.2,5.5,6.7]
y = np.power(x,10)
correlation,pvalue = stats.stats.pearsonr(x,y) 
print ('correlation',correlation)
print ('pvalue',pvalue)
correlation 0.765928796314
pvalue 0.0266964972088
```

```
可以看到，基因A、D相关系数，无论数值还是显著性都下降了。皮尔森相关系数是一种线性相关系数，因此如果两个变量呈线性关系的时候，具有最大的显著性。对于非线性关系（例如A、D的幂函数关系），则其对相关性的检测功效会下降。

这时我们可以考虑另外一个相关系数计算方法：斯皮尔曼等级相关。
```

#### 概述

```
当两个变量值以等级次序排列或以等级次序表示时，两个相应总体并不一定呈正态分布，样本容量也不一定大于30，表示这两变量之间的相关，称为Spearman等级相关。

简单点说，就是无论两个变量的数据如何变化，符合什么样的分布，我们只关心每个数值在变量内的排列顺序。如果两个变量的对应值，在各组内的排序顺位是相同或类似的，则具有显著的相关性。
```

![](http://img.clint-sfy.cn/python/math/相关系数8.png)

```
利用斯皮尔曼等级相关计算A、D基因表达量的相关性，结果是：r=1，p-value = 4.96e-05

这里斯皮尔曼等级相关的显著性显然高于皮尔森相关。这是因为虽然两个基因的表达量是非线性关系，但两个基因表达量在所有样本中的排列顺序是完全相同的，因为具有极显著的斯皮尔曼等级相关性。
```

```python
x = [10.35, 6.24, 3.18, 8.46, 3.21, 7.65, 4.32, 8.66, 9.12, 10.31]  
y = [5.13, 3.15, 1.67, 4.33, 1.76, 4.11, 2.11, 4.88, 4.99, 5.12]  
correlation,pvalue = stats.stats.spearmanr(x,y)  
print ('correlation',correlation)
print ('pvalue',pvalue)
correlation 1.0
pvalue 6.64689742203e-64


x = [10.35, 6.24, 3.18, 8.46, 3.21, 7.65, 4.32, 8.66, 9.12, 10.31]  
y = [5.13, 3.15, 1.67, 4.33, 1.76, 4.11, 2.11, 4.88, 4.99, 5.12]
x = scipy.stats.stats.rankdata(x)
y = scipy.stats.stats.rankdata(y)
print (x,y)
correlation,pvalue = stats.stats.spearmanr(x,y)  

print ('correlation',correlation)
print ('pvalue',pvalue)
[ 10.   4.   1.   6.   2.   5.   3.   7.   8.   9.] [ 10.   4.   1.   6.   2.   5.   3.   7.   8.   9.]
correlation 1.0
pvalue 6.64689742203e-64
```

#### 案例

![](http://img.clint-sfy.cn/python/math/相关系数9.png)

![](http://img.clint-sfy.cn/python/math/相关系数10.png)

### 5. 肯德尔和谐系数

```
当多个（两个以上）变量值以等级次序排列或以等级次序表示，描述这几个变量之间的一致性程度的量，称为肯德尔和谐系数。它常用来表示几个评定者对同一组学生成绩用等级先后评定多次之间的一致性程度。
```

![](http://img.clint-sfy.cn/python/math/相关系数11.png)

![](http://img.clint-sfy.cn/python/math/相关系数12.png)

#### 案例1：同一评价者无相同等级评定时

![](http://img.clint-sfy.cn/python/math/相关系数13.png)

![](http://img.clint-sfy.cn/python/math/相关系数14.png)

#### 案例2：同一评价者有相同等级评定时

![](http://img.clint-sfy.cn/python/math/相关系数15.png)

![](http://img.clint-sfy.cn/python/math/相关系数16.png)

#### 肯德尔和谐系数的显著性检验

```
肯德尔和谐系数的显著性检验
评分者人数(k)在3-20之间，被评者(N)在3-7之间时，可查《肯德尔和谐系数(W)显著性临界值表》，检验W是否达到显著性水平。若实际计算的S值大于k、N相同的表内临界值 ，则W达到显著水平。

当K=6 N=6，查表得检验水平分别为α = 0.01，α = 0.05的临界值各为S0.01 = 282.4，S0.05 = 221.4，均小于实算的S=546，故W达到显著水平，认为6位教师对6篇论文的评定相当一致。

当被评者n＞7时，则可用如下的x2统计量对W是否达到显著水平作检验。
```

```python
x1 = [10, 9, 8, 7, 6]
x2 = [10, 8, 9, 6, 7]

tau, p_value = stats.kendalltau(x1, x2)
print ('tau',tau)
print ('p_value',p_value)
tau 0.6
p_value 0.141644690295
```

### 6. 质量相关分析

```python
质量相关是指一个变量为质，另一个变量为量，这两个变量之间的相关。如智商、学科分数、身高、体重等是表现为量的变量，男与女、优与劣、及格与不及格等是表现为质的变量。

质与量的相关主要包括二列相关、点二列相关、多系列相关。
```
#### 二列相关
```
二列相关
当两个变量都是正态连续变量．其中一个变量被人为地划分成二分变量(如按一定标推将属于正态连续变量的学科考试分数划分成及格与不及格，录取与未录取，把某一体育项目测验结果划分成通过与未通过，达标与末达标，把健康状况划分成好与差，等等)，表示这两个变量之间的相关，称为二列相关。

二列相关的使用条件：
- 两个变量都是连续变量，且总体呈正态分布，或总体接近正态分布，至少是单峰对称分布。
- 两个变量之间是线性关系。
- 二分变量是人为划分的，其分界点应尽量靠近中值。
- 样本容量应当大于80。
```

![](http://img.clint-sfy.cn/python/math/相关系数17.png)

#### 案例

![](http://img.clint-sfy.cn/python/math/相关系数18.png)

#### 点二列相关

```
当两个变量其中一个是正态连续性变量，另一个是真正的二分名义变量(例如，男与女，已婚和未婚，色盲与非色盲，生与死，等等)，这时，表示这两个变量之间的相关，称为点二列相关。
```

![](http://img.clint-sfy.cn/python/math/相关系数19.png)

![](http://img.clint-sfy.cn/python/math/相关系数20.png)

```python
x = [1,0,0,0,0,0,0,1,1,1,1,0,1,1,1,1,1,0,0,0]
y = [84,82,76,60,72,74,76,84,88,90,78,80,92,94,96,88,90,78,76,74]
stats.pointbiserialr(x, y)
PointbiserialrResult(correlation=0.7849870641173371, pvalue=4.1459279734903919e-05)
```

### 7. 品质相关分析

```python
两个变量都是按质划分成几种类别，表示这两个变量之间的相关称为品质相关。

如，一个变量按性别分成男与女，另一个变量按学科成绩分成及格与不及格；又如，一个变量按学校类别分成重点及非重点，另一个变量按学科成绩分成优、良、中、差，等等。
```

#### 列联相关系数

![](http://img.clint-sfy.cn/python/math/相关性分析21.png)

![](http://img.clint-sfy.cn/python/math/相关性分析22.png)

#### φ相关

![](http://img.clint-sfy.cn/python/math/相关分析23.png)

![](http://img.clint-sfy.cn/python/math/相关分析24.png)

### 8. 偏相关与复相关

#### 偏相关分析

```
在多要素所构成的地理系统中，先不考虑其它要素的影响，而单独研究两个要素之间的相互关系的密切程度，这称为偏相关。用以度量偏相关程度的统计量，称为偏相关系数。

在分析变量x1和x2之间的净相关时，当控制了变量x3的线性作用后，x1和x2之间的一阶偏相关系数定义为：
```

![](http://img.clint-sfy.cn/python/math/相关分析25.png)

```
偏相关系数的性质
偏相关系数分布的范围在-1到1之间
偏相关系数的绝对值越大，表示其偏相关程度越大
偏相关系数的绝对值必小于或最多等于由同一系列资料所求得的复相关系数，即 R1·23≥|r12·3|
```

![](http://img.clint-sfy.cn/python/math/相关分析26.png)

#### 复相关系数

```
反映几个要素与某一个要素之间 的复相关程度。复相关系数介于0到1之间。

复相关系数越大，则表明要素（变量）之间的相关程度越密切。复相关系数为1，表示完全相关；复相关系数为0，表示完全无关。

复相关系数必大于或至少等于单相关系数的绝对值。
```

![](http://img.clint-sfy.cn/python/math/相关分析27.png)

## 14. 方差分析

### 1. 方差分析概述

检验多个总体均值是否相等，通过分析察数据的误差判断各总体均值是否相等

为了对几个行业的服务质量进行评价，消费者协会在四个行业分别抽取了不同的企业作为样本。最近一年中消费者对总共23家企业投诉的次数如下表

![](http://img.clint-sfy.cn/python/math/方差分析1.png)

```
要做的事：
分析四个行业之间的服务质量是否有显著差异，也就是要判断“行业”对“投诉次数”是否有显著影响

如果它们的均值相等，就意味着“行业”对投诉次数是没有影响的，即它们之间的服务质量没有显著差异；如果均值不全相等，则意味着“行业”对投诉次数是有影响的，它们之间的服务质量有显著差异
```

```
相关概念：
    因素或因子(factor)：所要检验的对象，要分析行业对投诉次数是否有影响，行业是要检验的因素或因子
    水平或处理(treatment)：因素的不同表现，即每个自变量的不同取值称为因素的水平
    观察值：在每个因素水平下得到的样本值，每个行业被投诉的次数就是观察值
    试验：这里只涉及一个因素，因此称为单因素四水平的试验
    总体：因素的每一个水平可以看作是一个总体，比如零售业、旅游业、航空公司、家电制造业可以看作是四个总体
    样本数据：被投诉次数可以看作是从这四个总体中抽取的样本数据
```

```
不同行业被投诉的次数是有明显差异的
即使是在同一个行业，不同企业被投诉的次数也明显不同
家电制造也被投诉的次数较高，航空公司被投诉的次数较低
行业与被投诉次数之间有一定的关系

但是
仅从散点图上观察还不能提供充分的证据证明不同行业被投诉的次数之间有显著差异
这种差异也可能是由于抽样的随机性所造成的
需要有更准确的方法来检验这种差异是否显著，也就是进行方差分析
之所以叫方差分析，因为虽然我们感兴趣的是均值，但在判断均值之间是否有差异时则需要借助于方差
```

### 2. 方差的比较

```
基本思想：
比较两类误差，以检验均值是否相等
比较的基础是方差比
如果系统(处理)误差显著地不同于随机误差，则均值就是不相等的；反之，均值就是相等的

随机误差：
因素的同一水平(总体)下，样本各观察值之间的差异
比如，同一行业下不同企业被投诉次数是不同的
这种差异可以看成是随机因素的影响，称为随机误差

系统误差：
因素的不同水平(不同总体)下，各观察值之间的差异
比如，不同行业之间的被投诉次数之间的差异
这种差异可能是由于抽样的随机性所造成的，也可能是由于行业本身所造成的，后者所形成的误差是由系统性因素造成的，称为系统误差

组内方差：
因素的同一水平(同一个总体)下样本数据的方差
比如，零售业被投诉次数的方差
组内方差只包含随机误差

组间方差：
因素的不同水平(不同总体)下各样本之间的方差
比如，四个行业被投诉次数之间的方差
组间方差既包括随机误差，也包括系统误差

方差的比较：
- 若不同行业对投诉次数没有影响，则组间误差中只包含随机误差，没有系统误差。这时，组间误差与组内误差经过平均后的数值就应该很接近，它们的比值就会接近1
- 若不同行业对投诉次数有影响，在组间误差中除了包含随机误差外，还会包含有系统误差，这时组间误差平均后的数值就会大于组内误差平均后的数值，它们之间的比值就会大于1
- 这个比值大到某种程度时，就可以说不同水平之间存在着显著差异，也就是自变量对因变量有影响
- 判断行业对投诉次数是否有显著影响，实际上也就是检验被投诉次数的差异主要是由于什么原因所引起的。如果这种差异主要是系统误差，说明不同行业对投诉次数有显著影响
```

#### 前提

```
每个总体都应服从正态分布
对于因素的每一个水平，其观察值是来自服从正态分布总体的简单随机样本
比如，每个行业被投诉的次数必需服从正态分布

各个总体的方差必须相同
各组观察数据是从具有相同方差的总体中抽取的
比如，四个行业被投诉次数的方差都相等
观察值是独立的
比如，每个行业被投诉的次数与其他行业被投诉的次数独立
在上述假定条件下，判断行业对投诉次数是否有显著影响，实际上也就是检验具有同方差的四个正态总体的均值是否相等
```

### 3. 方差分析计算方法

#### 单因素方差分析

- 模型中有一个自变量（因素）和一个观测变量,其实就是关于在一个影响因素的不同水平下，观测变量均值差异的显著性检验。

提出假设：

- H0: μ1 = μ2 = 。。。 =μk ，自变量对因变量没有显著影响
- 即H1: μ1 μ2 。。。μ4 不完全相等， 自变量对因变量有显著影响

拒绝原假设，只表明至少有两个总体的均值不相等，并不意味着所有的均值都不相等

#### 检验的统计量

- 水平的均值
- 全部观察值的总均值
- 误差平方和
- 均方(MS)

![](http://img.clint-sfy.cn/python/math/方差分析3.png)

![](http://img.clint-sfy.cn/python/math/方差分析4.png)

```
平方和之间的关系
总离差平方和(SST)、误差项离差平方和(SSE)、水平项离差平方和 (SSA) 之间的关系
SST = SSA + SSE
```

```
SST反映全部数据总的误差程度；SSE反映随机误差的大小；SSA反映随机误差和系统误差的大小

如果原假设成立，则表明没有系统误差，组间平方和SSA除以自由度后的均方与组内平方和SSE和除以自由度后的均方差异就不会太大；如果组间均方显著地大于组内均方，说明各水平(总体)之间的差异不仅有随机误差，还有系统误差，判断因素的水平是否对其观察值有影响，实际上就是比较组间方差与组内方差之间差异的大小
```

```
均方MS
各误差平方和的大小与观察值的多少有关，为消除观察值多少对误差平方和大小的影响，需要将其平均，这就是均方，也称为方差，计算方法是用误差平方和除以相应的自由度
```

```
各自自由度
SST 的自由度为n-1，其中n为全部观察值的个数
SSA的自由度为k-1，其中k为因素水平(总体)的个数
SSE 的自由度为n-k
```

![](http://img.clint-sfy.cn/python/math/方差分析5.png)

```
根据给定的显著性水平，在F分布表中查找与第一自由度df1＝k-1、第二自由度df2=n-k 相应的临界值

若F>Fα ，则拒绝原假设H0 ，表明均值之间的差异是显著的，所检验的因素对观察值有显著影响
若F<Fα ，则不拒绝原假设H0 ，不能认为所检验的因素对观察值有显著影响
```

#### 案例

在评价某药物耐受性及安全性的I期临床试验中，对符合纳入标准的30名健康自愿者随机分为3组每组10名，各组注射剂量分别为0.5U、1U、2U，观察48小时部分凝血活酶时间（s）试问不同剂量的部分凝血活酶时间有无不同？

![](http://img.clint-sfy.cn/python/math/方差分析2.png)

### 4. 多重比较

- 通过对总体均值之间的配对比较来进一步检验到底哪些均值之间存在差异
- 可采用Fisher提出的最小显著差异方法，简写为LSD
- LSD方法是对检验两个总体均值是否相等的t检验方法的总体方差估计而得到的

![](http://img.clint-sfy.cn/python/math/方差分析6.png)

实例：颜色对销售额的影响

### 5. 多因素方差分析

```
无交互效应的多因素方差分析
有交互效应的多因素方差分析

主效应与交互效应
主效应（main effect）：各个因素对观测变量的单独影响称为主效应。
交互效应（interaction effect）：各个因素不同水平的搭配所产生的新的影响称为交互效应。
双因素方差分析的类型
双因素方差分析中因素A和B对结果的影响相互独立时称为无交互效应的双因素方差分析。
如果除了A和B对结果的单独影响外还存在交互效应，这时的双因素方差分析称为有交互效应的双因素方差分析 。
```

![](http://img.clint-sfy.cn/python/math/方差分析7.png)

```
双因素方差分析的步骤
提出假设
要说明因素A有无显著影响，就是检验如下假设：
H0：因素A不同水平下观测变量的总体均值无显著差异。

H1：因素A不同水平下观测变量的总体均值存在显著差异。

要说明因素B有无显著影响，就是检验如下假设：
H0：因素B不同水平下观测变量的总体均值无显著差异。

H1：因素B不同水平下观测变量的总体均值存在显著差异。

在有交互效应的双因素方差中，要说明两个因素的交互效应是否显著，还要检验第三组零假设和备择假设：
H0：因素A和因素B的交互效应对观测变量的总体均值无显著差异。

H1：因素A和因素B的交互效应对观测变量的总体均值存在显著差异。
```

![](http://img.clint-sfy.cn/python/math/方差分析8.png)

实例：
有四个品牌的彩电在五个地区销售，为分析彩电的品牌(品牌因素)和销售地区(地区因素)对销售量是否有影响，对每个品牌在各地区的销售量取得以下数据。试分析品牌和销售地区对彩电的销售量是否有显著影响？(α=0.05)

![](http://img.clint-sfy.cn/python/math/方差分析9.png)

### 6. 实例

```python
# 单因素方差分析

# 呷哺呷哺3个城市不同用户评分
from scipy.stats import f_oneway  
a = [10,9,9,8,8,7,7,8,8,9]        
b = [10,8,9,8,7,7,7,8,9,9]  
c = [9,9,8,8,8,7,6,9,8,9]  
f,p = f_oneway(a,b,c)  
print (f)  
print (p)
0.101503759398
0.903820890369

# 不能认为所检验的因素对观察值有显著影响
```

```python
# 多因素方差分析
from scipy import stats  
import pandas as pd  
import numpy as np  
from statsmodels.formula.api import ols  
from statsmodels.stats.anova import anova_lm  
  
environmental =  [5,5,5,5,5,4,4,4,4,4,3,3,3,3,3,2,2,2,2,2,1,1,1,1,1]       
ingredients    = [5,4,3,2,1,5,4,3,2,1,5,4,3,2,1,5,4,3,2,1,5,4,3,2,1]    
score      =     [5,5,4,3,2,5,4,4,3,2,4,4,3,3,2,4,3,2,2,2,3,3,3,2,1]  
  
data = {'E':environmental, 'I':ingredients, 'S':score}  
df = pd.DataFrame(data)  
df.head()

#（~）隔离因变量和自变量 (左边因变量，右边自变量 )
#（+）分隔各个自变量
#（:）表示两个自变量交互影响
formula = 'S~E+I +E:I '                           
                                                 
model = ols(formula,df).fit()                   
results = anova_lm(model)                       
print (results)  
# P值很小，拒绝假设
# E和I对结果有显著影响，之间并无交互
            df  sum_sq    mean_sq           F        PR(>F)
E          1.0    7.22   7.220000   54.539568  2.896351e-07
I          1.0   18.00  18.000000  135.971223  1.233581e-10
E:I        1.0    0.64   0.640000    4.834532  3.924030e-02
Residual  21.0    2.78   0.132381         NaN           NaN
```



## 15. 聚类分析

### 1. 层次聚类概述

```
层次聚类(Hierarchical Clustering)是聚类算法的一种，通过计算不同类别数据点间的相似度来创建一棵有层次的嵌套聚类树。在聚类树中，不同类别的原始数据点是树的最低层，树的顶层是一个聚类的根节点。创建聚类树有自下而上合并和自上而下分裂两种方法。

作为一家公司的人力资源部经理，你可以把所有的雇员组织成较大的簇，如主管、经理和职员；然后你可以进一步划分为较小的簇，例如，职员簇可以进一步划分为子簇：高级职员，一般职员和实习人员。所有的这些簇形成了层次结构，可以很容易地对各层次上的数据进行汇总或者特征化。
```

![](http://img.clint-sfy.cn/python/math/聚类1.png)

```
直观来看，上图中展示的数据划分为2个簇或4个簇都是合理的，甚至，如果上面每一个圈的内部包含的是大量数据形成的数据集，那么也许分成16个簇才是所需要的。

论数据集应该聚类成多少个簇，通常是在讨论我们在什么尺度上关注这个数据集。层次聚类算法相比划分聚类算法的优点之一是可以在不同的尺度上（层次）展示数据集的聚类情况。

基于层次的聚类算法（Hierarchical Clustering）可以是凝聚的（Agglomerative）或者分裂的（Divisive），取决于层次的划分是“自底向上”还是“自顶向下”
```

### 2. 层次聚类流程

```
自底向上的合并算法
层次聚类的合并算法通过计算两类数据点间的相似性，对所有数据点中最为相似的两个数据点进行组合，并反复迭代这一过程。简单的说层次聚类的合并算法是通过计算每一个类别的数据点与所有数据点之间的距离来确定它们之间的相似性，距离越小，相似度越高。并将距离最近的两个数据点或类别进行组合，生成聚类树。

相似度的计算
层次聚类使用欧式距离来计算不同类别数据点间的距离（相似度）。
```

![](http://img.clint-sfy.cn/python/math/聚类2.png)

![](http://img.clint-sfy.cn/python/math/聚类3.png)

```
两个组合数据点间的距离
计算两个组合数据点间距离的方法有三种，分别为Single Linkage，Complete Linkage和Average Linkage。在开始计算之前，我们先来介绍下这三种计算方法以及各自的优缺点。

Single Linkage：方法是将两个组合数据点中距离最近的两个数据点间的距离作为这两个组合数据点的距离。这种方法容易受到极端值的影响。两个很相似的组合数据点可能由于其中的某个极端的数据点距离较近而组合在一起。

Complete Linkage：Complete Linkage的计算方法与Single Linkage相反，将两个组合数据点中距离最远的两个数据点间的距离作为这两个组合数据点的距离。Complete Linkage的问题也与Single Linkage相反，两个不相似的组合数据点可能由于其中的极端值距离较远而无法组合在一起。

Average Linkage：Average Linkage的计算方法是计算两个组合数据点中的每个数据点与其他所有数据点的距离。将所有距离的均值作为两个组合数据点间的距离。这种方法计算量比较大，但结果比前两种方法更合理。

我们使用Average Linkage计算组合数据点间的距离。下面是计算组合数据点(A,F)到(B,C)的距离，这里分别计算了(A,F)和(B,C)两两间距离的均值。
```

![](http://img.clint-sfy.cn/python/math/聚类4.png)

### 3. 层次聚类实例

```python
import pandas as pd

seeds_df = pd.read_csv('./datasets/seeds-less-rows.csv')
seeds_df.head()

seeds_df.grain_variety.value_counts()  
varieties = list(seeds_df.pop('grain_variety'))

samples = seeds_df.values
samples
array([[ 14.88  ,  14.57  ,   0.8811,   5.554 ,   3.333 ,   1.018 ,   4.956 ],
       
#距离计算的 还有树状图
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

#进行层次聚类
mergings = linkage(samples, method='complete')
#树状图结果
fig = plt.figure(figsize=(10,6))
dendrogram(mergings,
           labels=varieties,
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.show()
  
```

![](http://img.clint-sfy.cn/python/math/聚类5.png)

```python
#得到标签结果
#maximum height自己指定
from scipy.cluster.hierarchy import fcluster
labels = fcluster(mergings, 6, criterion='distance')

df = pd.DataFrame({'labels': labels, 'varieties': varieties})
ct = pd.crosstab(df['labels'], df['varieties'])
ct
```

```python
# 不同距离的选择会产生不同的结果
import pandas as pd

scores_df = pd.read_csv('./datasets/eurovision-2016-televoting.csv', index_col=0)
country_names = list(scores_df.index)
scores_df.head()

#缺失值填充，没有的就先按满分算吧
scores_df = scores_df.fillna(12)
from sklearn.preprocessing import normalize
samples = normalize(scores_df.values)
samples
array([[ 0.09449112,  0.56694671,  0.        , ...,  0.        ,
         0.28347335,  0.        ],
       
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

mergings = linkage(samples, method='single')
fig = plt.figure(figsize=(10,6))
dendrogram(mergings,
           labels=country_names,
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.show()
```

![](http://img.clint-sfy.cn/python/math/聚类6.png)

```python
mergings = linkage(samples, method='complete')
fig = plt.figure(figsize=(10,6))
dendrogram(mergings,
           labels=country_names,
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.show()
```

![](http://img.clint-sfy.cn/python/math/聚类7.png)

### 4. Kmeans概述



### 5. Kmeans工作流程



### 6. Kmeans可视化展示



### 7. DBSCAN聚类算法



### 8. DBSCAN工作流程



### 9. DBSCAN可视化展示



### 10. 多种聚类算法概述



### 11. 案例 聚类



## 16. 贝叶斯分析