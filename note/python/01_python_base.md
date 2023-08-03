# Python语法

## python语言基础

### 1. 数值运算

基本数值操作

```python
abs(15.6)
15.6

round(15.6)
16

round(15.4)
15

min(2,3,4,5)
2
max(2,3,4,5)
5

1.3e-5
1.3e-05

1.3e5
130000.0

0xFF
255
```

### 2.  字符串

```
tang_str = 'hello python'

tang = 'hello'+'python'
tang
'hellopython'

tang_str * 3
```

#### 字符串操作

```python
tang = '1 2 3 4 5'
tang.split
['1', '2', '3', '4', '5']

tang = '1,2,3,4,5'
tang = tang.split(',')
```

```
tang_str = ' '
tang_str.join(tang)
'1 2 3 4 5'
```

```
tang = 'hello python'
tang.replace('python','world')

'hello world'
```

```
tang2.upper()
'HELLO WORLD'
```

```python
tang = '    hello python    '
tang.strip()  # 去空操作
'hello python'

tang.lstrip()
'hello python    '
tang.rstrip()
'    hello python'
```

```python
'{} {} {}'.format('tang','yu','di')
'tang yu di'
'{2} {1} {0}'.format('tang','yu','di')
'di yu tang'
'{tang} {yu} {di}'.format(tang = 10, yu =5, di = 1)
'10 5 1'
```

```python
tang = 'tang yu di:'
b = 456.0
c = 789 
result = '%s %f %d' % (tang,b,c) 
result
'tang yu di: 456.000000 789'
```

### 3. 索引

```python
tang = 'tang yu di'
tang[0]
't'
tang[5]
'y'
tang[-1]
'i
```

```python
tang[0:4] # 切片 左闭右开
'tang'
tang[5:]
'yu di'
tang[:7]
'tang yu'
tang[1:-2] 
'ang yu '
tang[-3:]
' di'
tang[:]
'tang yu di'
tang[::2] # 步长为2 取偶数列
'tn ud'
```

### 4. list结构

```python
tang = []
type(tang)
list
```

```python
tang = [1,2,3,4]
tang = ['1','2','3','4']
tang = [1,'tangyudi',3.5]
tang = list([1,2,3])
tang
```

#### 操作

```python
a = [123,456]
b = ['tang','yudi']
a + b
[123, 456, 'tang', 'yudi']


a * 3
[123, 456, 123, 456, 123, 456]


a[0:]
[123, 456]
```

```python
a
[1, 2, 3, 4, 5, 6, 7, 8, 9]
del a[0]
a
[2, 3, 4, 5, 6, 7, 8, 9]
del a[3:]
a
[2, 3, 4]
```

```python
a = [1,2,3,4,5,6,7,8,9]
8 in a
False

tang = 'tang yu di'
'tang' in tang
True
```

```python
a = [1,2,[3,4]]
a
[1, 2, [3, 4]]

a[2]
[3, 4]
```

```python
tang =['apple','banana','apple','apple','apple','banana','banana']
tang.count('apple')
4

tang =['apple','1','2','3','4','5','6']
tang.index('apple') # 找索引
0
```

#### 列表添加

```python
tang = []
tang.append(['tang','yudi'])
tang
['tang', 'tang', 'tang', ['tang', 'yudi']]


tang.insert(2,'python')
tang
['tang', 'tang', 'python', 'tang', ['tang', 'yudi'], 'tang', 'tang']


tang.remove(['tang', 'yudi'])
tang
['tang', 'tang', 'python', 'tang', 'tang', 'tang']

tang.pop(1)
```

```python 
tang = [1,2,3,9,6,3,2]
tang.sort()
tang
[1, 2, 2, 3, 3, 6, 9]

tang = [1,2,3,9,6,3,2]
tang2 = sorted(tang)
[1, 2, 2, 3, 3, 6, 9]

tang = ['di','yu','tang']
tang.reverse()
['tang', 'yu', 'di']
```

### 5. 字典

```python
tang = {}
type(tang)
dict

tang = dict()
type(tang)

tang = dict()
type(tang)
dict
```

#### 字典结构操作

```python
tang['first'] = 123
tang
{'first': 123, 'python': 456}

tang['python']
456



tang = {'tang':123,'yu':456,'di':789}
tang
{'di': 789, 'tang': 123, 'yu': 456}

tang_value = [1,2,3]
tang = {}
tang['yudi'] = tang_value
tang['yudi2'] = 3
tang['yudi2'] = '4'
{'yudi': [1, 2, 3], 'yudi2': '4'}


tang = dict([('tang',123),('yudi',456)])
tang
{'tang': 123, 'yudi': 456}


tang['tang'] += 1
tang
{'tang': 125, 'yudi': 456}

tang.get('tang') # 取值

tang.pop('tang')
tang
{'yudi': 456}

del tang['yudi']


tang = {'tang':123,'yudi':456}
tang2 = {'tang':789,'python':888}
tang.update(tang2)
tang
{'python': 888, 'tang': 789, 'yudi': 456}
```

```python
tang.keys()
dict_keys(['tang', 'python', 'yudi'])

tang.values()
dict_values([789, 888, 456])

tang.items()
dict_items([('tang', 789), ('python', 888), ('yudi', 456)])
```

### 6. 集合

```python
tang = set([123,123,123,456,456,456,789])
tang
{123, 456, 789}
```

```python
a = {1,2,3,4}
b = {2,3,4,5}
a.union(b)  # 并集
{1, 2, 3, 4, 5}

a|b # 并集
{1, 2, 3, 4, 5}

b.intersection(a) # 交集
{2, 3, 4}
a & b
{2, 3, 4}

a.difference(b) # 差集
{1}
b.difference(a)
{5}
a - b
{1}
b - a
{5}

a = {1,2,3,4,5,6}
b = {2,3,4}
b.issubset(a) # 是否
True
a.issubset(b)
False

b <= a
True
b > a
False
a <= a
True
a < a
False

a = {1,2,3}
a.add(4)
a
{1, 2, 3, 4}

a.update([4,5,6])
a
{1, 2, 3, 4, 5, 6}

a.remove(1)
a
{2, 3, 4, 5, 6}

a.pop()
a
{3, 4, 5, 6}
```

### 7. 赋值机制

为了提高内存效率，如果值较小，两个地址是一样的

```python
tang = 1000
yudi = tang
id(tang)
2683811812688
id(yudi)
2683811812688
```

### 8. 判断结构 

```python
tang = 50
if tang >200:
    print ('200')
elif tang < 100:
    print ('100')
else:
    print ('100-200')
    

tang = [123,456,789]
if 123 in tang:
    print ('ok')
ok

tang = {'tang':123,'yudi':456}
if 'tang' in tang:
    print  ('ok')
ok
```

### 9.  循环结构

```python
tangs = set(['tang','yu','di'])
while tangs:
    tang = tangs.pop()
    print (tang)
    
for name in tangs:
    print (name)
```

### 10. python函数

```python
def add_ab(a=1,b=2):
    return (a+b)
tang = add_ab()
tang


def add_number(a,*args):  # 可以不指定输入个数
    b = 0
    for i in args:
        a += i
        b += a
    return a,b
a,b = add_number(1,2,3)
print (a,b)
6 9


def add_number2(a,**kwargs):
    for arg,value in kwargs.items():
        print (arg,value)
add_number2(1,x=2,y=3)
y 3
x 2
```

### 11. python模块和包

```python
%%writefile tang.py   # 写成一个脚本
tang_v = 10

def tang_add(tang_list):
    tang_sum = 0
    for i in range(len(tang_list)):
        tang_sum += tang_list[i]
    return tang_sum
tang_list = [1,2,3,4,5]
print (tang_add(tang_list))

%run tang.py
15

import tang
15
```

### 12. 异常

```python
import math

for i in range(10):
    try:
        input_number = input('write a number')
        
        if input_number == 'q':
            break
        result = 1/math.log(float(input_number))
        print (result)
    except ValueError:
        print ('ValueError: input must > 0')
    except ZeroDivisionError:
        print ('log(value) must != 0')
    except Exception:
        print ('ubknow error')
```

```python
class TangError(ValueError):
    pass

cur_list = ['tang','yu','di']
while True:
    cur_input = input()
    if cur_input not in cur_list:
        raise TangError('Invalid input: %s' %cur_input)
```

### 13. 文件操作

```python
%%writefile tang.txt
hello python
tang yu di
jin tian tian qi bu cuo

txt = open('./data/tang.txt')
txt_read = txt.read()


lines = txt.readlines()
print (type(lines))
print (lines)
<class 'list'>
['hello python\n', 'tang yu di\n', 'jin tian tian qi bu cuo']

txt.close()
```

```python
txt = open('tang_write.txt','w')
txt.write('jin tian tian qi bu cuo')
txt.write('\n')
txt.write('tang yu di')
txt.close()

txt = open('tang_write.txt','w')
for i in range(100):
    txt.write(str(i)+'\n')
txt2 = open('tang_write.txt','r')
print (txt2.read())
```

```python
txt = open('tang_write.txt','w')
try:
    for i in range(100):
        10/(i-50)
        txt.write(str(i)+'\n')
except Exception:
    print ('error:',i)
finally:
    txt.close()
    
with open('tang_write.txt','w') as f:
    f.write('jin tian tian qi bu cuo')
```

### 14. 类

```python
class people:
    '帮助信息：XXXXXX'
    #所有实力都会共享
    number = 100
    #构造函数，初始化的方法，当创建一个类的时候，首先会调用它
    def __init__(self,name,age):
        self.name = name
        self.age = age
    def display(self):
        print ('number = :',people.number)
    def display_name(self):
        print (self.name)
```

```python 
people.__doc__
'帮助信息：XXXXXX'

p1 = people('tangyudi',30)
```

### 15. 时间

```python
import time

print (time.time())


print (time.localtime(time.time()))
time.struct_time(tm_year=2017, tm_mon=11, tm_mday=15, tm_hour=14, tm_min=59, tm_sec=5, tm_wday=2, tm_yday=319, tm_isdst=0)


print (time.asctime(time.localtime(time.time())))
Wed Nov 15 15:00:15 2017
        
print (time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()))
2017-11-15 15:02:07

import calendar
print (calendar.month(2017,11))
#print (help(calendar.month))
```

## numpy

### 1. 概述

```python
import numpy as np

array = np.array([1,2,3,4,5])
print (type(array))
<class 'numpy.ndarray'>

array2 = array + 1  #list就不行
array2
array([3, 4, 5, 6, 7])

array.shape
(5,)
```

### 2. array结构

对于ndarray结构来说，里面所有的元素必须是同一类型的 如果不是的话，会自动的向下进行转换

```python
tang_list = [1,2,3,4,5]
tang_array = np.array(tang_list)
tang_array
```

```python 
tang_array.dtype
dtype('int32')

tang_array.itemsize
4

tang_array.shape
(5,)

tang_array.size
5

np.size(tang_array)
5

np.shape(tang_array) 
(5,)

tang_array.ndim # 获取数组的维度
1

tang_array.fill(0)
tang_array
array([0, 0, 0, 0, 0])
```

```python
tang_array[1,1] = 10
tang_array
array([[ 1,  2,  3],
       [ 4, 10,  6],
       [ 7,  8,  9]])

tang_array[1]
array([ 4, 10,  6])

tang_array[:,1]
array([ 2, 10, 8])

tang_array[0,0:2]
array([1, 2])

tang_array2 = tang_array.copy()
```

```python 
tang_array = np.arange(0,100,10)  # 按照间隔
tang_array
array([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

mask = np.array([0,0,0,1,1,1,0,0,1,1],dtype=bool)
mask
array([False, False, False,  True,  True,  True, False, False,  True,  True], dtype=bool)

tang_array[mask]
array([30, 40, 50, 80, 90])
```

```python 
random_array = np.random.rand(10)
random_array
array([ 0.51388374,  0.57986996,  0.05474169,  0.5019837 ,  0.82705166,
        0.95557716,  0.83348612,  0.32385451,  0.52586287,  0.92505535])

np.where(tang_array > 30)
(array([3, 4], dtype=int64),)
```

```python
tang_array = np.array([10,20,30,40,50])
tang_array > 30
array([False, False, False,  True,  True], dtype=bool)

np.where(tang_array > 30)
(array([3, 4], dtype=int64),)

tang_array = np.array([1,2,3,4,5],dtype=np.float32)
tang_array
array([ 1.,  2.,  3.,  4.,  5.], dtype=float32)

tang_array.dtype
dtype('float32')

tang_array.nbytes  # 将返回该数组所占用的字节数
20
```

```python
tang_array = np.array([1,10,3.5,'str'],dtype = np.object)
tang_array
array([1, 10, 3.5, 'str'], dtype=object)

tang_array * 2
array([2, 20, 7.0, 'strstr'], dtype=object)

tang_array = np.array([1,2,3,4,5])
np.asarray(tang_array,dtype = np.float32)
array([ 1.,  2.,  3.,  4.,  5.], dtype=float32)

tang_array2 = np.asarray(tang_array,dtype = np.float32)
array([ 1.,  2.,  3.,  4.,  5.], dtype=float32)

tang_array.astype(np.float32) #转换类型
```

### 3. 数值计算

```python 
import numpy as np
tang_array = np.array([[1,2,3],[4,5,6]])

np.sum(tang_array) # 求和

np.sum(tang_array,axis=0) # 指定要进行的操作是沿着什么轴（维度）
tang_array.sum(axis = 0)
array([5, 7, 9])

np.sum(tang_array,axis=1)  #列相加
tang_array.sum(axis = 1)
array([6, 15])
```

```python
tang_array.prod() # 相乘

tang_array.prod(axis = 0)
array([ 4, 10, 18])

tang_array.prod(axis = 1)
array([  6, 120])
```

```python 
tang_array.min()
1
tang_array.min(axis = 0)
array([1, 2, 3])

tang_array.min(axis = 1)
array([1, 4])

tang_array.max()
6
```

#### 找到索引位置

``` python 
tang_array.argmin() # 将返回它扁平化后的最小元素的索引，应该使用np.unravel_index()函数将其还原到二维坐标下。
0

tang_array.argmin(axis = 0)
array([0, 0, 0], dtype=int64)

tang_array.argmin(axis=1)
array([0, 0], dtype=int64)

tang_array.mean() # 求均值
3.5
```

#### 标准差

```python
tang_array.std()
1.707825127659933
tang_array.std(axis = 1)
array([ 0.81649658,  0.81649658])
```

#### 方差的计算

```python
tang_array.var()  # 方差
2.9166666666666665

tang_array
array([[1, 2, 3], [4, 5, 6]])

# tang_array.clip(a_min=None, a_max=None)会返回截取（裁剪）之后的数组
tang_array.clip(2,4)
array([[2, 2, 3],[4, 4, 4]])

tang_array = np.array([1.2,3.56,6.41])
tang_array.round()  # 舍入到指定小数位数的新数组
array([ 1.,  4.,  6.])

tang_array.round(decimals=1)  # 保存指定位数的小数
array([ 1.2,  3.6,  6.4])
```

### 4. 排序

```python 
import numpy as np
tang_array = np.array([[1.5,1.3,7.5],
                      [5.6,7.8,1.2]])

np.sort(tang_array)
array([[ 1.3,  1.5,  7.5],
       [ 1.2,  5.6,  7.8]])

np.sort(tang_array,axis = 0)
array([[ 1.5,  1.3,  1.2],
       [ 5.6,  7.8,  7.5]])

# 将返回一个索引数组，该数组将tang_array数组的元素从小到大排序后的索引值
np.argsort(tang_array)
array([[1, 0, 2],
       [2, 0, 1]], dtype=int64)

# 返回一个开始值为0，结束值为10，含有10个等间距元素的一维数组。
tang_array = np.linspace(0,10,10)
tang_array
array([  0.        ,   1.11111111,   2.22222222,   3.33333333,
         4.44444444,   5.55555556,   6.66666667,   7.77777778,
         8.88888889,  10.        ])

# 在tang_array中，值2.5应插入到索引位置3，值6.5应插入到索引位置6，值9.5应插入到索引位置9。
# 用于在已排序的数组中查找指定值应插入的索引位置
values = np.array([2.5,6.5,9.5])
np.searchsorted(tang_array,values)
array([3, 6, 9], dtype=int64)
```

```python
tang_array = np.array([[1,0,6],
                       [1,7,0],
                       [2,3,1],
                       [2,4,0]])
# lexsort(keys, axis=None)可以按照键的值进行稳定排序，其中keys为排序键，axis为要排序的轴
# 按照第一列降序，第三列升序
# 返回一个数组，其中元素是tang_array的每一行的索引值，这些索引值代表了经过排序后该行所处的位置
index = np.lexsort([-1*tang_array[:,0],tang_array[:,2]])
index
array([0, 1, 3, 2], dtype=int64)

tang_array = tang_array[index] 
tang_array
array([[2, 4, 0],
       [1, 7, 0],
       [2, 3, 1],
       [1, 0, 6]])
```

### 5. 数组形状

```python 
import numpy as np
tang_array = np.arange(10)
tang_array
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

tang_array.shape
(10,)

tang_array.shape = 2,5
tang_array
array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9]])

tang_array.reshape(1,10)
array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
# 大小必须不能改变
```

```python
tang_array = np.arange(10)
tang_array.shape
(10,)
# np.newaxis是在NumPy中广泛使用的一个特殊索引，用于增加数组的维度。
tang_array = tang_array[np.newaxis,:]
tang_array.shape
(1, 10)

tang_array = np.arange(10)
tang_array.shape
(10,)

tang_array = tang_array[:,np.newaxis]
tang_array.shape
(10, 1)

tang_array = tang_array[:,np.newaxis,np.newaxis]
tang_array.shape
(10, 1, 1, 1)

tang_array = tang_array.squeeze() #用于移除多余的单维度
tang_array.shape
(10,)
```

```python
# 用于将数组进行转置操作。在二维情况下，它相当于将数组的行和列进行对调；在多维情况下，它可以实现任意轴的对调。
tang_array.transpose()
tang_array.T
```

#### 数组的连接

```python
a = np.array([[123,456,678],[3214,456,134]])
b = np.array([[1235,3124,432],[43,13,134]])

c = np.concatenate((a,b))
array([[ 123,  456,  678],
       [3214,  456,  134],
       [1235, 3124,  432],
       [  43,   13,  134]])


c = np.concatenate((a,b),axis = 0)
c
array([[ 123,  456,  678],
       [3214,  456,  134],
       [1235, 3124,  432],
       [  43,   13,  134]])

c = np.concatenate((a,b),axis = 1)
c
array([[ 123,  456,  678, 1235, 3124,  432],
       [3214,  456,  134,   43,   13,  134]])
```

```python
np.vstack((a,b)) #表示对这两个数组按垂直方向进行堆叠（竖直方向上堆叠）
array([[ 123,  456,  678],
       [3214,  456,  134],
       [1235, 3124,  432],
       [  43,   13,  134]])

np.hstack((a,b))
array([[ 123,  456,  678, 1235, 3124,  432],
       [3214,  456,  134,   43,   13,  134]])
```

```python
a
array([[ 123,  456,  678],
       [3214,  456,  134]])
# 这两个方法都可以将多维数组降为一维。
a.flatten() # 返回的将是一个拷贝（copy）。
array([ 123,  456,  678, 3214,  456,  134])

a.ravel() # 返回的将是一个视图。
array([ 123,  456,  678, 3214,  456,  134])
```

### 6. 数组生成

```python 
import numpy as np

np.arange(10)
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

np.arange(2,20,2) # 会生成一个从2开始，步长为2到20的一维数组
array([ 2,  4,  6,  8, 10, 12, 14, 16, 18])

np.arange(2,20,2,dtype=np.float32)
array([  2.,   4.,   6.,   8.,  10.,  12.,  14.,  16.,  18.], dtype=float32)

np.linspace(0,10,10)
array([  0.        ,   1.11111111,   2.22222222,   3.33333333,
         4.44444444,   5.55555556,   6.66666667,   7.77777778,
         8.88888889,  10.        ])

# 会生成一个在对数尺度上均匀分布的一维数组 默认是10 
# 起始数：0  终止数：1*10 数组长度：5  均匀分布
np.logspace(0,1,5)
array([  1.        ,   1.77827941,   3.16227766,   5.62341325,  10.        ])
```

```python
x = [1,2]
y = [3,4,5]
# 是用于生成网格点坐标矩阵的函数
# 那么 x 和 y 都会被广播为 n*m 的二维数组。
x, y= np.meshgrid(x,y)
x
[[1, 2],
 [1, 2],
 [1, 2]]

y
[[3, 3],
 [4, 4],
 [5, 5]]
```

```pyton
np.r_[0:10:1] 
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


np.c_[0:10:1]
array([[0],
       [1],
       [2],
       [3],
       [4],
       [5],
       [6],
       [7],
       [8],
       [9]])
```

```python
np.zeros(3)
array([ 0.,  0.,  0.])

np.zeros((3,3))
array([[ 0.,  0.,  0.],
       [ 0.,  0.,  0.],
       [ 0.,  0.,  0.]])
np.ones((3,3))
array([[ 1.,  1.,  1.],
       [ 1.,  1.,  1.],
       [ 1.,  1.,  1.]])
np.ones((3,3)) * 8
array([[ 8.,  8.,  8.],
       [ 8.,  8.,  8.],
       [ 8.,  8.,  8.]])

a = np.empty(6)
(6,)
a.fill(1)
a
array([ 1.,  1.,  1.,  1.,  1.,  1.])

# 清0 和 清1操作
tang_array = np.array([1,2,3,4])
np.zeros_like(tang_array)
array([0, 0, 0, 0])
np.ones_like(tang_array)
array([1, 1, 1, 1])

# 对角矩阵
np.identity(5)
array([[ 1.,  0.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.],
       [ 0.,  0.,  0.,  1.,  0.],
       [ 0.,  0.,  0.,  0.,  1.]])
```



### 7. 运算

```python 
x = np.array([5,5])
y = np.array([2,2])

np.multiply(x,y)
array([10, 10])
```

```python
# 计算两个向量 x 和 y 的点积（也称为内积或标量积）
# 对于两个长度相同的一维数组，点积的结果等于它们的对应元素相乘之和
np.dot(x,y)
20

x.shape = 2,1
array([[5],
       [5]])
y.shape = 1,2
array([[2, 2]])


np.dot(x,y)
array([[10, 10],
       [10, 10]])
np.dot(y,x)
array([[20]])
```

```python
# 与或非
y = np.array([1,1,1,4])
x = np.array([1,1,1,2])
x == y
array([ True,  True,  True, False], dtype=bool)
np.logical_and(x,y)
array([ True,  True,  True,  True], dtype=bool)
np.logical_or(x,y)
array([ True,  True,  True,  True], dtype=bool)
np.logical_not(x,y)
array([0, 0, 0, 0])
```

### 8. 随机模块

```python
import numpy as np
#所有的值都是从0到1
np.random.rand(3,2)
array([[ 0.87876027,  0.98090867],
       [ 0.07482644,  0.08780685],
       [ 0.6974858 ,  0.35695858]])


#返回的是随机的整数，左闭右开
np.random.randint(10,size = (5,4))
array([[8, 0, 3, 7],
       [4, 6, 3, 4],
       [6, 9, 9, 8],
       [9, 1, 4, 0],
       [5, 9, 0, 5]])

np.random.rand()
# 用于生成服从均匀分布的随机数的函数
np.random.random_sample() # 没有参数时和np.random.rand()一样，可以加形状

# 数组长度3，0-9生成随机数
np.random.randint(0,10,3)
array([7, 7, 5])

# 这些随机数都是从一个均值为 mu，标准差为 sigma 的正态分布中生成的。
mu, sigma = 0,0.1
np.random.normal(mu,sigma,10)

# 将设置数组中用于表示小数的位数为 2
np.set_printoptions(precision = 2)
mu, sigma = 0,0.1
np.random.normal(mu,sigma,10)
array([ 0.01,  0.02,  0.12, -0.01, -0.04,  0.07,  0.14, -0.08, -0.01, -0.03])
```

#### 洗牌

```python
tang_array = np.arange(10)
tang_array
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
np.random.shuffle(tang_array)
tang_array
array([6, 2, 5, 7, 4, 3, 1, 0, 8, 9])
```

#### 随机种子

```python
np.random.seed(100) #在同一个随机数种子下使用相同的随机数生成算法和参数，可以得到相同的随机数序列。
mu, sigma = 0,0.1
np.random.normal(mu,sigma,10)
array([-0.17,  0.03,  0.12, -0.03,  0.1 ,  0.05,  0.02, -0.11, -0.02,  0.03])
```

### 9. 读写

```python
%%writefile tang.txt
1 2 3 4 5 6
2 3 5 8 7 9
```

```python
data = []
with open('tang.txt') as f:
    for line in f.readlines():
        fileds = line.split()
        cur_data = [float(x) for x in fileds]
        data.append(cur_data)
data = np.array(data)
data
array([[ 1.,  2.,  3.,  4.,  5.,  6.],
       [ 2.,  3.,  5.,  8.,  7.,  9.]])
```

```python
data = np.loadtxt('tang.txt')
data
array([[ 1.,  2.,  3.,  4.,  5.,  6.],
       [ 2.,  3.,  5.,  8.,  7.,  9.]])

```

```python

%%writefile tang2.txt
1,2,3,4,5,6
2,3,5,8,7,9

data = np.loadtxt('tang2.txt',delimiter = ',')
data
array([[ 1.,  2.,  3.,  4.,  5.,  6.],
       [ 2.,  3.,  5.,  8.,  7.,  9.]])


%%writefile tang2.txt
x,y,z,w,a,b
1,2,3,4,5,6
2,3,5,8,7,9
Overwriting tang2.txt
# skiprows : 去掉几行
# delimiter = ',' :分隔符
# usecols = (0,1,4) ：指定使用哪几列
data = np.loadtxt('tang2.txt',delimiter = ',',skiprows = 1)
data
array([[ 1.,  2.,  3.,  4.,  5.,  6.],
       [ 2.,  3.,  5.,  8.,  7.,  9.]])
```

```python
tang_array = np.array([[1,2,3],[4,5,6]])
np.savetxt('tang4.txt',tang_array)
np.savetxt('tang4.txt',tang_array,fmt='%d')
np.savetxt('tang4.txt',tang_array,fmt='%d',delimiter = ',')
np.savetxt('tang4.txt',tang_array,fmt='%.2f',delimiter = ',')
```

```python
# 读写array结构
tang_array = np.array([[1,2,3],[4,5,6]])
np.save('tang_array.npy',tang_array)

tang = np.load('tang_array.npy')
tang
array([[1, 2, 3],
       [4, 5, 6]])

tang_array2 = np.arange(10)
tang_array2
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# 存两个array
np.savez('tang.npz',a=tang_array,b=tang_array2)
data = np.load('tang.npz')
data.keys()
['b', 'a']
data['a']
array([[1, 2, 3],
       [4, 5, 6]])
data['b']
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

## Pandas

### 1. 概述

```python
import pandas as pd
df = pd.read_csv('')

df.head(5)

df.info() # 返回当前信息
df.index
df.columns
df.dtypes
df.values
```

### 2. 基本操作

```python
df = pd.DataFrame(data)


df['Age'] # 取指定的一列
df['Age'][:5] # 取前五个

df = df.set_index('Name')  # 指定某一列为索引

age = df['Age']
age.mean()
age.min()

df.describe() # 统计  包括样本值、均值、最小、最大、四分位
```

### 3. 索引

```python
df = pd.read_csv('')
df[['Age','Fare']][:5]
 
# iloc 用position1去定位
# loc 用label去定位 df.loc['A':'C' , 's':'d'] 标签行到行  字段列到列
df.iloc[0:5 , 1:3]

df = df.set_index('Name')
df.loc['Laina']
df.loc['Laina':"Alex",'Fare']

df['Fare'] > 40 #bool
df[df['Fare'] > 40]

df.loc[df['sex'] == 'male', 'Age'].mean()
(df['Age'] > 70).sum()
```

### 4. groupby

```python
df.groupby('key').aggregate(np.sum)

df.groupby('sex')['Age'].mean()
```

### 5. 数值运算

```python
df.sum(axis = 0)
df.median # 中位数
```

```python
# 二元统计
df.cov()  #协方差
df.corr() #相关系数  

df['age'].value_counts #在不同年龄段 人数个数
df['age'].value_counts(ascending = True,bins = 5) #会升序 分成五组

df['age'].count() #整体有多少个
```

### 6. 对象操作

```python
data = [10,11,12]
index = ['a','b','c']
s = pd.Series(data = data , index = index)

s1 = s.copy()
s1['a'] = 100
s1.replace(to_replace = 100 , value=101 , inplace=False)
s1.index = ['a','d','b']
s1.rename(index = {'a' = 'A'} , inplace = True)

s1.append(s2 , ignore_index = False) # 不会生成新索引
```

```python
# 删除操作
del s1['A']
s1.drop(['d','b'] , inplace = True)
```

### 7. merge

```python
pd.merge(left , right , on = 'key') # 合并表
pd.merge(left , right , on = ['key1','key2'],how='outer')
```

### 8. 显示设置

```python
pd.get_option('display.max_rows')
60

pd.set_option('display.max_rows',6)
```

### 9. 数据透视表

```python
example.privot(index = 'Category' , colums='Month' , values='Amount') 
```

### 10. 时间操作

```python
import pandas as pd

ts = pd.Timestamp('2017-11-24')
ts.month
ts.day

s = pd.Series(['','',''])
ts = pd.to_datetime(s)

pd.Series(pd.data_range(start='2017-11-24' , periods = 10 . freq = '12H'))
data['time'] = pd.to_datatime(data['Time'])

data.between_time('8:00','12:00')

data.resample('D').mean().head

```

### 11. 字符串操作

```python
import pandas as pd
s Series
s.str.lower()
s.str.len()

index.str.strip()
index.str.lstrip() # 去左边的空格

df = pd.DataFrame(np.random.randn(3,2) , colums )
```

### 数据处理技巧

```python
import panads as pd

g1 = pd.read_csv('')
g1.head

g1.shape
g1.info(字段 = 'deep')

for dtype in ['float64','int64','object']:
    selected = g1.select_dytpe(include = [dtype])
    mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
    mean_usage_md = mean_usage_b/1024**2
    print('平均内存占用' , dtype , mean_usage_mb)
    
import numpy as np
int_types = ['uint8' , 'int8' , 'int16' , 'int32' , 'int64']
for it in int_types:
    print np.iinfo(it)

def mem_usage(pandas_pbj):
    if isinstance(pandas_obj , pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else:
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024**2
    return '{:03.2f} MB' , format(usae_mb)

g1_int = g1_select_dtypes(include = ['int64'])
coverted_int = g1_int.apply(pd.to_numeric , downcast='unsigned')
print(mem_usage(g1_int))
print(mem_usage(coverted_int))

g1_float = g1.select_dtypes(intclude = ['float64'])
converted_float = g1_float.apply(pd.to_numeric , downcast='float')
print(mem_usage(g1_float))
print(mem_usage(coverted_float))

optimized_g1 = g1.copy()
optimized_g1[coverted_int.columns] = coverted_int
optimized_g1[converted_float.columns] = coverted_float
print(g1)
print(optimized_g1)

g1_obj = g1.select_dtypes(include = ['object']).copy()
dow = g1_obj.day_of_week
dow_cat = dow.astype('category')

converted_obj = pd.DataFrame()
for col in g1_obj.columns:
    num_ unique.values = len(g1._obj[co1].unique())
    num_ total_values = len(g1_obj[co1])
    if num_unique_values / num_total_values < 0.5:
    	converted_ obj. 1oc[:,co1] = gl_obj[co1].astype(' category' )
    else:
    	converted_ obj. loc[:,col] = gl._obj[co1]
        
print(mem_ usage(gl_obj))
print(mem_usage (converted_obj)|
751.64 MB
51.67 MB

```

## Matplotlib

### 1. 基本操作

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

plt.plot([1,2,3,4,5],[1,4,9,16,25])
plt.xlabel('xlabel',fontsize = 16)
plt.ylabel('ylabel')

plt.plot([1,2,3,4,5],[1,4,9,16,25],'-.')
plt.xlabel('xlabel',fontsize = 16)
plt.ylabel('ylabel',fontsize = 16)

plt.plot([1,2,3,4,5],[1,4,9,16,25],'-.',color='r')
plt.xlabel('xlabel',fontsize = 16)
plt.ylabel('ylabel',fontsize = 16)
```

| 字符        | 类型       | 字符   | 类型      |
| ----------- | ---------- | ------ | --------- |
| `  '-'	` | 实线       | `'--'` | 虚线      |
| `'-.'`      | 虚点线     | `':'`  | 点线      |
| `'.'`       | 点         | `','`  | 像素点    |
| `'o'`       | 圆点       | `'v'`  | 下三角点  |
| `'^'`       | 上三角点   | `'<'`  | 左三角点  |
| `'>'`       | 右三角点   | `'1'`  | 下三叉点  |
| `'2'`       | 上三叉点   | `'3'`  | 左三叉点  |
| `'4'`       | 右三叉点   | `'s'`  | 正方点    |
| `'p'`       | 五角点     | `'*'`  | 星形点    |
| `'h'`       | 六边形点1  | `'H'`  | 六边形点2 |
| `'+'`       | 加号点     | `'x'`  | 乘号点    |
| `'D'`       | 实心菱形点 | `'d'`  | 瘦菱形点  |
| `'_'`       | 横线点     |        |           |

颜色
表示颜色的字符参数有：

| 字符  | 颜色          |
| ----- | ------------- |
| `‘b’` | 蓝色，blue    |
| `‘g’` | 绿色，green   |
| `‘r’` | 红色，red     |
| `‘c’` | 青色，cyan    |
| `‘m’` | 品红，magenta |
| `‘y’` | 黄色，yellow  |
| `‘k’` | 黑色，black   |
| `‘w’` | 白色，white   |

### 2.  风格设置



### 3. 条形图

```python
import numpy as np
import matplotlib
matplotlib.use('nbagg')
import matplotlib.pyplot as plt

np.random.seed(0)
x = np.arange(5)
y = np.random.randint(-5,5,5)
print (y)
fig,axes = plt.subplots(ncols = 2)
v_bars = axes[0].bar(x,y,color='red') # 横着画
h_bars = axes[1].barh(x,y,color='red')# 竖着画

axes[0].axhline(0,color='grey',linewidth=2) # 在0处加条线
axes[1].axvline(0,color='grey',linewidth=2)
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/matplotlib/直方图1.png)

```python
fig,ax = plt.subplots()
v_bars = ax.bar(x,y,color='lightblue')
for bar,height in zip(v_bars,y):  # 大于0一个颜色 ，小于0一个颜色
    if height < 0:
        bar.set(edgecolor = 'darkred',color = 'green',linewidth = 3)
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/matplotlib/直方图2.png)

```python
x = np.linspace(0,10,200)
y1 = 2*x +1
y2 = 3*x +1.2
y_mean = 0.5*x*np.cos(2*x) + 2.5*x +1.1
fig,ax = plt.subplots()
ax.fill_between(x,y1,y2,color='red')
ax.plot(x,y_mean,color='black')
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/matplotlib/直方图3.png)

```python
mean_values = [1,2,3]
variance = [0.2,0.4,0.5]
bar_label = ['bar1','bar2','bar3']

x_pos = list(range(len(bar_label)))
plt.bar(x_pos,mean_values,yerr=variance,alpha=0.3)
max_y = max(zip(mean_values,variance))
plt.ylim([0,(max_y[0]+max_y[1])*1.2])
plt.ylabel('variable y')
plt.xticks(x_pos,bar_label)
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/matplotlib/直方图4.png)

```python
x1 = np.array([1,2,3])
x2 = np.array([2,2,3])

bar_labels = ['bat1','bar2','bar3']
fig = plt.figure(figsize = (8,6))
y_pos = np.arange(len(x1))
y_pos = [x for x in y_pos]

plt.barh(y_pos,x1,color='g',alpha = 0.5)
plt.barh(y_pos,-x2,color='b',alpha = 0.5)

plt.xlim(-max(x2)-1,max(x1)+1)
plt.ylim(-1,len(x1)+1)
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/matplotlib/直方图5.png)

```python
green_data = [1, 2, 3]
blue_data = [3, 2, 1]
red_data = [2, 3, 3]
labels = ['group 1', 'group 2', 'group 3']

pos = list(range(len(green_data))) 
width = 0.2 
fig, ax = plt.subplots(figsize=(8,6))

plt.bar(pos,green_data,width,alpha=0.5,color='g',label=labels[0])
plt.bar([p+width for p in pos],blue_data,width,alpha=0.5,color='b',label=labels[1])
plt.bar([p+width*2 for p in pos],red_data,width,alpha=0.5,color='r',label=labels[2])
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/matplotlib/直方图6.png)

```python
data = range(200, 225, 5)

bar_labels = ['a', 'b', 'c', 'd', 'e']
fig = plt.figure(figsize=(10,8))
y_pos = np.arange(len(data))

plt.yticks(y_pos, bar_labels, fontsize=16)
bars = plt.barh(y_pos,data,alpha = 0.5,color='g')
plt.vlines(min(data),-1,len(data)+0.5,linestyle = 'dashed')
for b,d in zip(bars,data):
    plt.text(b.get_width()+b.get_width()*0.05,b.get_y()+b.get_height()/2,'{0:.2%}'.format(d/min(data)))
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/matplotlib/直方图7.png)

```python
mean_values = range(10,18)
x_pos = range(len(mean_values))

import matplotlib.colors as col
import matplotlib.cm as cm

cmap1 = cm.ScalarMappable(col.Normalize(min(mean_values),max(mean_values),cm.hot))
cmap2 = cm.ScalarMappable(col.Normalize(0,20,cm.hot))

plt.subplot(121)
plt.bar(x_pos,mean_values,color = cmap1.to_rgba(mean_values))

plt.subplot(122)
plt.bar(x_pos,mean_values,color = cmap2.to_rgba(mean_values))
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/matplotlib/直方图8.png)

```python
patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')

fig = plt.gca()

mean_value = range(1,len(patterns)+1)
x_pos = list(range(len(mean_value)))

bars = plt.bar(x_pos,mean_value,color='white')

for bar,pattern in zip(bars,patterns):
    bar.set_hatch(pattern)
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/matplotlib/直方图9.png)

### 4. 盒图

```python
import matplotlib.pyplot as plt
import numpy as np

tang_data = [np.random.normal(0,std,100) for std in range(1,4)]
fig = plt.figure(figsize = (8,6))
plt.boxplot(tang_data,notch=False,sym='s',vert=True)

plt.xticks([y+1 for y in range(len(tang_data))],['x1','x2','x3'])
plt.xlabel('x')
plt.title('box plot')
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/matplotlib/盒图1.png)

```python
tang_data = [np.random.normal(0,std,100) for std in range(1,4)]
fig = plt.figure(figsize = (8,6))
bplot = plt.boxplot(tang_data,notch=False,sym='s',vert=True)

plt.xticks([y+1 for y in range(len(tang_data))],['x1','x2','x3'])
plt.xlabel('x')
plt.title('box plot')

for components in bplot.keys():
    for line in bplot[components]:
        line.set_color('black')
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/matplotlib/盒图2.png)

```python
tang_data = [np.random.normal(0,std,100) for std in range(1,4)]
fig = plt.figure(figsize = (8,6))
plt.boxplot(tang_data,notch=False,sym='s',vert=False)

plt.yticks([y+1 for y in range(len(tang_data))],['x1','x2','x3'])
plt.ylabel('x')
plt.title('box plot')
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/matplotlib/盒图3.png)

```python
tang_data = [np.random.normal(0,std,100) for std in range(1,4)]
fig = plt.figure(figsize = (8,6))
plt.boxplot(tang_data,notch=True,sym='s',vert=False)

plt.xticks([y+1 for y in range(len(tang_data))],['x1','x2','x3'])
plt.xlabel('x')
plt.title('box plot')
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/matplotlib/盒图4.png)

```python
tang_data = [np.random.normal(0,std,100) for std in range(1,4)]
fig = plt.figure(figsize = (8,6))
bplot = plt.boxplot(tang_data,notch=False,sym='s',vert=True,patch_artist=True)

plt.xticks([y+1 for y in range(len(tang_data))],['x1','x2','x3'])
plt.xlabel('x')
plt.title('box plot')

colors = ['pink','lightblue','lightgreen']
for pathch,color in zip(bplot['boxes'],colors):
    pathch.set_facecolor(color)
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/matplotlib/盒图5.png)

```python
fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(12,5))
tang_data = [np.random.normal(0,std,100) for std in range(6,10)]
axes[0].violinplot(tang_data,showmeans=False,showmedians=True)
axes[0].set_title('violin plot')

axes[1].boxplot(tang_data)
axes[1].set_title('box plot')

for ax in axes:
    ax.yaxis.grid(True)
    ax.set_xticks([y+1 for y in range(len(tang_data))])
plt.setp(axes,xticks=[y+1 for y in range(len(tang_data))],xticklabels=['x1','x2','x3','x4'])
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/matplotlib/盒图6.png)

### 5. 直方图

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.random.normal(0,20,1000)
bins = np.arange(-100,100,5)

plt.hist(data,bins=bins)
plt.xlim([min(data)-5,max(data)+5])
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/matplotlib/直方图10.png)

```python
import random
data1 = [random.gauss(15,10) for i in range(500)]
data2 = [random.gauss(5,5) for i in range(500)]
bins = np.arange(-50,50,2.5)

plt.hist(data1,bins=bins,label='class 1',alpha = 0.3)
plt.hist(data2,bins=bins,label='class 2',alpha = 0.3)
plt.legend(loc='best')
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/matplotlib/直方图11.png)

### 6. 散点图

```python
mu_vec1 = np.array([0,0])
cov_mat1 = np.array([[2,0],[0,2]])

x1_samples = np.random.multivariate_normal(mu_vec1, cov_mat1, 100)
x2_samples = np.random.multivariate_normal(mu_vec1+0.2, cov_mat1+0.2, 100)
x3_samples = np.random.multivariate_normal(mu_vec1+0.4, cov_mat1+0.4, 100)

plt.figure(figsize = (8,6))
plt.scatter(x1_samples[:,0],x1_samples[:,1],marker ='x',color='blue',alpha=0.6,label='x1')
plt.scatter(x2_samples[:,0],x2_samples[:,1],marker ='o',color='red',alpha=0.6,label='x2')
plt.scatter(x3_samples[:,0],x3_samples[:,1],marker ='^',color='green',alpha=0.6,label='x3')
plt.legend(loc='best')
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/matplotlib/散点图1.png)

```python
x_coords = [0.13, 0.22, 0.39, 0.59, 0.68, 0.74, 0.93]
y_coords = [0.75, 0.34, 0.44, 0.52, 0.80, 0.25, 0.55]

plt.figure(figsize = (8,6))
plt.scatter(x_coords,y_coords,marker='s',s=50)

for x,y in zip(x_coords,y_coords):
    plt.annotate('(%s,%s)'%(x,y),xy=(x,y),xytext=(0,-15),textcoords = 'offset points',ha='center')
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/matplotlib/散点图2.png)

```python
mu_vec1 = np.array([0,0])
cov_mat1 = np.array([[1,0],[0,1]])
X = np.random.multivariate_normal(mu_vec1, cov_mat1, 500)
fig = plt.figure(figsize=(8,6))

R=X**2
R_sum=R.sum(axis = 1)

plt.scatter(X[:,0],X[:,1],color='grey',marker='o',s=20*R_sum,alpha=0.5)
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/matplotlib/散点图3.png)

### 7. 3D图

```python
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)

x = np.arange(-4,4,0.25)
y = np.arange(-4,4,0.25)

X,Y = np.meshgrid(x,y)

Z = np.sin(np.sqrt(X**2+Y**2))
ax.plot_surface(X,Y,Z,rstride = 1,cstride = 1,cmap='rainbow')
ax.contour(X,Y,Z,zdim='z',offset = -2 ,cmap='rainbow')

ax.set_zlim(-2,2)
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/matplotlib/3d图.png)

```python
fig = plt.figure()
ax = fig.gca(projection='3d')

theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)

ax.plot(x,y,z)
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/matplotlib/3d图2.png)

```python
np.random.seed(1)
def randrange(n,vmin,vmax):
    return (vmax-vmin)*np.random.rand(n)+vmin


fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')
n = 100
for c,m,zlow,zhigh in [('r','o',-50,-25),('b','x','-30','-5')]:
    xs = randrange(n,23,32)
    ys = randrange(n,0,100)
    zs = randrange(n,int(zlow),int(zhigh))
    ax.scatter(xs,ys,zs,c=c,marker=m)

ax.view_init(40,0) # 可以换方向    
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/matplotlib/3d图3.png)

```python
fig = plt.figure()  
ax = fig.add_subplot(111, projection='3d') 

for c, z in zip(['r', 'g', 'b', 'y'], [30, 20, 10, 0]): 
    xs = np.arange(20)
    ys = np.random.rand(20)
    cs = [c]*len(xs)
    ax.bar(xs,ys,zs = z,zdir='y',color = cs,alpha = 0.5)
plt.show()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/matplotlib/3d图4.png)

### 8. pie与子图

```python
m = 51212.
f = 40742.
m_perc = m/(m+f)
f_perc = f/(m+f)

colors = ['navy','lightcoral']
labels = ["Male","Female"]

plt.figure(figsize=(8,8))
paches,texts,autotexts = plt.pie([m_perc,f_perc],labels = labels,autopct = '%1.1f%%',explode=[0,0.05],colors = colors)

for text in texts+autotexts:
    text.set_fontsize(20)
for text in autotexts:
    text.set_color('white')
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/matplotlib/pie.png)

```python
ax1 = plt.subplot2grid((3,3),(0,0))
ax2 = plt.subplot2grid((3,3),(1,0))
ax3 = plt.subplot2grid((3,3),(0,2),rowspan=3)
ax4 = plt.subplot2grid((3,3),(2,0),colspan = 2)
ax5 = plt.subplot2grid((3,3),(0,1),rowspan=2)
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/matplotlib/子图1.png)

```python
import numpy as np

x = np.linspace(0,10,1000)
y2 = np.sin(x**2)
y1 = x**2

fig,ax1 = plt.subplots()

left,bottom,width,height = [0.22,0.45,0.3,0.35]
ax2 = fig.add_axes([left,bottom,width,height])

ax1.plot(x,y1)
ax2.plot(x,y2)
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/matplotlib/子图2.png)

```python
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2., 1.02*height,
        "{:,}".format(float(height)),
        ha='center', va='bottom',fontsize=18)
        
top10_arrivals_countries = ['CANADA','MEXICO','UNITED\nKINGDOM',\
                            'JAPAN','CHINA','GERMANY','SOUTH\nKOREA',\
                            'FRANCE','BRAZIL','AUSTRALIA']
top10_arrivals_values = [16.625687, 15.378026, 3.934508, 2.999718,\
                         2.618737, 1.769498, 1.628563, 1.419409,\
                         1.393710, 1.136974]
arrivals_countries = ['WESTERN\nEUROPE','ASIA','SOUTH\nAMERICA',\
                      'OCEANIA','CARIBBEAN','MIDDLE\nEAST',\
                      'CENTRAL\nAMERICA','EASTERN\nEUROPE','AFRICA']
arrivals_percent = [36.9,30.4,13.8,4.4,4.0,3.6,2.9,2.6,1.5]

fig, ax1 = plt.subplots(figsize=(20,12))
tang = ax1.bar(range(10),top10_arrivals_values,color='blue')
plt.xticks(range(10),top10_arrivals_countries,fontsize=18)
ax2 = inset_axes(ax1,width = 6,height = 6,loc = 5)
explode = (0.08, 0.08, 0.05, 0.05,0.05,0.05,0.05,0.05,0.05)
patches, texts, autotexts = ax2.pie(arrivals_percent,labels=arrivals_countries,autopct='%1.1f%%',explode=explode)

for text in texts+autotexts:
    text.set_fontsize(16)
for spine in ax1.spines.values():
    spine.set_visible(False)

autolabel(tang) 
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/matplotlib/子图3.png)

## Seaborn

### 1. 基础风格

```python
import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
def sinplot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)
```

```
5种主题风格
* darkgrid
* whitegrid
* dark
* white
* ticks
```

```python
sns.set_style("whitegrid")
data = np.random.normal(size=(20, 6)) + np.arange(6) / 2
sns.boxplot(data=data)
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/seaborn/风格1.png)

```python
sns.set_style("dark")
sinplot()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/seaborn/风格2.png)

```python
sns.set_style("white")
sinplot()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/seaborn/风格3.png)

```python
sns.set_style("ticks")
sinplot()
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/seaborn/风格4.png)

```python
sinplot()
sns.despine() # 不要上框线和右框线
sns.despine(offset=10) # 设置图距离下面的距离
sns.despine(left=True) # 不要左框线

```

```python
# 上下两个图  用两个不同的主题
with sns.axes_style("darkgrid"):
    plt.subplot(211)
    sinplot()
plt.subplot(212)
sinplot(-1)
```

```python
sns.set_context("paper")
plt.figure(figsize=(8, 6))
sinplot()

sns.set_context("talk")

sns.set_context("poster")

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5}) # 这个还行
```

### 2. 调色

* 颜色很重要
* color_palette()能传入任何Matplotlib所支持的颜色
* color_palette()不写参数则默认颜色
* set_palette()设置所有图的颜色

```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(rc={"figure.figsize": (6, 6)})

current_palette = sns.color_palette()
sns.palplot(current_palette)
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/seaborn/调色1.png)

6个默认的颜色循环主题： deep, muted, pastel, bright, dark, colorblind

#### 圆形画板

当你有六个以上的分类要区分时，最简单的方法就是在一个圆形的颜色空间中画出均匀间隔的颜色(这样的色调会保持亮度和饱和度不变)。这是大多数的当他们需要使用比当前默认颜色循环中设置的颜色更多时的默认方案。

最常用的方法是使用hls的颜色空间，这是RGB值的一个简单转换。

```python
sns.palplot(sns.color_palette("hls", 8))
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/seaborn/调色2.png)

```python
data = np.random.normal(size=(20, 8)) + np.arange(8) / 2
sns.boxplot(data=data,palette=sns.color_palette("hls", 8))
```

```
hls_palette()函数来控制颜色的亮度和饱和
* l-亮度 lightness 
* s-饱和 saturation
```

```python
sns.palplot(sns.hls_palette(8, l=.7, s=.9))
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/seaborn/调色3.png)

```python
sns.palplot(sns.color_palette("Paired",8))
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/seaborn/调色4.png)

####  使用xkcd颜色来命名颜色 ###
xkcd包含了一套众包努力的针对随机RGB色的命名。产生了954个可以随时通过xdcd_rgb字典中调用的命名颜色。

```python
plt.plot([0, 1], [0, 1], sns.xkcd_rgb["pale red"], lw=3)
plt.plot([0, 1], [0, 2], sns.xkcd_rgb["medium green"], lw=3)
plt.plot([0, 1], [0, 3], sns.xkcd_rgb["denim blue"], lw=3)
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/seaborn/调色5.png)

```python
colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
sns.palplot(sns.xkcd_palette(colors))
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/seaborn/调色6.png)

####  连续色板
色彩随数据变换，比如数据越来越重要则颜色越来越深

```python
sns.palplot(sns.color_palette("Blues"))
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/seaborn/调色7.png)

```python
# 如果想要翻转渐变，可以在面板名称中添加一个_r后缀
sns.palplot(sns.color_palette("BuGn_r"))
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/seaborn/调色8.png)

#### cubehelix_palette()调色板
色调线性变换

```python
sns.palplot(sns.color_palette("cubehelix", 8))
sns.palplot(sns.cubehelix_palette(8, start=.5, rot=-.75))
sns.palplot(sns.cubehelix_palette(8, start=.75, rot=-.150))
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/seaborn/调色9.png)

####  light_palette() 和dark_palette()调用定制连续调色板

```python
sns.palplot(sns.light_palette("green"))
sns.palplot(sns.dark_palette("purple"))
sns.palplot(sns.light_palette("navy", reverse=True))
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/seaborn/调色10.png)

```python
sns.palplot(sns.light_palette((210, 90, 60), input="husl"))
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/seaborn/调色11.png)

### 3. 单变量分析绘图

```python
%matplotlib inline
import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(color_codes=True)
np.random.seed(sum(map(ord, "distributions")))
```

```python
x = np.random.normal(size=100)
sns.distplot(x,kde=False)

sns.distplot(x, bins=20, kde=False)
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/seaborn/单变量分析1.png)

```python
x = np.random.gamma(6, size=200)
sns.distplot(x, kde=False, fit=stats.gamma)
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/seaborn/单变量分析2.png)

```python
# 观测两个变量之间的分布关系最好用散点图
sns.jointplot(x="x", y="y", data=df);
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/seaborn/单变量分析3.png)

```python
x, y = np.random.multivariate_normal(mean, cov, 1000).T
with sns.axes_style("white"):
    sns.jointplot(x=x, y=y, kind="hex", color="k")
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/seaborn/单变量分析4.png)

```python
iris = sns.load_dataset("iris")
sns.pairplot(iris)
# 有4个特征  观察特征之间的关系
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/seaborn/单变量分析5.png)

### 4. REG

```PYTHON
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(color_codes=True)

np.random.seed(sum(map(ord, "regression")))

tips = sns.load_dataset("tips")
```

```python
# regplot()和lmplot()都可以绘制回归关系,推荐regplot()
sns.regplot(x="total_bill", y="tip", data=tips)
sns.lmplot(x="total_bill", y="tip", data=tips);

sns.regplot(data=tips,x="size",y="tip")
sns.regplot(x="size", y="tip", data=tips, x_jitter=.05)

anscombe = sns.load_dataset("anscombe")
sns.regplot(x="x", y="y", data=anscombe.query("dataset == 'I'"),
           ci=None, scatter_kws={"s": 100})

sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'II'"),
           ci=None, scatter_kws={"s": 80})
sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'II'"),
           order=2, ci=None, scatter_kws={"s": 80});

sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips);
sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips,
           markers=["o", "x"], palette="Set1");

sns.lmplot(x="total_bill", y="tip", hue="smoker", col="time", data=tips);
sns.lmplot(x="total_bill", y="tip", hue="smoker",
           col="time", row="sex", data=tips);

f, ax = plt.subplots(figsize=(5, 5))
sns.regplot(x="total_bill", y="tip", data=tips, ax=ax);

sns.lmplot(x="total_bill", y="tip", col="day", data=tips,
           col_wrap=2, size=4);

sns.lmplot(x="total_bill", y="tip", col="day", data=tips,
           aspect=.8);
```

### 5. category

```python
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)

np.random.seed(sum(map(ord, "categorical")))
titanic = sns.load_dataset("titanic")
tips = sns.load_dataset("tips")
iris = sns.load_dataset("iris")
```

```python
sns.stripplot(x="day", y="total_bill", data=tips);

# 重叠是很常见的现象，但是重叠影响我观察数据的量了
sns.stripplot(x="day", y="total_bill", data=tips, jitter=True)
sns.swarmplot(x="day", y="total_bill", data=tips)
sns.swarmplot(x="day", y="total_bill", hue="sex",data=tips)

sns.swarmplot(x="total_bill", y="day", hue="time", data=tips);
```

```python
sns.boxplot(x="day", y="total_bill", hue="time", data=tips);
sns.violinplot(x="total_bill", y="day", hue="time", data=tips);

sns.violinplot(x="day", y="total_bill", hue="sex", data=tips, split=True);
sns.violinplot(x="day", y="total_bill", data=tips, inner=None)
sns.swarmplot(x="day", y="total_bill", data=tips, color="w", alpha=.5)

sns.barplot(x="sex", y="survived", hue="class", data=titanic);
sns.pointplot(x="sex", y="survived", hue="class", data=titanic);

sns.pointplot(x="class", y="survived", hue="sex", data=titanic,
              palette={"male": "g", "female": "m"},
              markers=["^", "o"], linestyles=["-", "--"]);

sns.boxplot(data=iris,orient="h");
sns.factorplot(x="day", y="total_bill", hue="smoker", data=tips)
sns.factorplot(x="day", y="total_bill", hue="smoker", data=tips, kind="bar")
sns.factorplot(x="day", y="total_bill", hue="smoker",
               col="time", data=tips, kind="swarm")

sns.factorplot(x="time", y="total_bill", hue="smoker",
               col="day", data=tips, kind="box", size=4, aspect=.5)
```

### 6.  FacetGrid

```python
%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt

sns.set(style="ticks")
np.random.seed(sum(map(ord, "axis_grids")))
```

```python
tips = sns.load_dataset("tips")
total_bill	tip	sex	smoker	day	  time	size

g = sns.FacetGrid(tips, col="time")
g.map(plt.hist, "tip");
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/seaborn/facet1.png)

```python
g = sns.FacetGrid(tips, col="sex", hue="smoker")
g.map(plt.scatter, "total_bill", "tip", alpha=.7)
g.add_legend();
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/seaborn/facet2.png)

```python
g = sns.FacetGrid(tips, row="smoker", col="time", margin_titles=True)
g.map(sns.regplot, "size", "total_bill", color=".1", fit_reg=False, x_jitter=.1);
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/seaborn/facet3.png)

```python
from pandas import Categorical
ordered_days = tips.day.value_counts().index
print (ordered_days)
ordered_days = Categorical(['Thur', 'Fri', 'Sat', 'Sun'])
g = sns.FacetGrid(tips, row="day", row_order=ordered_days,
                  size=1.7, aspect=4,)
g.map(sns.boxplot, "total_bill");
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/seaborn/facet4.png)

```python
pal = dict(Lunch="seagreen", Dinner="gray")
g = sns.FacetGrid(tips, hue="time", palette=pal, size=5)
g.map(plt.scatter, "total_bill", "tip", s=50, alpha=.7, linewidth=.5, edgecolor="white")
g.add_legend();
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/seaborn/facet5.png)

```python
g = sns.FacetGrid(tips, hue="sex", palette="Set1", size=5, hue_kws={"marker": ["^", "v"]})
g.map(plt.scatter, "total_bill", "tip", s=100, linewidth=.5, edgecolor="white")
g.add_legend();
```

![](https://cdn.staticaly.com/gh/clint-sfy/blogcdn@master/python/seaborn/facet6.png)

```python
iris = sns.load_dataset("iris")
g = sns.PairGrid(iris)
g.map(plt.scatter);

g = sns.PairGrid(iris)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter);

g = sns.PairGrid(iris, hue="species")
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend();

g = sns.PairGrid(iris, vars=["sepal_length", "sepal_width"], hue="species")
g.map(plt.scatter);
```

### 7. HeatMap

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np; 
np.random.seed(0)
import seaborn as sns;
sns.set()

uniform_data = np.random.rand(3, 3)
print (uniform_data)
heatmap = sns.heatmap(uniform_data)
```

```python
flights = sns.load_dataset("flights")
flights.head()

flights = flights.pivot("month", "year", "passengers")
print (flights)
ax = sns.heatmap(flights)

ax = sns.heatmap(flights, annot=True,fmt="d") # 显示数字
ax = sns.heatmap(flights, linewidths=.5)

ax = sns.heatmap(flights, cmap="YlGnBu")
ax = sns.heatmap(flights, cbar=False)
```

