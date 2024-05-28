# numpy基础

## 1-使用numpy构建基本函数

### 1.1-sigmoid function和np.exp（）

​	在使用np.exp（）之前，你将使用math.exp（）实现Sigmoid函数。然后，你将知道为什么np.exp（）比math.exp（）更可取。

**练习**：构建一个返回实数x的sigmoid的函数。将math.exp（x）用于指数函数。

![image-20240523132329935](images/image-20240523132329935.png)

​	也称为逻辑函数。一种非线性函数，即可用于机器学习（逻辑回归），也能用于深度学习。

​	深度学习中主要使用的是矩阵和向量，因此numpy更为实用，对于使用math写的sigmid函数，如果传入向量或者矩阵会报类型不匹配的错误，因此使用np.exp()。

​	如果![image-20240523133219046](images/image-20240523133219046.png)是行向量，则![image-20240523133230569](images/image-20240523133230569.png)会将指数函数应用于x的每个元素。输出为：![image-20240523133247952](images/image-20240523133247952.png)

```python
import math 
import numpy as np

# example of np.exp
x = np.array([1, 2, 3])
print(np.exp(x)) # result is (exp(1), exp(2), exp(3))
```

output:

```python
[ 2.71828183  7.3890561  20.08553692]
```

如果x是向量，则𝑠=𝑥+3或![image-20240523133848886](images/image-20240523133848886.png)之类的Python运算将输出与x维度大小相同的向量s。

```PY
x = np.array([1, 2, 3])
print (x + 3)
```

output：

```python
[4 5 6]
```

综上，对于np.exp()所实现的sigmoid函数，可以输入矩阵，向量等，输出输入如下所示。

![image-20240523134217046](images/image-20240523134217046.png)

```python
def sigmoid(x):

    s = 1 / (1 + np.exp(-x))
    
    return s
```



### 1.2- Sigmoid gradient

**练习**：创建函数sigmoid_grad（）计算sigmoid函数相对于其输入x的梯度。 公式为：

​	此处对应于二分法，x一般为w^T*x+b，使用梯度下降法的时候会从成本函数一步步向前求偏导，以找到成本函数最小的值与点，合理利用链式法则以求得相应的导数。

![image-20240523134357263](images/image-20240523134357263.png)

```python
def sigmoid_derivative(x):
    s = sigmoid(x)
    ds = s * (1-s)
    return ds

x = np.array([1, 2, 3])
print ("sigmoid_derivative(x) = " + str(sigmoid_derivative(x)))
```

output:

```PYTHON
sigmoid_derivative(x) = [0.19661193 0.10499359 0.04517666]
```

### 1.3- 重塑数组

深度学习中两个常用的numpy函数是np.shape()和np.reshape()。
-X.shape用于获取矩阵/向量X的shape（维度）。
-X.reshape（...）用于将X重塑为其他尺寸。

例如，在计算机科学中，图像由shape为(𝑙𝑒𝑛𝑔𝑡ℎ,ℎ𝑒𝑖𝑔ℎ𝑡,𝑑𝑒𝑝𝑡ℎ=3)的3D数组表示，（长，宽，RGB=3）但是，当你读取图像作为算法的输入时，会将其转换为维度为(𝑙𝑒𝑛𝑔𝑡ℎ∗ℎ𝑒𝑖𝑔ℎ𝑡∗3,1)的向量。换句话说，将3D阵列“展开”或重塑为1D向量。

![image-20240523135829704](images/image-20240523135829704.png)

**练习**：实现`image2vector()` ,该输入采用维度为(length, height, 3)的输入，并返回维度为(length\*height\*3 , 1)的向量。例如，如果你想将形为（a，b，c）的数组v重塑为维度为(a*b, 3)的向量，则可以执行以下操作：

```python
v = v.reshape((v.shape[0]*v.shape[1], v.shape[2])) # v.shape[0] = a ; v.shape[1] = b ; v.shape[2] = c
```

-请不要将图像的尺寸硬编码为常数。而是通过image.shape [0]等来查找所需的数量。

```python
def image2vector(image):

    v = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)

    return v



image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])

print ("image2vector(image) = " + str(image2vector(image)))
```

output:

```python
image2vector(image) = [[0.67826139]
 [0.29380381]
 [0.90714982]
 [0.52835647]
 [0.4215251 ]
 [0.45017551]
 [0.92814219]
 [0.96677647]
 [0.85304703]
 [0.52351845]
 [0.19981397]
 [0.27417313]
 [0.60659855]
 [0.00533165]
 [0.10820313]
 [0.49978937]
 [0.34144279]
 [0.94630077]]
```

### 1.4-行标准化

​	对数据进行标准化（归一化）。 由于归一化后梯度下降的收敛速度更快，通常会表现出更好的效果。 通过归一化，也就是将x更改为![image-20240523141832233](images/image-20240523141832233.png)（将x的每个行向量除以其范数（模值）)。

例如：

![image-20240523141947128](images/image-20240523141947128.png)

then

![image-20240523141954325](images/image-20240523141954325.png)

and

![image-20240523141958558](images/image-20240523141958558.png)

**练习**：执行 normalizeRows（）来标准化矩阵的行。 将此函数应用于输入矩阵x之后，x的每一行应为单位长度（即长度为1）向量。

```python
#linalg = linear（线性）+ algebra（代数），norm则表示范数。
#x_norm=np.linalg.norm(x, ord=None, axis=None, keepdims=False)
```

- x: 表示矩阵（也可以是一维）

- ord：范数类型

  ![image-20240523142909611](images/image-20240523142909611.png)

- axis, axis=0 表示按列向量来进行处理，求多个列向量的范数; axis =1 表示按行向量来进行处理，求多个行向量的范数

- keepdims：表示是否保持矩阵的二维特性，True表示保持，False表示不保持，默认为False

```python
def normalizeRows(x):
    x_norm = np.linalg.norm(x, axis = 1, keepdims = True)
    # Divide x by its norm.
    x = x / x_norm
    ### END CODE HERE ###

    return x

x = np.array([
    [0, 3, 4],
    [1, 6, 4]])
print("normalizeRows(x) = " + str(normalizeRows(x)))
```

output：

```PYTHON
normalizeRows(x) = [[0.         0.6        0.8       ]
 [0.13736056 0.82416338 0.54944226]]
```

**注意**：
在normalizeRows（）中，尝试print查看 x_norm和x的维度， 会发现它们具有不同的w维度。 鉴于x_norm采用x的每一行的范数，这是正常的。 因此，x_norm具有相同的行数，但只有1列。 那么，当你将x除以x_norm时，它是如何工作的？ 这就是所谓的广播broadcasting，我们现在将讨论它！

此处对应的是笔记中的广播，通俗来讲就是会智能化解决维度不匹配的问题。

### 1.5- 广播和softmax函数

在numpy中要理解的一个非常重要的概念是“广播”。 这对于在不同形状的数组之间执行数学运算非常有用。

**练习**: 使用numpy实现softmax函数。 你可以将softmax理解为算法需要对两个或多个类进行分类时使用的标准化函数。

​	简而言之，输入一个矩阵，取e指数后每行进行单位化，对于这个函数有什么作用将在本专业的第二门课中了解有关softmax的更多信息。

![image-20240523144022058](images/image-20240523144022058.png)

```python
def softmax(x):

    x_exp = np.exp(x)

    x_sum = np.sum(x_exp, axis = 1, keepdims = True)

    s = x_exp / x_sum
    
    return s


x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])
print("softmax(x) = " + str(softmax(x)))
```

output:

```PYTHON
softmax(x) = [[9.80897665e-01 8.94462891e-04 1.79657674e-02 1.21052389e-04
  1.21052389e-04]
 [8.78679856e-01 1.18916387e-01 8.01252314e-04 8.01252314e-04
  8.01252314e-04]]
```

## 2-向量化

​	在深度学习中，通常需要处理非常大的数据集。 因此，非计算最佳函数可能会成为算法中的巨大瓶颈，并可能使模型运行一段时间。 为了确保代码的高效计算，我们将使用向量化。 例如，尝试区分点/外部/元素乘积之间的区别。

​	下方运算时间为0，dic、doc相同，电脑运算太快，代码太少，format函数精确到后八位都是0。

​	但大运算量避免使用显示for，使用np函数库。

### **点积(dot product)**

又叫标量积、数量积。

​	![image-20240523163946065](images/image-20240523163946065.png)

for版本：

```python
import time

x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

### CLASSIC DOT PRODUCT OF VECTORS IMPLEMENTATION ###
tic = time.process_time()
dot = 0
for i in range(len(x1)):
    dot+= x1[i]*x2[i]
toc = time.process_time()
print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")
```

np版本：

```PYTHON
x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

### VECTORIZED DOT PRODUCT OF VECTORS ###
tic = time.process_time()
dot = np.dot(x1,x2)
toc = time.process_time()
print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")
```



output:

```PYTHON
dot = 278
 ----- Computation time = 0.0ms
```



### **叉积(corss product)**

​	或向量积(vector product )

![image-20240523164123958](images/image-20240523164123958.png)

### **外积(outer product)**

在线性代数中，两个坐标向量的外积是一个矩阵。如果这两个向量的维度是*n*和*m*，那么它们的外积是一个*n* × *m*矩阵。更一般地说，给定两个张量（多维数字数组），它们的外积是张量。张量的外积也称为张量积，可用于定义张量代数。

![image-20240523164405819](images/image-20240523164405819.png)

for版本:

```PYTHON
### CLASSIC OUTER PRODUCT IMPLEMENTATION ###
tic = time.process_time()
outer = np.zeros((len(x1),len(x2))) # we create a len(x1)*len(x2) matrix with only zeros
for i in range(len(x1)):
    for j in range(len(x2)):
        outer[i,j] = x1[i]*x2[j]
toc = time.process_time()
print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")
```

np版本：

```PYTHON
### VECTORIZED OUTER PRODUCT ###
tic = time.process_time()
outer = np.outer(x1,x2)
toc = time.process_time()
print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")
```



output:

```PYTHON
outer = [[81. 18. 18. 81.  0. 81. 18. 45.  0.  0. 81. 18. 45.  0.  0.]
 [18.  4.  4. 18.  0. 18.  4. 10.  0.  0. 18.  4. 10.  0.  0.]
 [45. 10. 10. 45.  0. 45. 10. 25.  0.  0. 45. 10. 25.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [63. 14. 14. 63.  0. 63. 14. 35.  0.  0. 63. 14. 35.  0.  0.]
 [45. 10. 10. 45.  0. 45. 10. 25.  0.  0. 45. 10. 25.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [81. 18. 18. 81.  0. 81. 18. 45.  0.  0. 81. 18. 45.  0.  0.]
 [18.  4.  4. 18.  0. 18.  4. 10.  0.  0. 18.  4. 10.  0.  0.]
 [45. 10. 10. 45.  0. 45. 10. 25.  0.  0. 45. 10. 25.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]
 ----- Computation time = 0.0ms
```

​	不同于`np.multiply()`和`*` 操作符（相当于Matlab / Octave中的 `.*`）执行逐元素的乘法，`np.dot()`执行的是矩阵-矩阵或矩阵向量乘法，

### 2.1-实现L1和L2损失函数

**练习**：实现L1损失函数的Numpy向量化版本。 我们会发现函数abs（x）（x的绝对值）很有用。

**提示**：
-损失函数用于评估模型的性能。 损失越大，预测![image-20240523170837643](images/image-20240523170837643.png) 与真实值![image-20240523170844510](images/image-20240523170844510.png)的差异也就越大。 在深度学习中，我们使用诸如Gradient Descent之类的优化算法来训练模型并最大程度地降低成本。

- L1损失函数定义为：

![image-20240523170721050](images/image-20240523170721050.png)

```PYTHON
def L1(yhat, y):
    loss = np.sum(np.abs(y - yhat))
    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat,y)))
```

output:

```python
L1 = 1.1
```

**练习**：实现L2损失函数的Numpy向量化版本。 有好几种方法可以实现L2损失函数，但是还是np.dot（）函数更好用。 提醒一下，如果𝑥=[𝑥1,𝑥2,...,𝑥𝑛]，则`np.dot（x，x）`=![image-20240523171230965](images/image-20240523171230965.png)

- L2损失函数定义为：

![image-20240523171322543](images/image-20240523171322543.png)

此种算法是上述提到的矩阵的点积算法。

```python
def L2(yhat, y):
    loss = np.dot((y - yhat),(y - yhat).T)
    return loss


yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2 = " + str(L2(yhat,y)))
```

output:

```python
L2 = 0.43
```

# 用神经网络思想实现Logistic回归

## 1- 安装包

**你将学习以下内容：**

- 建立学习算法的一般架构，包括：
  - 初始化参数
  - 计算损失函数及其梯度
  - 使用优化算法（梯度下降）
- 按正确的顺序将以上所有三个功能集成到一个主模型上。

```python
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset #自建库，不大很小，网上可找到。
```

## 2- 数据集

**问题说明**：你将获得一个包含以下内容的数据集（"data.h5"）：

-    标记为cat（y = 1）或非cat（y = 0）的**m_train**训练图像集
-    标记为cat或non-cat的**m_test**测试图像集
-    图像维度为（num_px，num_px，3），其中3表示3个通道（RGB）。 因此，每个图像都是正方形（高度= num_px）和（宽度= num_px）。

你将构建一个简单的图像识别算法，该算法可以将图片正确分类为猫和非猫。
让我们熟悉一下数据集吧， 首先通过运行以下代码来加载数据。

```python
# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

print("train_set_x shape: " + str(train_set_x_orig.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x shape: " + str(test_set_x_orig.shape))
print("test_set_y shape: " + str(test_set_y.shape))
```

我们在图像数据集（训练和测试）的末尾添加了"_orig"，以便对其进行预处理。 预处理后，我们将得到train_set_x和test_set_x（标签train_set_y和test_set_y不需要任何预处理）。

- **train_set_x_orig** ：保存的是训练集里面的图像数据（本训练集有209张64x64的图像）。
- **train_set_y_orig** ：保存的是训练集的图像对应的分类值（【0 | 1】，0表示不是猫，1表示是猫）。
- **test_set_x_orig** ：保存的是测试集里面的图像数据（本训练集有50张64x64的图像）。
- **test_set_y_orig** ： 保存的是测试集的图像对应的分类值（【0 | 1】，0表示不是猫，1表示是猫）。
- **classes** ： 保存的是以bytes类型保存的两个字符串数据，数据为：[b’non-cat’ b’cat’]。

train_set_x_orig和test_set_x_orig的每一行都是代表图像的数组。 你可以通过运行以下代码来可视化示例。 还可以随意更改`index`值并重新运行以查看其他图像。

```python
# Example of a picture
index = 5
plt.imshow(train_set_x_orig[index])
plt.show()  #使用vscode时，添加这个才会显示，并非在ipythonnote中运行。
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
```

output：

```python
y = [0], it's a 'non-cat' picture.
```

![image-20240523175255830](images/image-20240523175255830.png)

深度学习中的许多报错都来自于矩阵/向量尺寸不匹配。 如果你可以保持矩阵/向量的尺寸不变，那么将消除大多错误。

**练习：** 查找以下各项的值：

-    m_train（训练集示例数量）
-    m_test（测试集示例数量）
-    num_px（=训练图像的高度=训练图像的宽度）

“ train_set_x_orig”是一个维度为（m_train，num_px，num_px，3）的numpy数组。

```python
### START CODE HERE ### (≈ 3 lines of code)
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
### END CODE HERE ###

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
```

output：

```python
Number of training examples: m_train = 209
Number of testing examples: m_test = 50
Height/Width of each image: num_px = 64
Each image is of size: (64, 64, 3)
train_set_x shape: (209, 64, 64, 3)
train_set_y shape: (1, 209)
test_set_x shape: (50, 64, 64, 3)
test_set_y shape: (1, 50)
```

为了方便起见，你现在应该以维度(num_px ∗ num_px ∗ 3, 1)的numpy数组重塑维度（num_px，num_px，3）的图像。 此后，我们的训练（和测试）数据集是一个numpy数组，其中每列代表一个展平的图像。 应该有m_train（和m_test）列。

## 3-预处理数据集

**练习：** 重塑训练和测试数据集，以便将大小（num_px，num_px，3）的图像展平为单个形状的向量(num_px ∗ num_px ∗ 3, 1)。

将维度为（a，b，c，d）的矩阵X展平为形状为(b∗c∗d, a)的矩阵X_flatten时的一个技巧是：

​	在指定目标 shape 时存在一些技巧：

> - 1. -1 表示这个维度的值是从 x 的元素总数和剩余维度推断出来的。因此，有且只有一个维度可以被设置为-1,利用变换前后的元素数值相等推测出来。
> - 1. 0 表示实际的维数是从 x 的对应维数中复制出来的，因此 shape 中 0 的索引值不能超过 x 的维度，直接复制即可。

这里有一些例子来解释它们：

> - 1。给定一个形状为[2,4,6]的三维 Tensor x，目标形状为[6,8]，则将 x 变换为形状为[6,8]的 2-D Tensor，且 x 的数据保持不变。
> - 2。给定一个形状为[2,4,6]的三维 Tensor x，目标形状为[2,3,-1,2]，则将 x 变换为形状为[2,3,4,2]的 4-D Tensor，且 x 的数据保持不变。在这种情况下，目标形状的一个维度被设置为-1，这个维度的值是从 x 的元素总数和剩余维度推断出来的。
> - 3。给定一个形状为[2,4,6]的三维 Tensor x，目标形状为[-1,0,3,2]，则将 x 变换为形状为[2,4,3,2]的 4-D Tensor，且 x 的数据保持不变。在这种情况下，0 对应位置的维度值将从 x 的对应维数中复制，-1 对应位置的维度值由 x 的元素总数和剩余维度推断出来。

```python
X_flatten = X.reshape(X.shape [0]，-1).T     # 其中X.T是X的转置矩阵
```

多数机器学习以行为特征列为样本数。

```python
# Reshape the training and test examples

### START CODE HERE ### (≈ 2 lines of code)
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
### END CODE HERE ###

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))
```

output：

```python
train_set_x_flatten shape: (12288, 209)
train_set_y shape: (1, 209)
test_set_x_flatten shape: (12288, 50)
test_set_y shape: (1, 50)
sanity check after reshaping: [17 31 56 22 33] #“理智检查”（sanity check），用于验证数据是否按预期进行了重塑。 通过打印数组一小部分，通过视觉检查是否符合预期，选取前五个样本的第一个特征值。
```

机器学习中一个常见的预处理步骤是对数据集进行居中和标准化，这意味着你要从每个示例中减去整个numpy数组的均值，然后除以整个numpy数组的标准差。但是图片数据集则更为简单方便，并且只要将数据集的每一行除以255（像素通道的最大值），效果也差不多。

在训练模型期间，将要乘以权重并向一些初始输入添加偏差以观察神经元的激活。然后，使用反向梯度传播以训练模型。但是，让特征具有相似的范围以至渐变不会爆炸是非常重要的（后续章节）。

```python
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.
```

预处理数据集的常见步骤是：

- 找出数据的尺寸和维度（m_train，m_test，num_px等）
- 重塑数据集，以使每个示例都是大小为（num_px \*num_px \* 3，1）的向量
- “标准化”数据

## 4-算法的一般框架

使用神经网络思维方式建立Logistic回归。 下图说明了为什么“逻辑回归实际上是一个非常简单的神经网络！”

![image-20240523183605947](images/image-20240523183605947.png)

**算法的数学表达式**：

​	前方都是在预测，似乎只有损失函数和成本函数是与训练有关系，成本函数最终是要求偏导成w和b的函数，因此就是通过成本函数和损失函数来找寻使得J最小的w和b函数。

For one example![image-20240523183922055](images/image-20240523183922055.png)：

![image-20240523183932138](images/image-20240523183932138.png)

The cost is then computed by summing over all training examples:

![image-20240523184026470](images/image-20240523184026470.png)

**关键步骤**：

-    初始化模型参数
-    通过最小化损失来学习模型的参数
-    使用学习到的参数进行预测（在测试集上）
-    分析结果并得出结论

## 5-构建算法

建立神经网络的主要步骤是：
1.定义模型结构（例如输入特征的数量）
2.初始化模型的参数
3.循环：

-    计算当前损失（正向传播）
-    计算当前梯度（向后传播）
-    更新参数（梯度下降）

构建1-3，集成到一个称为“ model（）”的函数中。

### 5.1- sigmoid函数(辅助函数)

```python
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s
```

### 5.2- 初始化参数

**练习：** 实现参数初始化。 你必须将w初始化为零的向量使用np.zeros（）。

```python
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b

dim = 2
w, b = initialize_with_zeros(dim)
print ("w = " + str(w))
print ("b = " + str(b))
```

output：

```PYTHON
w = [[0.]
 [0.]]
b = 0
```

### 5.3 前向和后向传播函数

**练习：** 实现函数propagate（）来计算损失函数及其梯度。

正向传播：

- 得到X
- 计算![image-20240523190723683](images/image-20240523190723683.png)
- 计算损失函数：![image-20240523190732001](images/image-20240523190732001.png)

使用到以下两个公式：可从笔记中一步步推出来。

![image-20240523190754111](images/image-20240523190754111.png)

```python
def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)         
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    dw = 1 / m * np.dot(X, (A - Y).T)
    db = 1 / m * np.sum(A - Y)
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost) #从数组的形状中移除单维度的条目。如一些数组为[2,2，1]，其实质就为一个二维数组，但这样写会表现为三维数组，因此可以用squeeze函数将其变为[2,2]
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
#w, b, X, Y = np.array([[1],[2],[3]]), 2, np.array([[1,2,3],[3,4,3],[5,6,3]]), np.array([[1,0,1]])
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))
```

output：

```python
dw = [[0.99993216]
 [1.99980262]]
db = 0.49993523062470574
cost = 6.000064773192205
```

### 5.4-优化函数

- 初始化参数。
- 计算损失函数及其梯度。
- 使用梯度下降来更新参数。

**练习：** 通过最小化损失函数 𝐽 来学习 𝑤 和 𝑏。 对于参数𝜃，更新规则为𝜃=𝜃−𝛼 𝑑𝜃，其中𝛼是学习率。

```python
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    
    costs = []
    
    for i in range(num_iterations): #迭代次数
        

        grads, cost = propagate(w, b, X, Y)


        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:#记录每隔一定步骤（每 100 步）的损失值。
            costs.append(cost)#将损失值（cost）添加到名为costs的列表中。

        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

params, grads, costs = optimize(w, b, X, Y, num_iterations= 101, learning_rate = 0.009, print_cost = False)

print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print(costs)
```

output：

```PYTHON
w = [[0.10440664]
 [0.21535171]]
b = 1.5554725342883116
dw = [[0.89458411]
 [1.74622645]]
db = 0.4258211729530607
[6.000064773192205, 1.4313999565615696]
```

**练习：** 上一个函数将输出学习到的w和b。 我们能够使用w和b来预测数据集X的标签。实现`predict（）`函数。 预测分类有两个步骤：
1.计算![image-20240523193405123](images/image-20240523193405123.png)
2.将a的项转换为0（如果激活<= 0.5）或1（如果激活> 0.5)，并将预测结果存储在向量“ Y_prediction”中。 如果愿意，可以在for循环中使用if / else语句。

```python
def predict(w, b, X):
    #此处的X已经是被预处理过后的，即大小为(X.shape[0],X.shape[1])而不再是RGB三维数组。
    m = X.shape[1]  #获取样本数
    Y_prediction = np.zeros((1,m)) #1行m列
    w = w.reshape(X.shape[0], 1)#w为x对应的行，1列（后续要转置）
    A = sigmoid(np.dot(w.T, X) + b)#y_hat

    for i in range(A.shape[1]):
        if A[0, i] <= 0.5:
            Y_prediction[0, i] = 0 #不是猫
        else:
            Y_prediction[0, i] = 1 #猫
    
    assert(Y_prediction.shape == (1, m)) 
    
    return Y_prediction

print ("predictions = " + str(predict(w, b, X)))
```

output：

```python
predictions = [[1. 1.]] #结果表示测试了两张图片，都为猫。
```

## 6- 将所有功能合并到模型中

现在，将所有构件（在上一部分中实现的功能）以正确的顺序放在一起，从而得到整体的model()函数。

**练习：** 实现函数功能，使用以下符号：

-    Y_prediction对测试集的预测
-    Y_prediction_train对训练集的预测
-    w，损失，optimize（）输出的梯度

```python
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):

    w, b = initialize_with_zeros(X_train.shape[0]) #对w和b进行以X行为数的初始化[dim，1]

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w = parameters["w"]
    b = parameters["b"]#训练后的

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100)) #100-绝对误差（此处误差归一化了，因此乘以100）
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, #w从0通过梯度来计算权重矩阵，dw的值通过照片计算不一样。
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

```

​	通过训练集得到每张照片dw，然后计算损失函数，进而得到成本函数，dw的不同导致了损失函数不同，从而使得更新的w权重矩阵不同。

## 7- 学习率的选择

为了使梯度下降起作用，你必须明智地选择学习率。 学习率𝛼决定我们更新参数的速度。 如果学习率太大，我们可能会“超出”最佳值。 同样，如果太小，将需要更多的迭代才能收敛到最佳值。 这也是为什么调整好学习率至关重要。

让我们将模型的学习曲线与选择的几种学习率进行比较。 运行下面的单元格。 这大约需要1分钟。 还可以尝试与我们初始化要包含的“ learning_rates”变量的三个值不同的值，然后看看会发生什么。

```python
learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
```

output：

```python
learning rate is: 0.01
train accuracy: 99.52153110047847 %
test accuracy: 68.0 %

-------------------------------------------------------

learning rate is: 0.001
train accuracy: 88.99521531100478 %
test accuracy: 64.0 %

-------------------------------------------------------

learning rate is: 0.0001
train accuracy: 68.42105263157895 %
test accuracy: 36.0 %

-------------------------------------------------------
```

![image-20240523212116574](images/image-20240523212116574.png)

## 8-使用自己的图像进行测试

祝贺你完成此作业。 你可以使用自己的图像并查看模型的预测输出。 要做到这一点：
   1.单击此笔记本上部栏中的 "File"，然后单击"Open" 以在Coursera Hub上运行。
   2.将图像添加到Jupyter Notebook的目录中，在"images"文件夹中
   3.在以下代码中更改图像的名称
   4.运行代码，检查算法是否正确（1 = cat，0 = non-cat）！

```python
fname = '/home/kesci/input/deeplearningai17761/cat_in_iran.jpg'#文件路径
image = np.array(plt.imread(fname))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
my_predicted_image = predict(d["w"], d["b"], my_image)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
```

# 用1层隐藏层的神经网络分类二维数据
