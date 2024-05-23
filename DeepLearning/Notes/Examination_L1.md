# L1Week2

## 1-numpy

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

