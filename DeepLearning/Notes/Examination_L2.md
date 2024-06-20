# 神经网络初始化

欢迎来到“改善深度神经网络”的第一项作业。

训练神经网络需要指定权重的初始值，而一个好的初始化方法将有助于网络学习。

如果你完成了本系列的上一课程，则可能已经按照我们的说明完成了权重初始化。但是，如何为新的神经网络选择初始化？在本笔记本中，你能学习看到不同的初始化导致的不同结果。

好的初始化可以：

- 加快梯度下降、模型收敛
- 减小梯度下降收敛过程中训练（和泛化）出现误差的几率

首先，运行以下单元格以加载包和用于分类的二维数据集。

对于load_dataset()函数解析

```PYTHON
def load_dataset():
    np.random.seed(1)  # 设置随机种子以确保结果的可重复性
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)# 生成300个训练样本
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
    # train_X[:, 0] 和 train_X[:, 1] 代表数据点的 x 和 y 坐标，c=train_Y 表示颜色，根据train_Y为不同类别的点着色，s=40点的大小，cmap=plt.cm.Spectral 指定了一个颜色映射，用于给不同类别的点着不同的颜色。
    #train_X[:, 0] 表示获取 train_X 数组中所有行的第0列的元素。这里的 : 表示选择所有行，而 0 表示选择第0列。train_X 是一个二维数组，其中每一行代表一个数据点，第0列和第1列分别代表数据点的x坐标和y坐标。
 #    [[x1, y1],
 #    [x2, y2],
 #    [x3, y3],
 #        ...]
    #那么 train_X[:, 0] 将会是 [x1, x2, x3, ...]，而 train_X[:, 1] 将会是 [y1, y2, y3, ...]。
train_X[:, 1] 表示获取 train_X 数组中所有行的第1列的元素。同样地，: 表示选择所有行，1 表示选择第1列。
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    return train_X, train_Y, test_X, test_Y
```

in：

```python
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from lib.init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
from lib.init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec


plt.rcParams['figure.figsize'] = (7.0, 4.0) # 图表的默认大小设置为宽7英寸、高4英寸。
plt.rcParams['image.interpolation'] = 'nearest'#将图像缩放时使用的插值方法设置为’nearest’，意味着在缩放图像的区域内，将使用原始像素值，这会导致图像看起来像块状，没有平滑处理。
plt.rcParams['image.cmap'] = 'gray' #将图像的默认色彩映射设置为’gray’，即以灰度模式显示图像。

# load image dataset: blue/red dots in circles
train_X, train_Y, test_X, test_Y = load_dataset() #上方解析
plt.show()
```

output:

![image-20240610135848779](images/image-20240610135848779.png)

使用分类器希望将蓝点和红点分开。

## 1-神经网络模型

使用已经实现了的3层神经网络。将尝试的初始化方法：

- *零初始化* ：在输入参数中设置`initialization = "zeros"`。
- *随机初始化* ：在输入参数中设置`initialization = "random"`，这会将权重初始化为较大的随机值。
- *He初始化* ：在输入参数中设置`initialization = "he"`，He初始化。

**说明**：请快速阅读并运行以下代码，在下一部分中，你将实现此`model()`调用的三种初始化方法。

```PYTHON
def model(X, Y, learning_rate = 0.01, num_iterations = 15000, print_cost = True, initialization = "he"):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (containing 0 for red dots; 1 for blue dots), of shape (1, number of examples)
    learning_rate -- learning rate for gradient descent 
    num_iterations -- number of iterations to run gradient descent
    print_cost -- if True, print the cost every 1000 iterations
    initialization -- flag to choose which initialization to use ("zeros","random" or "he")
    
    Returns:
    parameters -- parameters learnt by the model
    """
        
    grads = {} # to keep track of the gradients 梯度
    costs = [] # to keep track of the loss 损失函数
    m = X.shape[1] # number of examples  样本数
    layers_dims = [X.shape[0], 10, 5, 1] #三层神经网络
    
    # Initialize parameters dictionary.
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)

    # Loop (gradient descent)
    #梯度下降算法
    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        a3, cache = forward_propagation(X, parameters) #前向传播
        
        # Loss
        cost = compute_loss(a3, Y)#损失函数

        # Backward propagation.
        grads = backward_propagation(X, Y, cache) #反向传播
        
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)#梯度下降更新参数
        
        # Print the loss every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)
            
    # plot the loss
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters
```

### 1.1-零初始化

在神经网络中有两种类型的参数要初始化：

- 权重矩阵$(W^{[1]}, W^{[2]}, W^{[3]}, ..., W^{[L-1]}, W^{[L]})$
- 偏差向量 $(b^{[1]}, b^{[2]}, b^{[3]}, ..., b^{[L-1]}, b^{[L]})$

**练习**：实现以下函数以将所有参数初始化为零。 稍后你会看到此方法会报错，因为它无法“打破对称性”。总之先尝试一下，看看会发生什么。确保使用正确维度的np.zeros（（..，..））。

​	在神经网络中，如果将所有参数初始化为零，会导致一个问题称为“对称性破坏”。意味着在网络的每一层，所有神经元的权重更新将是相同的。这是因为，如果权重相同，那么在前向传播过程中，每个神经元接收到的输入和梯度将是相同的，因此在反向传播时它们的权重更新也会相同。这样，无论网络有多少层或多少神经元，每一层的所有神经元都会执行相同的操作，相当于网络没有多个神经元的学习能力。

```PYTHON
def initialize_parameters_zeros(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    parameters = {}
    L = len(layers_dims)            # number of layers in the network
    
    for l in range(1, L): #到L-1
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.zeros((layers_dims[l],layers_dims[l-1])) #维度，[当前层，前一层]，通常依靠预处理后的X来确定
        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))#参数b的维度，[当前层，1]
        ### END CODE HERE ###
    return parameters

parameters = initialize_parameters_zeros([3,2,1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
```

output：

```PYTHON
W1 = [[0. 0. 0.]
 [0. 0. 0.]]
b1 = [[0.]
 [0.]]
W2 = [[0. 0.]]
b2 = [[0.]]
```

运行以下代码使用零初始化并迭代15,000次以训练模型。

```python
parameters = model(train_X, train_Y, initialization = "zeros")
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)
```

output:

```python
Cost after iteration 0: 0.6931471805599453
Cost after iteration 1000: 0.6931471805599453
Cost after iteration 2000: 0.6931471805599453
Cost after iteration 3000: 0.6931471805599453
Cost after iteration 4000: 0.6931471805599453
Cost after iteration 5000: 0.6931471805599453
Cost after iteration 6000: 0.6931471805599453
Cost after iteration 7000: 0.6931471805599453
Cost after iteration 8000: 0.6931471805599453
Cost after iteration 9000: 0.6931471805599453
Cost after iteration 10000: 0.6931471805599455
Cost after iteration 11000: 0.6931471805599453
Cost after iteration 12000: 0.6931471805599453
Cost after iteration 13000: 0.6931471805599453
Cost after iteration 14000: 0.6931471805599453
On the train set:
Accuracy: 0.5
On the test set:
Accuracy: 0.5
```

![image-20240610160011776](images/image-20240610160011776.png)

性能确实很差，损失也没有降低，为什么呢？让我们看一下预测的详细信息和决策边界：

```python
print ("predictions_train = " + str(predictions_train))
print ("predictions_test = " + str(predictions_test))
```

output:

```python
predictions_train = [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0]]
predictions_test = [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
```

```python
plt.title("Model with Zeros initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)#绘制决策边界。
```

output:

![image-20240610160400535](images/image-20240610160400535.png)

该模型预测的每个示例都为0。

背景被涂成黄色，代表决策边界区域,由于没有可见的线条或曲线来分隔这两类数据点，这表明模型没有学会区分它们。

通常，将所有权重初始化为零会导致网络无法打破对称性。 这意味着每一层中的每个神经元都将学习相同的东西，并且你不妨训练每一层$n^{[l]}=1$的神经网络，且该网络的性能不如线性分类器，例如逻辑回归。

**因此结论如下**：

- 权重$W^{[l]}$应该随机初始化以打破对称性。
- 将偏差$b^{[l]}$初始化为零是可以的。只要随机初始化了$W^{[l]}$，对称性仍然会破坏。

### 1.2-随机初始化

​	为了打破对称性，让我们随机设置权重。 在随机初始化之后，每个神经元可以继续学习其输入的不同特征。 在本练习中，你将看到如果将权重随机初始化为非常大的值会发生什么。

​	**练习**：实现以下函数，将权重初始化为较大的随机值（按*10缩放），并将偏差设为0。 将 `np.random.randn(..,..) * 10`用于权重，将`np.zeros((.., ..))`用于偏差。

```python
# GRADED FUNCTION: initialize_parameters_random

def initialize_parameters_random(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    np.random.seed(3)               # This seed makes sure your "random" numbers will be the as ours
    parameters = {}
    L = len(layers_dims)            # integer representing the number of layers
    
    for l in range(1, L):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1])*10
        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
        ### END CODE HERE ###

    return parameters

parameters = initialize_parameters_random([3, 2, 1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
```

output:

```python
W1 = [[ 17.88628473   4.36509851   0.96497468]
 [-18.63492703  -2.77388203  -3.54758979]]
b1 = [[0.]
 [0.]]
W2 = [[-0.82741481 -6.27000677]]
b2 = [[0.]]
```

运行以下代码使用随机初始化迭代15,000次以训练模型。

```python
parameters = model(train_X, train_Y, initialization = "random")
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)
```

output:

```PYTHON
Cost after iteration 0: inf
Cost after iteration 1000: 0.6247924745506072
Cost after iteration 2000: 0.5980258056061102
Cost after iteration 3000: 0.5637539062842213
Cost after iteration 4000: 0.5501256393526495
Cost after iteration 5000: 0.5443826306793814
Cost after iteration 6000: 0.5373895855049121
Cost after iteration 7000: 0.47157999220550006
Cost after iteration 8000: 0.39770475516243037
Cost after iteration 9000: 0.3934560146692851
Cost after iteration 10000: 0.3920227137490125
Cost after iteration 11000: 0.38913700035966736
Cost after iteration 12000: 0.3861358766546214
Cost after iteration 13000: 0.38497629552893475
Cost after iteration 14000: 0.38276694641706693
On the train set:
Accuracy: 0.83
On the test set:
Accuracy: 0.86
```

![image-20240614115011370](images/image-20240614115011370.png)

因为数值舍入，你可能在0迭代之后看到损失为"inf"(infinite 无限)，我们会在之后用更复杂的数字实现解决此问题。

```python
print (predictions_train)
print (predictions_test)
```
output：
```python
[[1 0 1 1 0 0 1 1 1 1 1 0 1 0 0 1 0 1 1 0 0 0 1 0 1 1 1 1 1 1 0 1 1 0 0 1
  1 1 1 1 1 1 1 0 1 1 1 1 0 1 0 1 1 1 1 0 0 1 1 1 1 0 1 1 0 1 0 1 1 1 1 0
  0 0 0 0 1 0 1 0 1 1 1 0 0 1 1 1 1 1 1 0 0 1 1 1 0 1 1 0 1 0 1 1 0 1 1 0
  1 0 1 1 0 0 1 0 0 1 1 0 1 1 1 0 1 0 0 1 0 1 1 1 1 1 1 1 0 1 1 0 0 1 1 0
  0 0 1 0 1 0 1 0 1 1 1 0 0 1 1 1 1 0 1 1 0 1 0 1 1 0 1 0 1 1 1 1 0 1 1 1
  1 0 1 0 1 0 1 1 1 1 0 1 1 0 1 1 0 1 1 0 1 0 1 1 1 0 1 1 1 0 1 0 1 0 0 1
  0 1 1 0 1 1 0 1 1 0 1 1 1 0 1 1 1 1 0 1 0 0 1 1 0 1 1 1 0 0 0 1 1 0 1 1
  1 1 0 1 1 0 1 1 1 0 0 1 0 0 0 1 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 1 1 1
  1 1 1 1 0 0 0 1 1 1 1 0]]
[[1 1 1 1 0 1 0 1 1 0 1 1 1 0 0 0 0 1 0 1 0 0 1 0 1 0 1 1 1 1 1 0 0 0 0 1
  0 1 1 0 0 1 1 1 1 1 0 1 1 1 0 1 0 1 1 0 1 0 1 0 1 1 1 1 1 1 1 1 1 0 1 0
  1 1 1 1 1 0 1 0 0 1 0 0 0 1 1 0 1 1 0 0 0 1 1 0 1 1 0 0]]
```

```python
plt.title("Model with large random initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
```

output：

![image-20240614115411087](images/image-20240614115411087.png)

- 损失一开始很高是因为较大的随机权重值，对于某些数据，最后一层激活函数sigmoid输出的结果非常接近0或1，并且当该示例数据预测错误时，将导致非常高的损失。当$\log(a^{[3]}) = \log(0)$时，损失达到无穷大。
- 初始化不当会导致梯度消失/爆炸，同时也会减慢优化算法的速度。
- 训练较长时间的网络，将会看到更好的结果，但是使用太大的随机数进行初始化会降低优化速度。

**总结**：

- 将权重初始化为非常大的随机值效果不佳。
- 初始化为较小的随机值会更好。重要的问题是：这些随机值应为多小？让我们在下一部分中找到答案！

### 1.3-He初始化

​	尝试“He 初始化”，该名称以He等人的名字命名（类似于“Xavier初始化”，但Xavier初始化使用比例因子 `sqrt(1./layers_dims[l-1])`来表示权重𝑊[𝑙] ，而He初始化使用`sqrt(2./layers_dims[l-1])`）。

​	**练习**：实现以下函数，以He初始化来初始化参数。

​	**提示**：此函数类似于先前的`initialize_parameters_random(...)`。 唯一的不同是，无需将`np.random.randn(..,..)`乘以10，而是将其乘以2dimension of the previous layer，这是He初始化建议使用的ReLU激活层。

```python
def initialize_parameters_he(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer. 包含每层大小的 python 数组（列表）。
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims) - 1 # integer representing the number of layers  此处是包含每层大小的python数组，而并非直接定义的层数。
     
    for l in range(1, L + 1): #从1到L
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1])*np.sqrt(2./layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
        ### END CODE HERE ###
        
    return parameters
parameters = initialize_parameters_he([2, 4, 1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
```

output：

```python
W1 = [[ 1.78862847  0.43650985]
 [ 0.09649747 -1.8634927 ]
 [-0.2773882  -0.35475898]
 [-0.08274148 -0.62700068]]
b1 = [[0.]
 [0.]
 [0.]
 [0.]]
W2 = [[-0.03098412 -0.33744411 -0.92904268  0.62552248]]
b2 = [[0.]]
```

运行以下代码，使用He初始化并迭代15,000次以训练你的模型。

```py
parameters = model(train_X, train_Y, initialization = "he")
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)
```

output：

```python
Cost after iteration 0: 0.8830537463419761
Cost after iteration 1000: 0.6879825919728063
Cost after iteration 2000: 0.6751286264523371
Cost after iteration 3000: 0.6526117768893805
Cost after iteration 4000: 0.6082958970572938
Cost after iteration 5000: 0.5304944491717495
Cost after iteration 6000: 0.4138645817071794
Cost after iteration 7000: 0.3117803464844441
Cost after iteration 8000: 0.23696215330322562
Cost after iteration 9000: 0.1859728720920684
Cost after iteration 10000: 0.15015556280371808
Cost after iteration 11000: 0.12325079292273551
Cost after iteration 12000: 0.09917746546525937
Cost after iteration 13000: 0.08457055954024283
Cost after iteration 14000: 0.07357895962677366
On the train set:
Accuracy: 0.9933333333333333
On the test set:
Accuracy: 0.96
```

![image-20240617231530976](images/image-20240617231530976.png)

```python
plt.title("Model with He initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
# plot_decision_boundary 的函数，它用于绘制决策边界。这个函数接受三个参数：
	#一个 lambda 函数：lambda x: predict_dec(parameters, x.T)，它是一个匿名函数，用于对输入的 x 应用 predict_dec 函数。这里的 parameters 是预先定义的模型参数，而 x.T 是将输入的 x 转置。
    #train_X：这通常是一个包含训练数据特征的数组。
	#train_Y：这通常是一个包含训练数据标签的数组。
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
```

![image-20240617234257865](images/image-20240617234257865.png)

# 神经网络正则化

​	深度学习模型具有很高的灵活性和能力，如果训练数据集不够大，**将会造成一个严重的问题--过拟合**。尽管它在训练集上效果很好，但是学到的网络**不能应用到测试集中！**

​	首先导入要使用的包。

```python
# import packages
import numpy as np
import matplotlib.pyplot as plt
from lib.reg_utils import sigmoid, relu, plot_decision_boundary, initialize_parameters, load_2D_dataset, predict_dec
from lib.reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters
import sklearn
import sklearn.datasets
import scipy.io
from lib.testCases import *

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
```

**问题陈述**：你刚刚被法国足球公司聘为AI专家。他们希望你推荐预测法国守门员将球踢出的位置，以便法国队的球员可以用头将球击中。

![image-20240619213902974](images/image-20240619213902974.png)
	守门员将球踢到空中，每支球队的球员都在尽力用头击球

​	以下提供了法国过去10场比赛的二维数据集。

```python
train_X, train_Y, test_X, test_Y = load_2D_dataset()
```

output:

![image-20240619214954850](images/image-20240619214954850.png)

数据中每个点对应于足球场上的位置，在该位置上，法国守门员从足球场左侧射出球后，足球运动员用他/她的头部击中了球。

- 如果圆点为蓝色，则表示法国球员设法用头部将球击中
- 如果圆点为红色，则表示另一支球队的球员用头撞球

**你的目标**：运用深度学习模型预测守门员应将球踢到球场上的位置。

**数据集分析**：该数据集含有噪声，但一条将左上半部分（蓝色）与右下半部分（红色）分开的对角线会很比较有效。

首先尝试非正则化模型。然后学习如何对其进行正则化，并决定选择哪种模型来解决法国足球公司的问题。

## 1-非正则化模型

你将使用以下神经网络（已为你实现），可以如下使用此模型：

- 在*regularization mode*中，通过`lambd`将输入设置为非零值。我们使用`lambd`代替`lambda`，因为`lambda`是Python中的保留关键字。
- 在*dropout mode*中，将`keep_prob`设置为小于1的值

首先，你将尝试不进行任何正则化的模型。然后，你将实现：

- *L2 正则化* 函数：`compute_cost_with_regularization()`和`backward_propagation_with_regularization()`
- *Dropout* 函数：`forward_propagation_with_dropout()`和`backward_propagation_with_dropout()`

```PYTHON
def model(X, Y, learning_rate = 0.3, num_iterations = 30000, print_cost = True, lambd = 0, keep_prob = 1):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (output size, number of examples)
    learning_rate -- learning rate of the optimization
    num_iterations -- number of iterations of the optimization loop
    print_cost -- If True, print the cost every 10000 iterations
    lambd -- regularization hyperparameter, scalar
    keep_prob - probability of keeping a neuron active during drop-out, scalar.
    
    Returns:
    parameters -- parameters learned by the model. They can then be used to predict.
    """
        
    grads = {}
    costs = []                            # to keep track of the cost
    m = X.shape[1]                        # number of examples 样本数
    layers_dims = [X.shape[0], 20, 3, 1] #三层神经网络，第一层为预处理后的样本
    
    # Initialize parameters dictionary.
    parameters = initialize_parameters(layers_dims) #初始化参数w和b（多个）

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        if keep_prob == 1:
            a3, cache = forward_propagation(X, parameters)
        elif keep_prob < 1:
            a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob) #a3是最后一层的输出
        
        # Cost function
        if lambd == 0: #补偿学习率
            cost = compute_cost(a3, Y)
        else:
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)  #是对用w对成本函数进行修正，因此传入parameters获取w参数
            
        # Backward propagation.
        assert(lambd==0 or keep_prob==1)    # it is possible to use both L2 regularization and dropout, 
                                            # but this assignment will only explore one at a time
        if lambd == 0 and keep_prob == 1:
            grads = backward_propagation(X, Y, cache) #正常反向传播
        elif lambd != 0:
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)
        
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the loss every 10000 iterations
        if print_cost and i % 10000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)
    
    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters
parameters = model(train_X, train_Y)
print ("On the training set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)
plt.title("Model without regularization")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
plt.show()
```



在不进行任何正则化的情况下训练模型，并观察训练/测试集的准确性。

output：

```PYTHON
Cost after iteration 0: 0.6557412523481002
Cost after iteration 10000: 0.1632998752572417
Cost after iteration 20000: 0.13851642423284755
```

![image-20240619220547567](images/image-20240619220547567.png)



![image-20240619220641508](images/image-20240619220641508.png)

非正则化模型显然过度拟合了训练集，拟合了一些噪声点！现在让我们看一下减少过拟合的两种手段。

## 2-L2正则化	

避免过拟合的标准方法称为 **L2正则化**，它将损失函数从：
$$
J = -\frac{1}{m} \sum\limits_{i = 1}^{m} \large{(}\small  y^{(i)}\log\left(a^{[L](i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right) \large{)} \tag{1}
$$
修改到：
$$
J_{regularized} = \small \underbrace{-\frac{1}{m} \sum\limits_{i = 1}^{m} \large{(}\small y^{(i)}\log\left(a^{[L](i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right) \large{)} }_\text{cross-entropy cost} + \underbrace{\frac{1}{m} \frac{\lambda}{2} \sum\limits_l\sum\limits_k\sum\limits_j W_{k,j}^{[l]2} }_\text{L2 regularization cost} \tag{2}
$$
在神经网络中，这个公式中的 **k** 和 **j** 代表以下内容：

- **k** 表示神经网络中的第 **k** 个神经元或节点。
- **j** 表示与第 **k** 个神经元相连的权重。

假设有一个神经网络，隐藏层有3个神经元，每个神经元与输入层的4个特征相连。我们可以表示这个隐藏层的权重矩阵为：
$$
W^{[1]} = \begin{bmatrix}
0.5  ; -0.2 ; 0.8 ; 0.3 \\
-0.1 ; 0.6  ; -0.4; 0.7 \\
0.2  ; 0.4  ; 0.1 ; -0.5 \\
\end{bmatrix}
$$
现在，我们可以计算每个权重的平方并求和：
$$
\sum_{k}\sum_{j} W_{k,j}^{[1]2} = (0.5)^2 + (-0.2)^2 + 0.8^2 + 0.3^2 + (-0.1)^2 + 0.6^2 + (-0.4)^2 + 0.7^2 + 0.2^2 + 0.4^2 + 0.1^2 + (-0.5)^2
$$
计算结果为：
$$
\sum_{k}\sum_{j} W_{k,j}^{[1]2} = 2.63
$$
​	**练习**：实现`compute_cost_with_regularization（）`，以计算公式（2）的损失。要计算$\sum\limits_k\sum\limits_j W_{k,j}^{[l]2}$ ，请使用：

```python
np.sum(np.square(Wl))
```

必须对$W^{[1]}$，$W^{[2]}$和$W^{[3]}$执行此操作，然后将三个项相加并乘以$\frac{1}{m}\frac{\lambda}{2}$。

```PYTHON
def compute_cost_with_regularization(A3, Y, parameters, lambd):
    """
    Implement the cost function with L2 regularization. See formula (2) above.
    
    Arguments:
    A3 -- post-activation, output of forward propagation, of shape (output size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    parameters -- python dictionary containing parameters of the model
    
    Returns:
    cost - value of the regularized loss function (formula (2))
    """
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    
    cross_entropy_cost = compute_cost(A3, Y) # This gives you the cross-entropy part of the cost 计算损失函数
    
    ### START CODE HERE ### (approx. 1 line)
    L2_regularization_cost = (1./m*lambd/2)*(np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))) #计算L2正则化项
    ### END CODER HERE ###
    
    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost

A3, Y_assess, parameters = compute_cost_with_regularization_test_case()

print("cost = " + str(compute_cost_with_regularization(A3, Y_assess, parameters, lambd = 0.1)))
```

output:

```PYTHON
cost = 1.7864859451590758
```

当然，因为你更改了损失，所以还必须更改反向传播！ 必须针对新损失函数计算所有梯度。

**练习**：实现正则化后的反向传播。更改仅涉及dW1，dW2和dW3。对于每一个，你必须添加正则化项的梯度$\frac{d}{dW} ( \frac{1}{2}\frac{\lambda}{m}  W^2) = \frac{\lambda}{m} W$。

非常容易、根据导数四则运算、求导后依旧相加，即在dw后方添加$\frac{\lambda}{m} W$即可。

```PYTHON

def backward_propagation_with_regularization(X, Y, cache, lambd):
    #导数都是从传导图中直接推算得到
    """
    Implements the backward propagation of our baseline model to which we added an L2 regularization.

    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    cache -- cache output from forward_propagation()
    lambd -- regularization hyperparameter, scalar

    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """

    m = X.shape[1] #样本数
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y

    ### START CODE HERE ### (approx. 1 line)
    dW3 = 1./m * np.dot(dZ3, A2.T) + lambd/m * W3
    ### END CODE HERE ###
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    ### START CODE HERE ### (approx. 1 line)
    dW2 = 1./m * np.dot(dZ2, A1.T) + lambd/m * W2
    ### END CODE HERE ###
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    ### START CODE HERE ### (approx. 1 line)
    dW1 = 1./m * np.dot(dZ1, X.T) + lambd/m * W1
    ### END CODE HERE ###
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients

X_assess, Y_assess, cache = backward_propagation_with_regularization_test_case()

grads = backward_propagation_with_regularization(X_assess, Y_assess, cache, lambd = 0.7)
print ("dW1 = "+ str(grads["dW1"]))
print ("dW2 = "+ str(grads["dW2"]))
print ("dW3 = "+ str(grads["dW3"]))
```

output：

```PYTHON
dW1 = [[-0.25604646  0.12298827 -0.28297129]
 [-0.17706303  0.34536094 -0.4410571 ]]
dW2 = [[ 0.79276486  0.85133918]
 [-0.0957219  -0.01720463]
 [-0.13100772 -0.03750433]]
dW3 = [[-1.77691347 -0.11832879 -0.09397446]]
```

现在让我们使用L2正则化(𝜆=0.7)运行的模型。`model（）`函数将调用：

- `compute_cost_with_regularization`代替`compute_cost`
- `backward_propagation_with_regularization`代替`backward_propagation`

```PYTHON
parameters = model(train_X, train_Y, lambd = 0.7)
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)
```

output：

```python
Cost after iteration 0: 0.6974484493131264
Cost after iteration 10000: 0.2684918873282239
Cost after iteration 20000: 0.2680916337127301
On the train set:
Accuracy: 0.9383886255924171
On the test set:
Accuracy: 0.93
```

![image-20240620222750656](images/image-20240620222750656.png)

决策边界：

```python
plt.title("Model with L2-regularization")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
```

output：

![image-20240620222850603](images/image-20240620222850603.png)

- 𝜆  的值是你可以调整开发集的超参数。
- L2正则化使决策边界更平滑。如果𝜆 太大，则也可能“过度平滑”，从而使模型偏差较高。

### 2.1-L2正则化的原理

L2正则化基于以下假设：权重较小的模型比权重较大的模型更简单。因此，通过对损失函数中权重的平方值进行惩罚，可以将所有权重驱动为较小的值。比重太大会使损失过高！这将导致模型更平滑，输出随着输入的变化而变化得更慢。

 L2正则化的影响：

- 损失计算：
    \- 正则化条件会添加到损失函数中
- 反向传播函数：
    \- 有关权重矩阵的渐变中还有其他术语
- 权重最终变小（“权重衰减”）：
    \- 权重被推到较小的值。

