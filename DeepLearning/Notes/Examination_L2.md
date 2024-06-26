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

## 3-Dropout

**Dropout**是广泛用于深度学习的正则化技术。
**它会在每次迭代中随机关闭一些神经元。**

```
要了解Dropout，可以思考与朋友进行以下对话：  
- 朋友：“为什么你需要所有神经元来训练你的网络以分类图像？”。  
- 你：“因为每个神经元都有权重，并且可以学习图像的特定特征/细节/形状。我拥有的神经元越多，模型学习的特征就越丰富！”  
- 朋友：“我知道了，但是你确定你的神经元学习的是不同的特征而不是全部相同的特征吗？”  
- 你：“这是个好问题……同一层中的神经元实际上并不关联。应该绝对有可能让他们学习相同的图像特征/形状/形式/细节...这是多余的。为此应该有一个解决方案。”
```

​	在每次迭代中，以概率$1 - keep\_prob$或以概率$keep\_prob$（此处为50％）关闭此层的每个神经元。关闭的神经元对迭代的正向和反向传播均无助于训练。

![Image Name](https://cdn.kesci.com/upload/image/q1au928m1v.gif?imageView2/0/w/960/h/960)

$1^{st}$层：我们平均关闭了40％的神经元。$3^{rd}$层：我们平均关闭了20％的神经元。

​	当你关闭某些神经元时，实际上是在修改模型。Dropout背后的想法是，在每次迭代中，你将训练仅使用神经元子集的不同模型。通过Dropout，你的神经元对另一种特定神经元的激活变得不那么敏感，因为另一神经元可能随时关闭。

![Image Name](https://cdn.kesci.com/upload/image/q1aua8h2gs.gif?imageView2/0/w/960/h/960)

### 3.1-Dropout正向传播

​	**练习**：实现带有Dropout的正向传播。你正在使用3层的神经网络，并将为第一和第二隐藏层添加Dropout。我们不会将Dropout应用于输入层或输出层。

**说明**：
关闭第一层和第二层中的某些神经元。为此，将执行4个步骤：

1. 我们讨论了使用`np.random.rand（）`创建与$a^{[1]}$形状相同的变量$d^{[1]}$的方法，以随机获取0到1之间的数。在这里，你将使用向量化的实现，创建一个与$A^{[1]}$的矩阵维度相同的随机矩阵$D^{[1]} = [d^{[1](1)} d^{[1](2)} ... d^{[1](m)}]$。
2. 通过对𝐷[1]中的值进行阈值设置，将$D^{[1]}$的每个条目设置为0（概率为`1-keep_prob`）或1（概率为`keep_prob`）。提示：将矩阵X的所有条目设置为0（如果概率小于0.5）或1（如果概率大于0.5），则可以执行：`X = (X < 0.5)`。注意0和1分别对应False和True。
3. 将$A^{[1]}$设置为$A^{[1]} * D^{[1]}$（关闭一些神经元）。你可以将$D^{[1]}$ 视为掩码，这样当它与另一个矩阵相乘时，关闭某些值。
4. 将$A^{[1]}$除以`keep_prob`。通过这样做，你可以确保损失结果仍具有与dropout相同的期望值，相当于原来的有`keep_prob`的概率被留下来，即期望乘keep_prob，此处再除了就保持原期望。（此技术也称为反向dropout）

```PYTHON

def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
    """
    Implements the forward propagation: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.
    
    Arguments:
    X -- input dataset, of shape (output size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
    keep_prob - probability of keeping a neuron active during drop-out, scalar
    
    Returns:
    A3 -- last activation value, output of the forward propagation, of shape (1,1)
    cache -- tuple, information stored for computing the backward propagation
    """
    
    np.random.seed(1)
    
    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    ### START CODE HERE ### (approx. 4 lines)         # Steps 1-4 below correspond to the Steps 1-4 described above. 
    D1 = np.random.rand(A1.shape[0],A1.shape[1])               # Step 1: initialize matrix D1 = np.random.rand(..., ...)
    D1 = D1 < keep_prob                                      # Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)
    A1 = A1 * D1                                         # Step 3: shut down some neurons of A1
    A1 = A1 / keep_prob                                        # Step 4: scale the value of neurons that haven't been shut down
    ### END CODE HERE ###
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    ### START CODE HERE ### (approx. 4 lines)
    D2 = np.random.rand(A2.shape[0],A2.shape[1])               # Step 1: initialize matrix D2 = np.random.rand(..., ...)
    D2 = D2 < keep_prob                                         # Step 2: convert entries of D2 to 0 or 1 (using keep_prob as the threshold)
    A2 = A2 * D2                                         # Step 3: shut down some neurons of A2
    A2 = A2 / keep_prob                                      # Step 4: scale the value of neurons that haven't been shut down
    ### END CODE HERE ###
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)

    return A3, cache

X_assess, parameters = forward_propagation_with_dropout_test_case()

A3, cache = forward_propagation_with_dropout(X_assess, parameters, keep_prob = 0.7)
print ("A3 = " + str(A3))
```

output：

```PYTHON
X_assess, parameters = forward_propagation_with_dropout_test_case()

A3, cache = forward_propagation_with_dropout(X_assess, parameters, keep_prob = 0.7)
print ("A3 = " + str(A3))
```

### 3.2-Dropout反向传播

**练习**：实现带有dropout的反向传播。和之前一样，训练一个3层的网络。使用存储在缓存中的掩码$D^{[1]}$和$D^{[2]}$，添加dropout到第一和第二个隐藏层。

**说明**：
带有dropout的反向传播实现上非常容易。你将必须执行2个步骤：
1.在上方通过在$A^{[1]}$上应用掩码$D^{[1]}$来关闭正向传播过程中的某些神经元。在反向传播中，必须将相同的掩码$D^{[1]}$重新应用于$dA^{[1]}$来关闭相同的神经元。
2.在正向传播过程中，你已将$A^{[1]}$除以`keep_prob`。 因此，在反向传播中，必须再次将`dA1`除以`keep_prob`（计算的解释是，如果𝐴[1]被`keep_prob`缩放，则其派生的𝑑𝐴[1]也由相同的`keep_prob`缩放）。

```PYTHON
def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    """
    Implements the backward propagation of our baseline model to which we added dropout.
    
    Arguments:

    X -- input dataset, of shape (output size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    cache -- cache output from forward_propagation_with_dropout()
    keep_prob - probability of keeping a neuron active during drop-out, scalar
    
    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    
    m = X.shape[1] #样本数
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    dA2 = np.dot(W3.T, dZ3)
    ### START CODE HERE ### (≈ 2 lines of code)
    dA2 = dA2 * D2              # Step 1: Apply mask D2 to shut down the same neurons as during the forward propagation
    dA2 = dA2 / keep_prob           # Step 2: Scale the value of neurons that haven't been shut down
    ### END CODE HERE ###
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)

    dA1 = np.dot(W2.T, dZ2)
    ### START CODE HERE ### (≈ 2 lines of code)
    dA1 = dA1 * D1              # Step 1: Apply mask D1 to shut down the same neurons as during the forward propagation
    dA1 = dA1 / keep_prob             # Step 2: Scale the value of neurons that haven't been shut down
    ### END CODE HERE ###
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients

X_assess, Y_assess, cache = backward_propagation_with_dropout_test_case()

gradients = backward_propagation_with_dropout(X_assess, Y_assess, cache, keep_prob = 0.8)

print ("dA1 = " + str(gradients["dA1"]))
print ("dA2 = " + str(gradients["dA2"]))
```

output：

```PYTHON
dA1 = [[ 0.36544439  0.         -0.00188233  0.         -0.17408748]
 [ 0.65515713  0.         -0.00337459  0.         -0.        ]]
dA2 = [[ 0.58180856  0.         -0.00299679  0.         -0.27715731]
 [ 0.          0.53159854 -0.          0.53159854 -0.34089673]
 [ 0.          0.         -0.00292733  0.         -0.        ]]
```

现在让我们使用dropout（`keep_prob = 0.86`）运行模型。 这意味着在每次迭代中，你都以24％的概率关闭第1层和第2层的每个神经元。 函数`model()`将调用：

- `forward_propagation_with_dropout`而不是`forward_propagation`。
- `backward_propagation_with_dropout`，而不是`backward_propagation`。

```PYTHON
parameters = model(train_X, train_Y, keep_prob = 0.86, learning_rate = 0.3)

print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)
```

output：

```PYTHON
Cost after iteration 0: 0.6543912405149825
Cost after iteration 10000: 0.0610169865749056
Cost after iteration 20000: 0.060582435798513114
On the train set:
Accuracy: 0.9289099526066351
On the test set:
Accuracy: 0.95
```

![image-20240620231309071](images/image-20240620231309071.png)

决策边界

```python
plt.title("Model with dropout")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
```

![image-20240620231344425](images/image-20240620231344425.png)

**注意**：

- 使用dropout时的**常见错误**是在训练和测试中都使用。你只能在训练中使用dropout（随机删除节点）。
- 深度学习框架，例如[tensorflow](https://www.tensorflow.org/api_docs/python/tf/nn/dropout), [PaddlePaddle](http://doc.paddlepaddle.org/release_doc/0.9.0/doc/ui/api/trainer_config_helpers/attrs.html), [keras](https://keras.io/layers/core/#dropout)或者 [caffe](http://caffe.berkeleyvision.org/tutorial/layers/dropout.html) 附带dropout层的实现。

**关dropout应该记住的事情：**

- dropout是一种正则化技术。
- 仅在训练期间使用dropout，在测试期间不要使用。
- 在正向和反向传播期间均应用dropout。
- 在训练期间，将每个dropout层除以keep_prob，以保持激活的期望值相同。例如，如果keep_prob为0.5，则平均而言，我们将关闭一半的节点，因此输出将按0.5缩放，因为只有剩余的一半对解决方案有所贡献。除以0.5等于乘以2，因此输出现在具有相同的期望值。你可以检查此方法是否有效，即使keep_prob的值不是0.5。

**从此笔记本中记住的内容**：

- 正则化将帮助减少过拟合。
- 正则化将使权重降低到较低的值。
- L2正则化和Dropout是两种非常有效的正则化技术。

# 神经网络梯度检验

​	假设你是致力于在全球范围内提供移动支付的团队的一员，被上级要求建立深度学习模型来检测欺诈行为--每当有人进行支付时，你都应该确认该支付是否可能是欺诈性的，例如用户的帐户已被黑客入侵。

​	但是模型的反向传播很难实现，有时还会有错误。因为这是关键的应用任务，所以你公司的CEO要反复确定反向传播的实现是正确的。CEO要求你证明你的反向传播实际上是有效的！为了保证这一点，你将应用到“梯度检验”。

```python
import numpy as np
from lib.testCases import *
from lib.gc_utils import sigmoid, relu, dictionary_to_vector, vector_to_dictionary, gradients_to_vector
```

## 1-梯度检验原理

反向传播计算梯度$\frac{\partial J}{\partial \theta}$，其中𝜃表示模型的参数。使用正向传播和损失函数来计算$𝐽$。

由于正向传播相对容易实现，相信你有信心能做到这一点，确定100％计算正确的损失$𝐽$。为此，你可以使用$𝐽$来验证代码$\frac{\partial J}{\partial \theta}$。

让我们回顾一下导数（或者说梯度）的定义：
$$
\frac{\partial J}{\partial \theta} = \lim_{\varepsilon \to 0} \frac{J(\theta + \varepsilon) - J(\theta - \varepsilon)}{2 \varepsilon} \tag{1}
$$
我们知道以下内容：

- $\frac{\partial J}{\partial \theta}$是你要确保计算正确的对象。
- 你可以计算$J(\theta + \varepsilon)$和$J(\theta - \varepsilon)$（在𝜃是实数的情况下），因为要保证𝐽的实现是正确的。

让我们使用方程式（1）和 𝜀的一个小值来说服CEO你计算∂𝐽∂𝜃的代码是正确的！

## 2-一维梯度检查

思考一维线性函数$J(\theta) = \theta x$，该模型仅包含一个实数值参数$\theta$，并以$𝑥$作为输入。

你将实现代码以计算$ 𝐽(.)$及其派生$\frac{\partial J}{\partial \theta}$，然后，你将使用梯度检验来确保$𝐽$的导数计算正确。

下图显示了关键的计算步骤：首先从$𝑥$开始，再评估函数$𝐽(𝑥)$（正向传播），然后计算导数$\frac{\partial J}{\partial \theta}$（反向传播）。

![image-20240620233749298](images/image-20240620233749298.png)

**练习**：为此简单函数实现“正向传播”和“向后传播”。 即在两个单独的函数中，计算$𝐽(.)$ （正向传播）及其相对于$\theta$(反向传播）的导数。

```PYTHON
# GRADED FUNCTION: forward_propagation

def forward_propagation(x, theta):
    """
    Implement the linear forward propagation (compute J) presented in Figure 1 (J(theta) = theta * x)
    
    Arguments:
    x -- a real-valued input
    theta -- our parameter, a real number as well
    
    Returns:
    J -- the value of function J, computed using the formula J(theta) = theta * x
    """
    
    ### START CODE HERE ### (approx. 1 line)
    J = theta * x
    ### END CODE HERE ###
    
    return J
```

**练习**：现在，执行图1的反向传播步骤（导数计算）。也就是说，计算𝐽(𝜃)=𝜃𝑥 相对于 𝜃的导数。为避免进行演算，你应该得到$d\theta = \frac { \partial J }{ \partial \theta} = x$。

```PYTHON
# GRADED FUNCTION: backward_propagation

def backward_propagation(x, theta):
    """
    Computes the derivative of J with respect to theta (see Figure 1).
    
    Arguments:
    x -- a real-valued input
    theta -- our parameter, a real number as well
    
    Returns:
    dtheta -- the gradient of the cost with respect to theta
    """
    
    ### START CODE HERE ### (approx. 1 line)
    dtheta = x
    ### END CODE HERE ###

    
    return dtheta
```

**练习**：为了展示`backward_propagation（）`函数正确计算了梯度$\frac { \partial J }{ \partial \theta}$，让我们实施梯度检验。

**说明**：

- 首先使用上式（1）和$\varepsilon$的极小值计算“gradapprox”。以下是要遵循的步骤

$$
\theta^{+} = \theta + \varepsilon
$$

$$
\theta^{+} = \theta - \varepsilon
$$

$$
J^{+} = J(\theta^{+})
$$

$$
J^{-} = J(\theta^{-})
$$

$$
gradapprox = \frac{J^{+} - J^{-}}{2  \varepsilon}
$$

- 然后使用反向传播计算梯度，并将结果存储在变量“grad”中
- 最后，使用以下公式计算“gradapprox”和“grad”之间的相对差：

$$
difference = \frac {\mid\mid grad - gradapprox \mid\mid_2}{\mid\mid grad \mid\mid_2 + \mid\mid gradapprox \mid\mid_2} \tag{2}
$$

- `np.linalg.norm()` 是 NumPy 库中的一个函数，用于计算一个向量的范数。它可以计算不同维度的向量的不同范数，比如向量的 L1 范数、L2 范数等。
  1. **基本用法**：
     - `np.linalg.norm(x, ord=None, axis=None, keepdims=False)`
     - `x`：表示要计算范数的向量或矩阵。
     - `ord`：表示范数的类型。
     - `axis`：表示计算范数的轴。
     - 向量的范数：
       - L1 范数：表示求列和的最大值。
       - L2 范数：表示求特征值，然后求最大特征值的算术平方根。
       - L∞ 范数：表示求行和的最大值。
       - `ord=None`：表示求整体的矩阵元素平方和，再开根号。
- 你需要3个步骤来计算此公式：
  - 1. 使用np.linalg.norm（...）计算分子
  - 2. 计算分母，调用np.linalg.norm（...）两次
  - 3. 相除
- 如果差异很小（例如小于$10^{-7}$），则可以确信正确计算了梯度。否则，梯度计算可能会出错。

```python
# GRADED FUNCTION: gradient_check

def gradient_check(x, theta, epsilon = 1e-7):
    """
    Implement the backward propagation presented in Figure 1.
    
    Arguments:
    x -- a real-valued input
    theta -- our parameter, a real number as well
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)
    
    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """
    
    # Compute gradapprox using left side of formula (1). epsilon is small enough, you don't need to worry about the limit.
    ### START CODE HERE ### (approx. 5 lines)
    thetaplus = theta + epsilon                               # Step 1
    thetaminus = theta - epsilon                              # Step 2
    J_plus = forward_propagation(x, thetaplus)                                  # Step 3
    J_minus = forward_propagation(x, thetaminus)                                 # Step 4
    gradapprox = (J_plus - J_minus) / (2 * epsilon)                              # Step 5
    ### END CODE HERE ###

    # Check if gradapprox is close enough to the output of backward_propagation()
    ### START CODE HERE ### (approx. 1 line)
    grad = backward_propagation(x, theta)
    ### END CODE HERE ###

    ### START CODE HERE ### (approx. 1 line)
    numerator = np.linalg.norm(grad - gradapprox)                               # Step 1'
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)                            # Step 2'
    difference = numerator / denominator                              # Step 3'
    ### END CODE HERE ###
    
    if difference < 1e-7:
        print ("The gradient is correct!")
    else:
        print ("The gradient is wrong!")
    
    return difference
x, theta = 2, 4
difference = gradient_check(x, theta)
print("difference = " + str(difference))
```

output：

```PYTHON
The gradient is correct!
difference = 2.919335883291695e-10
```

## 3-N维梯度检验

在更一般的情况下，损失函数$𝐽$具有多个单个输入。当你训练神经网络时，$𝜃$实际上由多个矩阵$W^{[l]}$组成，并加上偏差$b^{[l]}$！重要的是要知道如何对高维输入进行梯度检验。

下图描述了欺诈检测模型的正向传播和反向传播：

![image-20240620235147746](images/image-20240620235147746.png)

让我们看一下正向传播和反向传播的实现。

```PYTHON
def forward_propagation_n(X, Y, parameters):
    """
    Implements the forward propagation (and computes the cost) presented in Figure 3.
    
    Arguments:
    X -- training set for m examples
    Y -- labels for m examples 
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
    
    Returns:
    cost -- the cost function (logistic cost for one example)
    """
    
    # retrieve parameters
    m = X.shape[1]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    # Cost
    logprobs = np.multiply(-np.log(A3),Y) + np.multiply(-np.log(1 - A3), 1 - Y)
    cost = 1./m * np.sum(logprobs)
    
    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
    
    return cost, cache
```

```PYTHON
def backward_propagation_n(X, Y, cache):
    """
    Implement the backward propagation presented in figure 2.
    
    Arguments:
    X -- input datapoint, of shape (input size, 1)
    Y -- true "label"
    cache -- cache output from forward_propagation_n()
    
    Returns:
    gradients -- A dictionary with the gradients of the cost with respect to each parameter, activation and pre-activation variables.
    """
    
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T) * 2
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 4./m * np.sum(dZ1, axis=1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients
```

### 3.1-N维梯度检验原理

你想将“gradapprox”与通过反向传播计算的梯度进行比较。公式仍然是：
$$
\frac{\partial J}{\partial \theta} = \lim_{\varepsilon \to 0} \frac{J(\theta + \varepsilon) - J(\theta - \varepsilon)}{2 \varepsilon} \tag{1}
$$
但是，$𝜃$不再是标量。 而是一个叫做“参数”的字典。 我们为你实现了一个函数"`dictionary_to_vector()`"。它将“参数”字典转换为称为“值”的向量，该向量是通过将所有参数($W^{[1]}$,$b^{[1]}$, $W^{[2]}$, $b^{[2]}$,$W^{[3]}$, $b^{[3]}$)重塑为向量并将它们串联而获得的。（此处生成的行为数量，列为1）

反函数是“`vector_to_dictionary`”，它输出回“parameters”字典。

![image-20240620235836335](images/image-20240620235836335.png)

将在 gradient_check_n()中用到这些函数

我们还使用gradients_to_vector()将“gradients”字典转换为向量“grad”。

- `np.copy()` 的主要功能是创建一个数组或矩阵的副本。
- 这个副本与原始数组或矩阵在内存中是独立的，对副本的修改不会影响到原始数据。

**练习**：实现gradient_check_n()。

**说明**：这是伪代码，可帮助你实现梯度检验。

For each i in num_parameters:

- 计算

  `J_plus [i] `:

  1. 将$\theta^{+}$设为 `np.copy(parameters_values)`  
  
  2. 将$\theta^{+}_i$设为$\theta^{+}_i + \varepsilon$
  
- 计算`J_minus [i]`：

    1.将$\theta^{-}$设为 `np.copy(parameters_values)`  

    2.将$\theta^{-}_i$设为$\theta^{-}_i - \varepsilon$


- $gradapprox[i] = \frac{J^{+}_i - J^{-}_i}{2 \varepsilon}$

因此，你将获得向量`gradapprox`，其中`gradapprox[i]`是相对于`parameter_values[i]`的梯度的近似值。现在，你可以将此`gradapprox`向量与反向传播中的梯度向量进行比较。就像一维情况（步骤1'，2'，3'）一样计算：
$$
difference = \frac {\| grad - gradapprox \|_2}{\| grad \|_2 + \| gradapprox \|_2 } \tag{3}
$$

```PYTHON

def gradient_check_n(parameters, gradients, X, Y, epsilon = 1e-7):
    """
    Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n
    
    Arguments:
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
    grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters. 
    x -- input datapoint, of shape (input size, 1)
    y -- true "label"
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)
    
    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """
    
    # Set-up variables
    parameters_values, _ = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0] # number of parameters （47，1） 行为数量
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    
    # Compute gradapprox
    for i in range(num_parameters):
        
        # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
        # "_" is used because the function you have to outputs two parameters but we only care about the first one
        ### START CODE HERE ### (approx. 3 lines)
        thetaplus = np.copy(parameters_values)                                      # Step 1
        thetaplus[i][0] = thetaplus[i][0] + epsilon                                # Step 2
        J_plus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaplus))    #成本函数     # Step 3
        ### END CODE HERE ###

        # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
        ### START CODE HERE ### (approx. 3 lines)
        thetaminus = np.copy(parameters_values)                                     # Step 1
        thetaminus[i][0] = thetaminus[i][0] - epsilon                            # Step 2        
        J_minus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaminus))                                  # Step 3
        ### END CODE HERE ###

        # Compute gradapprox[i]
        ### START CODE HERE ### (approx. 1 line)
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2.* epsilon)
        ### END CODE HERE ###

    # Compare gradapprox to backward propagation gradients by computing difference.
    ### START CODE HERE ### (approx. 1 line)
    numerator = np.linalg.norm(grad - gradapprox)                                           # Step 1'
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)                                         # Step 2'
    difference = numerator / denominator                                          # Step 3'
    ### END CODE HERE ###

    if difference > 1e-7:
        print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
    
    return difference

X, Y, parameters = gradient_check_n_test_case()

cost, cache = forward_propagation_n(X, Y, parameters)
gradients = backward_propagation_n(X, Y, cache)
difference = gradient_check_n(parameters, gradients, X, Y)
```

output：

```PYTHON
There is a mistake in the backward propagation! difference = 0.2850931567761624
```

​	看起来`backward_propagation_n`代码似乎有错误！很好，你已经实现了梯度检验。返回到`backward_propagation`并尝试查找/更正错误*（提示：检查dW2和db1）*。如果你已解决问题，请重新运行梯度检验。

你可以进行梯度检验来证明你的导数计算的正确吗？即使作业的这一部分没有评分，我们也强烈建议你尝试查找错误并重新运行梯度检验，直到确信实现了正确的反向传播。

**注意**

- 梯度检验很慢！用$\frac{\partial J}{\partial \theta} \approx  \frac{J(\theta + \varepsilon) - J(\theta - \varepsilon)}{2 \varepsilon}$逼近梯度在计算上是很耗费资源的。因此，我们不会在训练期间的每次迭代中都进行梯度检验。只需检查几次梯度是否正确。
- 至少如我们介绍的那样，梯度检验不适用于dropout。通常，你将运行不带dropout的梯度检验算法以确保你的backprop是正确的，然后添加dropout。
- 梯度检验可验证反向传播的梯度与梯度的数值近似值之间的接近度（使用正向传播进行计算）。
- 梯度检验很慢，因此我们不会在每次训练中都运行它。通常，你仅需确保其代码正确即可运行它，然后将其关闭并将backprop用于实际的学习过程。

# 算法优化

​	到目前为止，你一直使用梯度下降来更新参数并使损失降至最低。 在本笔记本中，你将学习更多高级的优化方法，以加快学习速度，甚至可以使你的损失函数的获得更低的最终值。 一个好的优化算法可以使需要训练几天的网络，训练仅仅几个小时就能获得良好的结果。
梯度下降好比在损失函数$J$上“下坡”。就像下图：**损失最小化好比在丘陵景观中寻找最低点**

![image-20240626223539471](images/image-20240626223539471.png)

​	在训练的每个步骤中，你都按照一定的方向更新参数，以尝试到达最低点。

​	**符号**：与往常一样，$\frac{\partial J}{\partial a } =da$适用于任何变量`a`。

​	首先，请运行以下代码以导入所需的库。

```PYTHON
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from lib.opt_utils import load_params_and_grads, initialize_parameters, forward_propagation, backward_propagation
from lib.opt_utils import compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset
from lib.testCases import *


plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
```

## 1 梯度下降

​	机器学习中一种简单的优化方法是梯度下降（gradient descent,GD）。当你对每个step中的所有m示例执行梯度计算步骤时，它也叫做“批量梯度下降”。

**热身练习**：实现梯度下降更新方法。 对于$l=1,...,L，$梯度下降规则为：
$$
W^{[l]} = W^{[l]} - \alpha \text{ } dW^{[l]} \tag{1}
$$

$$
b^{[l]} = b^{[l]} - \alpha \text{ } db^{[l]} \tag{2}
$$

其中L是层数，$α$是学习率。所有参数都应存储在 `parameters`字典中。请注意，迭代器`l`在`for` 循环中从0开始，而第一个参数是$W^{[1]}$和$b^{[1]}$。编码时需要将`l` 转换为`l+1`。

```PYTHON
def update_parameters_with_gd(parameters, grads, learning_rate):
    """
    Update parameters using one step of gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters to be updated:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients to update each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    learning_rate -- the learning rate, scalar.
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """

    L = len(parameters) // 2 # number of layers in the neural networks

    # Update rule for each parameter
    for l in range(L):
        ### START CODE HERE ### (approx. 2 lines)
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] -learning_rate*grads["db" + str(l+1)]
        ### END CODE HERE ###
        
    return parameters


parameters, grads, learning_rate = update_parameters_with_gd_test_case()
parameters = update_parameters_with_gd(parameters, grads, learning_rate)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
```

output：

```PYTHON
W1 = [[ 1.63535156 -0.62320365 -0.53718766]
 [-1.07799357  0.85639907 -2.29470142]]
b1 = [[ 1.74604067]
 [-0.75184921]]
W2 = [[ 0.32171798 -0.25467393  1.46902454]
 [-2.05617317 -0.31554548 -0.3756023 ]
 [ 1.1404819  -1.09976462 -0.1612551 ]]
b2 = [[-0.88020257]
 [ 0.02561572]
 [ 0.57539477]]
```

​	它的一种变体是随机梯度下降（SGD），它相当于mini版的批次梯度下降，其中每个mini-batch只有一个数据示例。刚刚实现的更新规则不会更改。不同的是，SGD一次仅在一个训练数据上计算梯度，而不是在整个训练集合上计算梯度。下面的代码示例说明了随机梯度下降和（批量）梯度下降之间的区别。

- **(Batch) Gradient Descent**:

```PYTHON
X = data_input  
Y = labels  
parameters = initialize_parameters(layers_dims)  
for i in range(0, num_iterations):  
    # Forward propagation  
    a, caches = forward_propagation(X, parameters)  
    # Compute cost.  
    cost = compute_cost(a, Y)  
    # Backward propagation.  
    grads = backward_propagation(a, caches, parameters)  
    # Update parameters.  
    parameters = update_parameters(parameters, grads)
```

- **Stochastic Gradient Descent**:

```python
X = data_input  
Y = labels  
parameters = initialize_parameters(layers_dims)  
for i in range(0, num_iterations):  
    for j in range(0, m): #m为样本数，相当于mini-batch为1
        # Forward propagation  
        a, caches = forward_propagation(X[:,j], parameters)  
        # Compute cost  
        cost = compute_cost(a, Y[:,j])  
        # Backward propagation  
        grads = backward_propagation(a, caches, parameters)  
        # Update parameters.  
        parameters = update_parameters(parameters, grads)
```

​	对于随机梯度下降，在更新梯度之前，只使用1个训练样例。当训练集大时，SGD可以更新的更快。但是这些参数会向最小值“摆动”而不是平稳地收敛。下图是一个演示例子：“+”表示损失的最小值。 SGD造成许多振荡以达到收敛。但是每个step中，计算SGD比使用GD更快，因为它仅使用一个训练示例（相对于GD的整个批次）。

![image-20240626225704314](images/image-20240626225704314.png)

**注意**：实现SGD总共需要3个for循环：
1.迭代次数
2.$m$个训练数据
3.各层上（要更新所有参数，从($(W^{[1]},b^{[1]}))$到($(W^{[L]},b^{[L]}))$

## 2 Mini-Batch 梯度下降

​	实际上，如果你既不使用整个训练集也不使用一个训练示例来执行每次更新，则通常会得到更快的结果。小批量梯度下降法在每个步骤中使用中间数量的示例。通过小批量梯度下降，你可以遍历小批量，而不是遍历各个训练示例。

![image-20240626230014816](images/image-20240626230014816.png)

**SGD vs Mini-Batch GD**
	“+”表示损失的最小值。在优化算法中使用mini-batch批处理通常可以加快优化速度。

**你应该记住**：

- 梯度下降，小批量梯度下降和随机梯度下降之间的差异是用于执行一个更新步骤的数据数量。
- 必须调整超参数学习率$α$。
- 在小批量的情况下，通常它会胜过梯度下降或随机梯度下降（尤其是训练集较大时）。

如何从训练集（X，Y）中构建小批次数据。

分两个步骤：

- **Shuffle**(打乱次序)：如下所示，创建训练集（X，Y）的随机打乱版本。X和Y中的每一列代表一个训练示例。注意，随机打乱是在X和Y之间同步完成的。这样，在随机打乱之后，X的$i^{th}$列就是对应于Y中$i^{th}标$签的示例。
  打乱数据集的目的是为了确保模型训练的泛化能力，避免模型对数据的顺序产生依赖。
  1. **防止模型学习到数据的顺序**：如果数据是按照某种顺序排列的（例如，根据标签或某个特征排序），模型可能会学习到这个顺序，而不是从数据中学习到真正的模式。
  2. **提高模型的鲁棒性**：通过随机打乱，每个Mini-batch中的样本都是随机选择的，这有助于模型学习到更加鲁棒的特征，不会对特定的数据排列方式过度拟合。
  3. **均匀分布Mini-batch**：在一些情况下，数据可能会按照类

​	**举个例子**： 假设我们有一个关于手写数字识别的数据集，数据是按照数字顺序排列的，即所有的0在一起，所有的1在一起，依此类推。如果我们不打乱数据，模型在训练的时候可能会先看到很多0，然后是很多1，这可能导致模型在训练初期对某些数字过度拟合。通过打乱数据，我们可以确保每个Mini-batch中包含各种数字的样本，这样模型就可以学习到更加通用的特征，提高其在未知数据上的表现。

![image-20240626230519542](images/image-20240626230519542.png)

- **Partition**(分割)：将打乱后的（X，Y）划分为大小为`mini_batch_size`（此处为64）的小批处理。请注意，训练示例的数量并不总是可以被`mini_batch_size`整除。最后的小批量可能较小，但是你不必担心，当最终的迷你批处理小于完整的`mini_batch_size`时，它将如下图所示：

![image-20240626230605105](images/image-20240626230605105.png)

**练习**：实现`random_mini_batches`。我们为你编码好了shuffling部分。为了帮助你实现partitioning部分，我们为你提供了以下代码，用于选择$1^{st}$和$2^{nd}$小批次的索引：

```python
first_mini_batch_X = shuffled_X[:, 0 : mini_batch_size] 
#shuffled_X数组中提取第一个Mini-batch。shuffled_X[:, 0 : mini_batch_size]表示选取所有行（:代表所有行）和从第0列到mini_batch_size列之前的所有列。即提取了从第0列开始的mini_batch_size数量的列。
second_mini_batch_X = shuffled_X[:, mini_batch_size : 2 * mini_batch_size]  
#提取第二个Mini-batch。shuffled_X[:, mini_batch_size : 2 * mini_batch_size]表示选取所有行和从mini_batch_size列到2 * mini_batch_size列之前的所有列。如果mini_batch_size是100，那么将提取从第100列到第199列的数据作为第二个Mini-batch。
...
```

​	请注意，最后一个小批次的结果可能小于`mini_batch_size=64`。令$⌊s⌋$代表$s$向下舍入到最接近的整数（在Python中为`math.floor（s）`）。若示例总数不是`mini_batch_size = 64`的倍数，则将有$\lfloor \frac{m}{mini\_batch\_size}\rfloor个$带有完整示例的小批次，数量为64最终的一批次中的示例将是$m-mini_\_batch_\_size \times \lfloor \frac{m}{mini\_batch\_size}\rfloor。$

### 打乱算法	

在机器学习中，使用`permutation`列表来重新排列`X`矩阵中的列是一种确保数据随机性的方法，这对于训练过程中的泛化能力是非常重要的。下面我将解释这个过程是如何运行的，以及为什么它会导致数据被打乱。

**原理**： `permutation`列表包含了一个随机序列，这个序列是从0到`m-1`的整数，其中`m`是样本的总数。当我们使用`[:, permutation]`进行索引操作时，NumPy会根据`permutation`列表中的顺序来重新排列`X`矩阵的列。由于`permutation`是随机生成的，所以这个操作会打乱`X`矩阵中列的原始顺序。

**为什么会打乱**： 打乱的原因是因为`permutation`列表是随机生成的，没有遵循任何特定的顺序。这意味着每次生成`permutation`时，列的新顺序都是不可预测的。因此，当我们应用这个随机序列到`X`矩阵时，列的顺序就会被打乱。

**举例说明**： 假设我们有一个小的数据集`X`，它包含5个样本（列）和一些特征（行）：

现在，我们生成一个随机排列`permutation`，假设它是`[2, 4, 0, 3, 1]`。当我们应用这个排列到`X`时：

```PYTHON
X = [ [x11, x12, x13, x14, x15],
      [x21, x22, x23, x24, x25],
      [x31, x32, x33, x34, x35] ]

shuffled_X = X[:, [2, 4, 0, 3, 1]]
```
结果shuffled_X将是：

```PYTHON
shuffled_X = [ [x13, x15, x11, x14, x12],
               [x23, x25, x21, x24, x22],
               [x33, x35, x31, x34, x32] ]
```

```PYTHON
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    #使用np.random.permutation函数生成一个从0到m-1的随机排列的数组，转换成一个列表，被存储在permutation。
    permutation = list(np.random.permutation(m))
    #使用permutation列表来重新排列X矩阵中的列。X是一个包含所有输入特征的矩阵，每一列代表一个训练样本。通过索引操作[:, permutation]，X矩阵中的列被按照permutation列表中的顺序重新排列，从而打乱原始数据集的顺序。
    #原理上方说明
    shuffled_X = X[:, permutation]
    #使用相同的permutation列表来重新排列Y矩阵中的列，Y是一个包含所有标签的矩阵。然后使用reshape((1,m))方法将Y矩阵重塑成一个形状为(1,m)的二维数组。
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    #计算可以！完整！分割的Mini-batch的数量。math.floor向下取整函数。
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k+1) * mini_batch_size]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch) #列表

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0: #m不能被mini_batch_size整除，说明有剩余的样本。
        ### START CODE HERE ### (approx. 2 lines)
        #处理最后一个mini_batch，它的大小小于mini_batch_size。
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

X_assess, Y_assess, mini_batch_size = random_mini_batches_test_case()
mini_batches = random_mini_batches(X_assess, Y_assess, mini_batch_size)

print ("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
print ("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
print ("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
print ("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
print ("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape)) 
print ("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))#智能二维转一维
print ("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))
```

```PYTHON
mini_batches[0][0][0][0:3]
```

- `mini_batches[0]`：这会选择`mini_batches`列表中的第一个元素，即第一个Mini-batch。
- `mini_batches[0][0]`：因为每个Mini-batch是一个包含两个元素的元组（`X`数据和`Y`数据），这里选择的是第一个Mini-batch中的`X`数据。
- `mini_batches[0][0][0]`：假设`X`数据是一个多维数组，这里选择的是`X`数据的第一行。
- `mini_batches[0][0][0][0:3]`：最后，这个表达式使用切片操作`[0:3]`来选择第一行中的前三个元素。

所以，整个表达式`mini_batches[0][0][0][0:3]`的意思是从第一个Mini-batch的`X`数据中提取第一行的前三个元素。

output：

```PYTHON
shape of the 1st mini_batch_X: (12288, 64)
shape of the 2nd mini_batch_X: (12288, 64)
shape of the 3rd mini_batch_X: (12288, 20)
shape of the 1st mini_batch_Y: (1, 64)
shape of the 2nd mini_batch_Y: (1, 64)
shape of the 3rd mini_batch_Y: (1, 20)
mini batch sanity check: [ 0.90085595 -0.7612069   0.2344157 ]
```

- Shuffling和Partitioning是构建小批次数据所需的两个步骤
- 通常选择2的幂作为最小批量大小，例如16、32、64、128。

## 3 Momentum

​	冲量（Momentum）是一种用于加速梯度下降算法的技术，特别是在面对小批量（Mini-batch）数据时。在小批量梯度下降中，每次更新参数时只考虑了一个子集的样本，这可能导致更新的方向不是最优的，因为它只代表了数据的一个小部分。这就像是在嘈杂的环境中寻找最佳路径，会有很多不必要的波动和转弯。冲量方法通过考虑过去梯度的信息来减少这种波动，使得参数更新的路径更加平滑。

​	**冲量的工作原理**： 冲量方法将过去梯度的指数加权平均值作为当前更新的一部分。这意味着不仅当前的梯度会影响更新，过去的梯度也会以一定的比例影响更新。这可以用以下公式表示：
$$
v_{t} = \beta v_{t-1} + (1 - \beta) \nabla_{\theta}J(\theta) 
$$

$$
\theta = \theta - \alpha v_{t}
$$

​	其中，($ v_{t}$ ) 是当前的冲量，( $\beta$ ) 是冲量的超参数（通常设置为接近1的值，如0.9），( $\nabla_{\theta}J(\theta)$ ) 是当前梯度，( $\theta $) 是参数，( $\alpha$ ) 是学习率。

![image-20240626235130124](images/image-20240626235130124.png)

红色箭头显示了带冲量的小批次梯度下降步骤所采取的方向。蓝点表示每一步的梯度方向（相对于当前的小批量）。让梯度影响$v$而不是仅遵循梯度，然后朝$v$的方向迈出一步。

**为什么使用冲量**：

- **减少振荡**：冲量可以减少参数更新过程中的振荡，使得梯度下降路径更加平滑。
- **加速收敛**：冲量有助于加速梯度下降的收敛速度，特别是在梯度的方向一致时。
- **逃离局部最小值**：冲量可以帮助算法逃离不是很好的局部最小值，因为累积的冲量可能足以推动参数跳出局部最小值。

**举例说明**： 想象一个小球在一个有凹凸不平的表面上滚动，目标是找到最低点。如果没有冲量，小球可能会在每个小坑里停下来，这些小坑代表局部最小值。但是，如果小球有冲量，即使它进入一个小坑，之前积累的速度也可能足以让它跳出小坑，继续向更低的地方移动。在梯度下降中，这意味着即使当前的梯度很小，之前积累的梯度也可以帮助参数继续向最小值方向移动。

**练习**：初始化速度。速度v是一个Python字典，需要使用零数组进行初始化。它的键与grads词典中的键相同，即：为$l=1,...,L：$

```PYTHON
v["dW" + str(l+1)] = ... #(numpy array of zeros with the same shape as parameters["W" + str(l+1)])  
v["db" + str(l+1)] = ... #(numpy array of zeros with the same shape as parameters["b" + str(l+1)])
```

**注意**：迭代器l在for循环中从0开始，而第一个参数是v["dW1"]和v["db1"]（在上标中为“1”）。这就是为什么我们在“for”循环中将`l`转换为`l+1`的原因。

```PYTHON
# GRADED FUNCTION: initialize_velocity

def initialize_velocity(parameters):
    """
    Initializes the velocity as a python dictionary with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    
    Returns:
    v -- python dictionary containing the current velocity.
                    v['dW' + str(l)] = velocity of dWl
                    v['db' + str(l)] = velocity of dbl
    """
    
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    
    # Initialize velocity
    for l in range(L):
        ### START CODE HERE ### (approx. 2 lines)
        v["dW" + str(l+1)] = np.zeros(parameters['W' + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters['b' + str(l+1)].shape)
        ### END CODE HERE ###
        
    return v

parameters = initialize_velocity_test_case()

v = initialize_velocity(parameters)
print("v[\"dW1\"] = " + str(v["dW1"]))
print("v[\"db1\"] = " + str(v["db1"]))
print("v[\"dW2\"] = " + str(v["dW2"]))
print("v[\"db2\"] = " + str(v["db2"]))
```

output：

```PYTHON
v["dW1"] = [[0. 0. 0.]
 [0. 0. 0.]]
v["db1"] = [[0.]
 [0.]]
v["dW2"] = [[0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]]
v["db2"] = [[0.]
 [0.]
 [0.]]
```

**练习**：实现带冲量的参数更新。冲量更新规则是，对于$l=1,...,L$:
$$
\begin{cases}  
v_{dW^{[l]}} = \beta v_{dW^{[l]}} + (1 - \beta) dW^{[l]} \\  
W^{[l]} = W^{[l]} - \alpha v_{dW^{[l]}}  
\end{cases}\tag{3}
$$

$$
\begin{cases}  
v_{db^{[l]}} = \beta v_{db^{[l]}} + (1 - \beta) db^{[l]} \\  
b^{[l]} = b^{[l]} - \alpha v_{db^{[l]}}   
\end{cases}\tag{4}
$$

​	其中L是层数，β是动量，α是学习率。所有参数都应存储在`parameters`字典中。请注意，迭代器`l`在`for`循环中从0开始，而第一个参数是$W^{[1]}$和$b^{[1]}$（在上标中为“1”）。因此，编码时需要将`l`转化至`l+1`。

- $( dW^{[l]} )$：这是第( l )层权重参数的梯度，它衡量了损失函数相对于权重的变化率。在数学上，它是损失函数对权重的偏导数。
- $( db^{[l]} )$：这是第( l )层偏置参数的梯度，它衡量了损失函数相对于偏置的变化率。同样地，它是损失函数对偏置的偏导数。

​	关于**学习笔记**中举出的例子，仅仅为了方便理解。

​	在实际应用中，我们不一定希望 **dw** 或 **db** 中的任何一个变化更大或更小。相反，我们希望通过适当的学习率和动量参数来调整它们，使模型能够稳定地收敛到最小损失。

​	使用动量的优化算法（如您提到的冲量更新规则）有助于平滑梯度的变化，避免因梯度波动过大而导致的不稳定更新。动量项 **β** 起到了一个平滑器的作用，它会累积过去梯度的信息，从而在更新参数时考虑历史梯度，这有助于加速学习过程并减少震荡。

​	在选择 **β** 和 **α**（学习率）时，通常需要通过实验来找到最佳值。一个较高的 **β**（接近1）会使动量效果更强，有助于平滑梯度更新，但也可能导致过去的梯度对当前更新的影响过大。一个较低的 **β** 会减少这种影响。学习率 **α** 控制了每次更新的步长，过高可能导致震荡，过低则可能导致学习过程缓慢。

```PYTHON
# GRADED FUNCTION: update_parameters_with_momentum

def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    Update parameters using Momentum
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- python dictionary containing the current velocity:
                    v['dW' + str(l)] = ...
                    v['db' + str(l)] = ...
    beta -- the momentum hyperparameter, scalar
    learning_rate -- the learning rate, scalar
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- python dictionary containing your updated velocities
    """

    L = len(parameters) // 2 # number of layers in the neural networks
    
    # Momentum update for each parameter
    for l in range(L):
        
        ### START CODE HERE ### (approx. 4 lines)
        # compute velocities
        v["dW" + str(l + 1)] = beta*v["dW" + str(l + 1)]+(1-beta)*grads['dW' + str(l+1)]
        v["db" + str(l + 1)] = beta*v["db" + str(l + 1)]+(1-beta)*grads['db' + str(l+1)]
        # update parameters
        parameters["W" + str(l + 1)] = parameters['W' + str(l+1)] - learning_rate*v["dW" + str(l + 1)] 
        parameters["b" + str(l + 1)] = parameters['b' + str(l+1)] - learning_rate*v["db" + str(l + 1)] 
        ### END CODE HERE ###
        
    return parameters, v

parameters, grads, v = update_parameters_with_momentum_test_case()

parameters, v = update_parameters_with_momentum(parameters, grads, v, beta = 0.9, learning_rate = 0.01)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
print("v[\"dW1\"] = " + str(v["dW1"]))
print("v[\"db1\"] = " + str(v["db1"]))
print("v[\"dW2\"] = " + str(v["dW2"]))
print("v[\"db2\"] = " + str(v["db2"]))
```

output:

```PYTHON
W1 = [[ 1.62544598 -0.61290114 -0.52907334]
 [-1.07347112  0.86450677 -2.30085497]]
b1 = [[ 1.74493465]
 [-0.76027113]]
W2 = [[ 0.31930698 -0.24990073  1.4627996 ]
 [-2.05974396 -0.32173003 -0.38320915]
 [ 1.13444069 -1.0998786  -0.1713109 ]]
b2 = [[-0.87809283]
 [ 0.04055394]
 [ 0.58207317]]
v["dW1"] = [[-0.11006192  0.11447237  0.09015907]
 [ 0.05024943  0.09008559 -0.06837279]]
v["db1"] = [[-0.01228902]
 [-0.09357694]]
v["dW2"] = [[-0.02678881  0.05303555 -0.06916608]
 [-0.03967535 -0.06871727 -0.08452056]
 [-0.06712461 -0.00126646 -0.11173103]]
v["db2"] = [[0.02344157]
 [0.16598022]
 [0.07420442]]
```



**注意**：

- 速度用零初始化。因此，该算法将花费一些迭代来“提高”速度并开始采取更大的步骤。
- 如果$β=0$，则它变为没有冲量的标准梯度下降。

**怎样选择β?**

- 冲量$β$越大，更新越平滑，因为我们对过去的梯度的考虑也更多。但是，如果$β$太大，也可能使更新变得过于平滑。
- $β$的常用值范围是0.8到0.999。如果你不想调整它，则$β=0.9$通常是一个合理的默认值。
- 调整模型的最佳$β$可能需要尝试几个值，以了解在降低损失函数J的值方面最有效的方法。

**你应该记住**：

- 冲量将过去的梯度考虑在内，以平滑梯度下降的步骤。它可以应用于批量梯度下降，小批次梯度下降或随机梯度下降。
- 必须调整冲量超参数$β$和学习率$α$。
