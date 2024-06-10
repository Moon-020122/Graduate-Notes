# 神经网络初始化

欢迎来到“改善深度神经网络”的第一项作业。

训练神经网络需要指定权重的初始值，而一个好的初始化方法将有助于网络学习。

如果你完成了本系列的上一课程，则可能已经按照我们的说明完成了权重初始化。但是，如何为新的神经网络选择初始化？在本笔记本中，你能学习看到不同的初始化导致的不同结果。

好的初始化可以：

- 加快梯度下降、模型收敛
- 减小梯度下降收敛过程中训练（和泛化）出现误差的几率

首先，运行以下单元格以加载包和用于分类的二维数据集。

## 1-神经网络模型

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

