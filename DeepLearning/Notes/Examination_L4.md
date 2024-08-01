# 逐步实现卷积神经网络

使用numpy实现卷积（CONV）和池化（POOL）层，包括正向传播和反向传播（可选）。

**符号**：

- 上标$[l]$表示第$l^{th}$层的对象。
    \- 例如：$a^{[4]}$是$4^{th}$层的激活。 $W^{[5]}$和$b^{[5]}$是$5^{th}$层的参数。
- 上标$(i)$表示第$i^{th}$个示例中的对象。
    \- 示例：$x^{(i)}$是$i^{th}$个训练数据的输入。     
- 下标i表示$i^{th}$的向量输入。
    \- 示例：$a_i^{[l]}$表示$l$层中的$i^{th}$个激活，假设这是全连接层（FC）。      
- $n_H, n_W和n_C$分别表示给定层的通道的高度，宽度和数量。如果要引用特定层l，则还可以写入 $n_H^{[l]},n_W^{[l]},n_C^{[l]}。$
- $n_{H_{prev}}, n_{W_{prev}}和n_{C_{prev}}$ 分别表示前一层的高度，宽度和通道数。如果引用特定层l，则也可以表示为$n_H^{[l-1]}, n_W^{[l-1]}, n_C^{[l-1]}$。

## 作业大纲

- 卷积函数，包括：
    \- 零填充
    \- 卷积窗口
    \- 正向卷积
    \- 反向卷积（可选）
- 池化函数，包括：
    \- 正向池化
    \- 创建mask
    \- 分配值
    \- 反向池化（可选）

​	使用 `numpy`从头开始实现这些函数。对于每个正向函数，都有其对应的反向等式。因此，在正向传播模块的每一步中，都将一些参数存储在缓存中。这些参数用于在反向传播时计算梯度。

![image-20240801171917375](images/image-20240801171917375.png)

## 1 卷积神经网络

​	卷积层将输入体积转换为不同大小的输出体积

![image-20240801171924236](images/image-20240801171924236.png)

​	首先实现两个辅助函数：一个用于零填充，另一个用于计算卷积函数本身。

### 1.1 零填充

零填充将在图像的边界周围添加零：

![image-20240801172830261](images/image-20240801172830261.png)

填充的主要好处有：

- 允许使用CONV层而不必缩小其高度和宽度。这对于构建更深的网络很重要，因为高度/宽度会随着更深的层而缩小。一个重要、特殊的例子是"same"卷积，其中高度/宽度在一层之后被精确保留。
- 有助于我们将更多信息保留在图像边缘。如果不进行填充，下一层的一部分值将会受到图像边缘像素的干扰。

**练习**：实现以下函数，该功能将使用零填充处理一个批次X的所有图像数据。使用np.pad。注意，如果要填充维度为$(5,5,5,5,5)$的数组“a”，则第二维的填充为`pad = 1`，第四维的填充为`pad = 3`，其余为`pad = 0`，你可以这样做：

```python
a = np.pad(a, ((0,0), (1,1), (0,0), (3,3), (0,0)), 'constant', constant_values = (..,..))
```

```python
# GRADED FUNCTION: zero_pad

def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
    as illustrated in Figure 1.
    
    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    m表示批次中的图像数量。
    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
	(pad1, pad2)：为相应维度在前后各填充 pad1和pad2 个单位。
	mode='constant'：指定填充模式为常数填充。在数组的边缘添加常数值。
	constant_values=0：指定填充值为0。在数组的边缘添加的值都是0。
    """
    
    ### START CODE HERE ### (≈ 1 line)
    X_pad = np.pad(X, ((0, 0),(pad, pad),(pad, pad),(0, 0)), 'constant', constant_values=0)
    ### END CODE HERE ###
    
    return X_pad
np.random.seed(1)
x = np.random.randn(4, 3, 3, 2)
x_pad = zero_pad(x, 2)
print ("x.shape =", x.shape)
print ("x_pad.shape =", x_pad.shape)
print ("x[1,1] =", x[1,1])
print ("x_pad[1,1] =", x_pad[1,1])

fig, axarr = plt.subplots(1, 2)
axarr[0].set_title('x')
axarr[0].imshow(x[0,:,:,0])
axarr[1].set_title('x_pad')
axarr[1].imshow(x_pad[0,:,:,0])
```

output：

```python
x.shape = (4, 3, 3, 2)
x_pad.shape = (4, 7, 7, 2)
x[1,1] = [[ 0.90085595 -0.68372786]
 [-0.12289023 -0.93576943]
 [-0.26788808  0.53035547]]
x_pad[1,1] = [[0. 0.]
 [0. 0.]
 [0. 0.]
 [0. 0.]
 [0. 0.]
 [0. 0.]
 [0. 0.]]
```

![image-20240801174945979](images/image-20240801174945979.png)

### 1.2 卷积的单个步骤

实现卷积的单个步骤，其中将滤波器（卷积核）应用于输入的单个位置。这将用于构建卷积单元，该卷积单元：

- 占用输入体积
- 在输入的每个位置都应用滤波器
- 输出另一个体积（通常大小不同）

![Image Name](https://cdn.kesci.com/upload/image/q1q1pdeatw.gif?imageView2/0/w/960/h/960)

​	滤波器大小为2x2，步幅为1（步幅=每次滑动时移动窗口的数量）

​	在计算机视觉应用中，左侧矩阵中的每个值都对应一个像素值，我们将3x3滤波器与图像进行卷积操作，首先将滤波器元素的值与原始矩阵相乘，然后将它们相加。在练习的第一步中，你将实现卷积的单个步骤，相当于仅对一个位置应用滤波器以获得单个实值输出。

```py
# GRADED FUNCTION: conv_single_step

def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
    of the previous layer.
    
    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    
    Returns:
    Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
    """

    ### START CODE HERE ### (≈ 2 lines of code)
    # Element-wise product between a_slice and W. Add bias. 加偏移，w为卷积核
    s = np.multiply(a_slice_prev, W) + b
    # Sum over all entries of the volume s
    Z = np.sum(s)
    ### END CODE HERE ###

    return Z

np.random.seed(1)
a_slice_prev = np.random.randn(4, 4, 3)
W = np.random.randn(4, 4, 3)
b = np.random.randn(1, 1, 1)

Z = conv_single_step(a_slice_prev, W, b)
print("Z =", Z)
```

output：

```python
Z = -23.16021220252078
```

### 1.3 卷积神经网络--正向传递

​	在正向传递中，你将使用多个滤波器对输入进行卷积。每个“卷积”都会输出一个2D矩阵。然后，你将堆叠这些输出以获得输出：

![Image Name](https://cdn.kesci.com/upload/image/q1q2blbfn.gif?imageView2/0/w/960/h/960)

​	使用滤波器W卷积输入A_prev。此函数将上一层的激活输出（对于一批m个输入）A_prev作为输入，F表示滤波器/权重（W）和偏置向量（b），其中每个滤波器都有自己的（单个）偏置。最后，你还可以访问包含stride和padding的超参数字典。

1.要在矩阵“a_prev”（5,5,3）的左上角选择一个2x2切片，请执行以下操作：

```python
a_slice_prev = a_prev[0:2,0:2,:]
```

使用定义的`start/end`索引定义`a_slice_prev`时将非常有用。
2.要定义a_slice，你需要首先定义其角点 `vert_start`, `vert_end`, `horiz_start` 和 `horiz_end`。该图可能有助于你找到如何在下面的代码中使用h，w，f和s定义每个角。

![image-20240801181509783](images/image-20240801181509783.png)

卷积的输出维度与输入维度相关公式为：
$$
n_H = \lfloor \frac{n_{H_{prev}} - f + 2 \times pad}{stride} \rfloor +1
$$

$$
n_W = \lfloor \frac{n_{W_{prev}} - f + 2 \times pad}{stride} \rfloor +1
$$

$$
n_C = \text{number of filters used in the convolution}
$$

不必考虑向量化，只使用for循环实现所有函数。

```python
# GRADED FUNCTION: conv_forward

def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    
    ### START CODE HERE ###
    # Retrieve dimensions from A_prev's shape (≈1 line)  获取待卷积矩阵的形状
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve dimensions from W's shape (≈1 line) 获取滤波器形状 储存大小、待卷积通道数、输出通道数（卷积核数）
    (f, f, n_C_prev, n_C) = W.shape
    
    # Retrieve information from "hparameters" (≈2 lines) 获得超参数
    stride = hparameters['stride']
    pad = hparameters['pad']
    
    # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor. (≈2 lines)
    n_H = 1 + int((n_H_prev + 2 * pad - f) / stride) #依照上面公式
    n_W = 1 + int((n_W_prev + 2 * pad - f) / stride)
    
    # Initialize the output volume Z with zeros. (≈1 line)
    Z = np.zeros((m, n_H, n_W, n_C)) #定义输出大小，后方会返回
    
    # Create A_prev_pad by padding A_prev
    A_prev_pad = zero_pad(A_prev, pad) #填充 此处返回的是第一个参数是图片的张数，可以在下方循环中处理
    
    for i in range(m):                               # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i]                               # Select ith training example's padded activation
        for h in range(n_H):                           # loop over vertical axis of the output volume
            for w in range(n_W):                       # loop over horizontal axis of the output volume
                for c in range(n_C):                   # loop over channels (= #filters) of the output volume
                    
                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h * stride #列
                    vert_end = vert_start + f
                    horiz_start = w * stride #行
                    horiz_end = horiz_start + f
                    
                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]#给出卷积的范围
                  
                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
                    Z[i, h, w, c] = np.sum(np.multiply(a_slice_prev, W[:, :, :, c]) + b[:, :, :, c])#给输出矩阵填值
                                        
    ### END CODE HERE ###
    
    # Making sure your output shape is correct
    assert(Z.shape == (m, n_H, n_W, n_C))
    
    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)  #这些值都没有改变，都是参数传进来后传出去
    
    return Z, cache

np.random.seed(1)
A_prev = np.random.randn(10,4,4,3)
W = np.random.randn(2,2,3,8)
b = np.random.randn(1,1,1,8)
hparameters = {"pad" : 2,
               "stride": 1}

Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
print("Z's mean =", np.mean(Z))
print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])
```

output：

```PYTHON
Z's mean = 0.15585932488906465
cache_conv[0][1][2][3] = [-0.20075807  0.18656139  0.41005165]
```

最后，CONV层还应包含一个激活，此情况下，我们将添加以下代码行：

```
# Convolve the window to get back one output neuron  
Z[i, h, w, c] = ...  
# Apply activation  
A[i, h, w, c] = activation(Z[i, h, w, c])
```

在这里你不需要做这个。

## 2 池化层

### 2.1 正向池化

在同一函数中实现最大池化和平均池化。

池化层（如最大池化）处理多通道输入时，每个通道独立进行池化操作。假设输入特征图有 $(n_c)$ 个通道，每个通道的尺寸为$ (n_H \times n_W)$，池化窗口的尺寸为 $(f \times f)$，步长为 (s)。

**练习**：实现池化层的正向传播。请遵循下述提示。

**提示**：
由于没有填充，因此将池化的输出维度绑定到输入维度的公式为：
$$
n_H = \lfloor \frac{n_{H_{prev}} - f}{stride} \rfloor +1\\
n_W = \lfloor \frac{n_{W_{prev}} - f}{stride} \rfloor +1\\
n_C = n_{C_{prev}}
$$

```python
# GRADED FUNCTION: pool_forward

def pool_forward(A_prev, hparameters, mode = "max"):
    """
    Implements the forward pass of the pooling layer
    
    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
    """
    
    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve hyperparameters from "hparameters"
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride) #整数向下取整
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    
    # Initialize output matrix A
    A = np.zeros((m, n_H, n_W, n_C))              #输出矩阵 下方返回
    
    ### START CODE HERE ###
    for i in range(m):                         # loop over the training examples
        for h in range(n_H):                     # loop on the vertical axis of the output volume
            for w in range(n_W):                 # loop on the horizontal axis of the output volume
                for c in range (n_C):            # loop over the channels of the output volume
                    
                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    
                    # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] #[张、列、行、通道]
                    
                    # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. Use np.max/np.mean.
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice) #填值
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)
    
    ### END CODE HERE ###
    
    # Store the input and hparameters in "cache" for pool_backward()
    cache = (A_prev, hparameters)
    
    # Making sure your output shape is correct
    assert(A.shape == (m, n_H, n_W, n_C))
    
    return A, cache
np.random.seed(1)
A_prev = np.random.randn(2, 4, 4, 3)
hparameters = {"stride" : 1, "f": 4}

A, cache = pool_forward(A_prev, hparameters)
print("mode = max")
print("A =", A)
print()
A, cache = pool_forward(A_prev, hparameters, mode = "average")
print("mode = average")
print("A =", A)
```

output：

```PYTHON
mode = max
A = [[[[1.74481176 1.6924546  2.10025514]]]


 [[[1.19891788 1.51981682 2.18557541]]]]

mode = average
A = [[[[-0.09498456  0.11180064 -0.14263511]]]


 [[[-0.09525108  0.28325018  0.33035185]]]]
```

