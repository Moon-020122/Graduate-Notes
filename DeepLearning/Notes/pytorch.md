# Pytorch

## nn.Sequentia

`nn.Sequentia` 是 PyTorch 中的一个容器模块，用于将多个神经网络层按顺序组合在一起，它的主要作用是简化模型的定义和构建过程，适用于简单的前馈神经网络。

- **顺序连接**：将多个层按顺序连接起来，输入数据依次通过这些层进行处理。

### Ex

1. **直接传入层**

   ```python
   import torch.nn as nn
   
   model = nn.Sequential(
       nn.Conv2d(1, 20, 5),
       nn.ReLU(),
       nn.Conv2d(20, 64, 5),
       nn.ReLU()
   )
   print(model)
   ```

   `nn.Sequential` 将两个卷积层和两个 ReLU 激活函数按顺序连接起来。

2. **使用有序字典**

   ```python
   from collections import OrderedDict
   import torch.nn as nn
   
   model = nn.Sequential(OrderedDict([
       ('conv1', nn.Conv2d(1, 20, 5)),
       ('relu1', nn.ReLU()),
       ('conv2', nn.Conv2d(20, 64, 5)),
       ('relu2', nn.ReLU())
   ]))
   print(model)
   ```

   这种方法允许为每一层指定名称，便于访问和调试。

3. **动态添加层**

   ```python
   import torch.nn as nn
   
   model = nn.Sequential()
   model.add_module('conv1', nn.Conv2d(1, 20, 5))
   model.add_module('relu1', nn.ReLU())
   model.add_module('conv2', nn.Conv2d(20, 64, 5))
   model.add_module('relu2', nn.ReLU())
   print(model)
   ```

   这种方法可以动态地添加层，适用于需要根据条件构建模型的情况。

## nn.Conv2d

`nn.Conv2d` 是 PyTorch 中用于实现二维卷积操作的函数。

```python
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
```

1. **`in_channels`**：输入图像的通道数。例如，对于RGB图像，`in_channels` 为3。
2. **`out_channels`**：卷积操作后输出的通道数。这个参数决定了卷积核的数量。
3. **`kernel_size`**：卷积核的大小，可以是单个整数或一个元组。例如，`kernel_size=3` 表示 $3\times3$ 的卷积核。
4. **`stride`**：卷积操作的步长，默认为1。可以是单个整数或一个元组。
5. **`padding`**：在输入的每一边添加的零填充，默认为0。可以是单个整数或一个元组。
6. **`padding_mode`**：控制填充的模式，可以是 `'zeros'`、`'reflect'`、`'replicate'` 或 `'circular'`，默认为 `'zeros'`。
7. **`dilation`**：卷积核元素之间的间距，默认为1。可以是单个整数或一个元组，相当在相邻元素之间进行0填充，填充后为矩阵。
8. **`groups`**：控制输入和输出通道的连接方式，默认为1。`groups=1` 表示标准卷积，`groups=in_channels` 表示深度卷积。
9. **`bias`**：如果为True，则在输出中添加一个可学习的偏置，默认为True。

##  nn.Flatten

`nn.Flatten` 用于将多维张量展平为二维张量。

```python
torch.nn.Flatten(start_dim=1, end_dim=-1)
```

- **`start_dim`**：开始展平的维度，默认值为1。
- **`end_dim`**：结束展平的维度，默认值为-1（即最后一个维度）。

`nn.Flatten` 将指定范围内的维度展平为一个单独的维度。例如，如果输入张量的形状为 `(N, C, H, W)`，使用默认参数 `start_dim=1` 和 `end_dim=-1`，则输出张量的形状将变为 `(N, C*H*W)`。如果 `start_dim=2`输入张量的形状 `(N, C, H, W)` 将变为 `(N*C, H*W)`。

## nn.Parameter

`nn.Parameter` 用于将一个不可训练的 `Tensor` 转换为可训练的参数，并将其绑定到模型中。

```python
torch.nn.Parameter(data=None, requires_grad=True)
```

- **`data`**：初始化参数的 `Tensor` 数据。
- **`requires_grad`**：布尔值，表示是否需要计算梯度，默认为 `True`。

**自动注册**：当 `nn.Parameter` 被赋值为模型的属性时，它会自动添加到模型的参数列表中，并在调用 `parameters()` 方法时出现。

## torch.randn

`torch.randn` 用于生成服从标准正态分布（均值为0，标准差为1）的随机数张量。

```python
torch.randn(*size, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False) → Tensor
```

- **`*size`**：定义输出张量的形状，可以是多个整数或一个包含多个整数的列表或元组。
- **`generator`**：用于生成随机数的伪随机数生成器（可选）。
- **`out`**：输出张量（可选）。
- **`dtype`**：返回张量的数据类型（可选）。
- **`layout`**：返回张量的布局（默认为 `torch.strided`）。
- **`device`**：返回张量的设备（默认为当前设备）。
- **`requires_grad`**：如果为 `True`，则自动求导记录操作（默认为 `False`）。
- **`pin_memory`**：如果设置为 `True`，返回的张量将分配在固定内存中（仅适用于 CPU 张量，默认为 `False`）。

`*` 符号在函数参数前面有特殊的含义，通常用于解包参数。`*size` 表示将传递给函数的多个参数解包为一个元组。

## nn.Dropout

```python
torch.nn.Dropout(p=0.5, inplace=False)
```

- **`p`**：丢弃的概率，默认为0.5。这意味着在每次前向传播中，每个神经元有50%的概率被丢弃。
- **`inplace`**：布尔值，表示是否在原地进行操作。如果设置为 `True`，则会直接修改输入数据；如果设置为 `False`，则不对输入数据做修改。

## torch.cat

`torch.cat` 用于在给定维度上连接（拼接）多个张量，将一组张量沿着指定的维度连接在一起。

```python
torch.cat(tensors, dim=0, out=None)
```

- **`tensors`**：要连接的张量序列。所有张量必须具有相同的形状（除了连接的维度）。
- **`dim`**：指定连接的维度。例如，如果 `dim=0`，则在第一个维度上连接张量；如果 `dim=1`，则在第二个维度上连接张量。
  - **输入张量**：`tensor1` 和 `tensor2` 都是形状为 `(2, 2)` 的张量。
    - **`dim=0`**：在第一个维度上拼接，结果是一个形状为 `(4, 2)` 的张量。
    - **`dim=1`**：在第二个维度上拼接，结果是一个形状为 `(2, 4)` 的张量。
- **`out`**：可选的输出张量。如果提供，结果将存储在这个张量中。

## nn.TransformerEncoderLayer

`nn.TransformerEncoderLayer` 用于实现 Transformer 编码器的一个单独的层，通常包括一个自注意力机制和一个前馈神经网络，可能包含层归一化（Layer Normalization）和残差连接（Residual Connection）。

```python
torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', layer_norm_eps=1e-05, batch_first=False, norm_first=False, bias=True, device=None, dtype=None)
```

- **`d_model`**：输入和输出的特征维度。

- **`nhead`**：多头自注意力机制中的头数。

- **`dim_feedforward`**：前馈神经网络中隐藏层的维度，默认为2048。

- **`dropout`**：Dropout 的比例，默认为0.1。

- **`activation`**：前馈神经网络中的激活函数，可以是字符串（如 `'relu'` 或 `'gelu'`）或一个可调用对象，默认为 `'relu'`。

- **`layer_norm_eps`**：层归一化中的 $\epsilon$ 值，默认为$e^{-5}$。

- **`batch_first`**：如果为 `True`，则输入和输出张量的形状为 `(batch, seq, feature)`，否则为 `(seq, batch, feature)`，默认为 `False`。

- **`norm_first`**：如果为 `True`，则在注意力和前馈操作之前进行层归一化，否则在之后进行，默认为 `False`。

  - ### 前归一化（Pre-Layer Normalization）

    - **位置**：在自注意力和前馈神经网络操作之前进行层归一化。
    - 优点
      - **稳定训练**：前归一化可以使梯度更稳定，减少梯度消失或爆炸的风险。
      - **更快收敛**：由于梯度更稳定，模型通常可以更快地收敛。
    - 缺点
      - **性能差异**：在某些任务中，前归一化可能会导致模型性能略低于后归一化。

    ### 后归一化（Post-Layer Normalization）

    - **位置**：在自注意力和前馈神经网络操作之后进行层归一化。
    - 优点
      - **更高性能**：在某些任务中，后归一化可能会带来更好的性能，Transformer 论文中采用的方法。
    - 缺点
      - **梯度不稳定**：在深层网络中，后归一化可能导致梯度消失或爆炸的问题，影响训练稳定性。

- **`bias`**：如果设置为 `False`，则线性层和层归一化层将不会学习一个加性偏置，默认为 `True`。

## nn.TransformerEncoder

`nn.TransformerEncoder` 用于实现 Transformer 编码器的堆叠，通常由多个 `nn.TransformerEncoderLayer` 组成，每个层都包含一个自注意力机制和一个前馈神经网络。

```python
torch.nn.TransformerEncoder(encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True)
```

- **`encoder_layer`**：一个 `nn.TransformerEncoderLayer` 对象的实例或一个继承自 `nn.Module` 的自定义编码器层。
- **`num_layers`**：编码器层的数量，即堆叠的层数。
- **`norm`**：层归一化（Layer Normalization）的模块或 `None`。如果为 `None`，则不使用层归一化。
- **`enable_nested_tensor`**：如果为 `True`，输入将自动转换为嵌套张量，这在填充率较高时可以提高性能，默认为 `True`。
- **`mask_check`**：如果为 `True`，在执行前检查掩码的有效性，默认为 `True`。

应用效果如下：

- **输入张量**：形状为 `(batch_size, seq_len, d_model)` 的随机张量。
- **编码器层**：使用自注意力机制和前馈神经网络处理输入张量。
- **输出张量**：形状与输入张量相同，但经过多个编码器层的处理。

## nn.LayerNorm

`nn.LayerNorm` 用于实现层归一化（Layer Normalization）。

```python
torch.nn.LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True, device=None, dtype=None)
```

- **`normalized_shape`**：要归一化的维度。例如，如果输入的形状为 `(batch_size, seq_len, embedding_dim)`，则 `normalized_shape` 应该是 `[embedding_dim]`。
- **`eps`**：一个小的常数，用于避免分母为零，默认为 `1e-05`。
- **`elementwise_affine`**：如果为 `True`，则会有可学习的仿射参数$ γ 和 β$，默认为 `True`。
- **`device`**：指定张量所在的设备（如 CPU 或 GPU）。
- **`dtype`**：指定张量的数据类型。

### 层归一化

​	层归一化（Layer Normalization）旨在解决深度神经网络中的梯度消失和梯度爆炸问题。它通过对每一层的输入进行归一化，使得输入的均值为0，方差为1，从而提高模型的稳定性和泛化能力。

层归一化的基本思想是对每个隐藏层的输入进行归一化，而不是依赖于批量数据的统计特性。

1. **计算均值和方差**： 对于每个样本的每一层输入，计算其均值和方差：
   $$
   \mu = \frac{1}{H} \sum_{i=1}^{H} x_i\\
   \sigma^2 = \frac{1}{H} \sum_{i=1}^{H} (x_i - \mu)^2
   $$
   其中，$H$ 是输入的维度，$x_i$ 是输入的第 $i$ 个元素。

2. **归一化操作**： 使用计算得到的均值和方差，对输入进行归一化：
   $$
   \hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
   $$
   

   其中，$ϵ $是一个小的常数，用于避免分母为零。

3. **缩放和平移**： 引入可学习的参数 $γ 和 β$，对归一化后的输出进行缩放和平移：
   $$
   y_i = \gamma \hat{x}_i + \beta
   $$
   其中，$γ 和 β$ 是与输入维度相同的可学习参数。

## nn.Linear

`nn.Linear` 用于实现全连接层（也称为线性层），接受输入特征并通过线性变换生成输出特征。

```python
torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
```

- **`in_features`**：输入特征的数量。
- **`out_features`**：输出特征的数量。
- **`bias`**：如果设置为 `True`，则会学习一个偏置项，默认为 `True`。
- **`device`**：指定张量所在的设备（如 CPU 或 GPU）。
- **`dtype`**：指定张量的数据类型。

`nn.Linear` 实现了一个线性变换，即：
$$
Y = XW^T + b
$$
​	其中，X 是输入张量，W 是权重矩阵，b 是偏置向量，Y 是输出张量。

## nn.CrossEntropyLoss

`nn.CrossEntropyLoss` 用于计算交叉熵损失的函数，`nn.CrossEntropyLoss` 结合了 `nn.LogSoftmax` 和 `nn.NLLLoss`，适用于多分类任务。它首先对输入进行 softmax 操作，然后计算负对数似然损失。

```python
class torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
```

- **weight**: 可选的张量，为每个类别的损失赋予不同的权重。
- **size_average**: 已弃用。请使用 `reduction` 参数。
- **ignore_index**: 指定一个忽略的类别索引，计算损失时会忽略该类别。
- **reduce**: 已弃用。请使用 `reduction` 参数。
- **reduction**: 指定应用于输出的归约方式。可以是 `'none'`（不归约）、`'mean'`（求平均）或 `'sum'`（求和）。

## torch.optim.Adam

`torch.optim.Adam` 实现 Adam 优化算法的类。

```python
class torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
```

- **params**: 需要优化的参数或参数组的可迭代对象。
- **lr**: 学习率（默认值为 0.001）。
- **betas**: 用于计算梯度和梯度平方的移动平均的系数（默认值为 (0.9, 0.999)）。
- **eps**: 为了提高数值稳定性而加到分母中的项（默认值为 1e-08）。
- **weight_decay**: 权重衰减（L2惩罚）（默认值为 0）。
- **amsgrad**: 是否使用 AMSGrad 变种（默认值为 False）。

Adam 优化算法通过计算每个参数的自适应学习率来加速深度神经网络的训练，结合动量和自适应学习率的优点。

## model.parameters

`model.parameters()` 用于返回模型的所有可训练参数。

# TorchVision

## transforms.Compose

`transforms.Compose` 用于将多个图像变换操作组合在一起。

```python
class torchvision.transforms.Compose(transforms)
```

- **transforms**: 一个包含多个变换操作的列表。这些变换操作将按顺序应用于数据。

如在Vit中的应用

```python
transforms.Compose([ #图片增强
            transforms.ToPILImage(), #转换为PIL图像
            transforms.RandomRotation(degrees = 15), #随机旋转
            transforms.ToTensor(), #转换为张量
            transforms.Normalize([0.5], [0.5]) #归一化
        ])
```

## tqdm

`tqdm` 是一个用于显示进度条的 Python 库，适用于长时间运行的循环或迭代操作。

```python
tqdm(iterable=None, desc=None, total=None, leave=True, file=None, ncols=None, mininterval=0.1, maxinterval=10.0, miniters=None, ascii=None, disable=False, unit='it', unit_scale=False, dynamic_ncols=False, smoothing=0.3, bar_format=None, initial=0, position=None, postfix=None, unit_divisor=1000, write_bytes=False, lock_args=None, nrows=None, colour=None, delay=0, gui=False)
```

- **iterable**: 可迭代对象，如列表、元组等。
- **desc**: 进度条前的描述文字。
- **total**: 进度条的总迭代次数，如果未指定则自动推断。
- **leave**: 迭代完成后是否保留进度条（默认 `True`）。
- **file**: 输出进度条的文件对象（默认 `sys.stderr`）。
- **ncols**: 进度条的宽度（默认根据终端宽度动态调整）。
- **mininterval**: 最小进度显示更新间隔（默认 `0.1` 秒）。
- **maxinterval**: 最大进度显示更新间隔（默认 `10` 秒）。
- **miniters**: 最小进度显示更新迭代次数。
- **ascii**: 是否使用 ASCII 字符显示进度条（默认 `False`）。
- **disable**: 是否禁用进度条（默认 `False`）。
- **unit**: 进度条的单位（默认 `it`）。
- **unit_scale**: 是否自动缩放单位（默认 `False`）。
- **dynamic_ncols**: 是否动态调整进度条宽度（默认 `False`）。
- **smoothing**: 进度条更新的平滑度（默认 `0.3`）。
- **bar_format**: 自定义进度条格式。
- **initial**: 进度条的初始位置（默认 `0`）。
- **position**: 进度条的位置（用于多进度条）。
- **postfix**: 进度条后缀信息。
- **unit_divisor**: 单位缩放的除数（默认 `1000`）。
- **write_bytes**: 是否以字节模式写入（默认 `False`）。
- **lock_args**: 锁定参数（用于多线程）。
- **nrows**: 进度条的行数。
- **colour**: 进度条的颜色。
- **delay**: 延迟显示进度条（默认 `0` 秒）。
- **gui**: 是否在 GUI 中显示进度条（默认 `False`）。