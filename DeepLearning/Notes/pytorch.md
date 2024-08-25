# Pytorch

## nn.Sequentia

`nn.Sequentia` 是 PyTorch 中的一个容器模块，用于将多个神经网络层按顺序组合在一起，它的主要作用是简化模型的定义和构建过程，适用于简单的前馈神经网络。

- **顺序连接**：将多个层按顺序连接起来，输入数据依次通过这些层进行处理。

**示例**

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

### LayerNorm

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

## nn.AdaptiveAvgPool2d

`nn.AdaptiveAvgPool2d` 用于对二维输入进行自适应平均池化，将输入特征图调整到指定的输出大小，通常作用于输入张量的最后两个维度。

```python
torch.nn.AdaptiveAvgPool2d(output_size)
```

- `output_size`：目标输出大小，可以是一个整数或一个包含两个整数的元组。如果是整数，则输出的高度和宽度都将等于该值；如果是元组，则分别指定输出的高度和宽度。

## nn.AdaptiveAvgPool1d

`nn.AdaptiveAvgPool1d` 用于对 1D 张量进行自适应平均池化操作。它可以将输入张量的长度调整为指定的输出长度，无论输入张量的长度是多少，通常作用于输入张量的最后一个维度。

```python
class torch.nn.AdaptiveAvgPool1d(output_size)
```

- **`output_size`**：目标输出长度。可以是一个整数或包含一个整数的元组。

## nn.init.kaiming_normal_

`nn.init.kaiming_normal_` 用于以 Kaiming 正态分布（也称为 He 正态分布）初始化张量的权重以确保在深层网络中信号的方差不会随着层数的增加而消失或爆炸。

```python
torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
```

- `tensor`：需要初始化的张量。
- `a`：负斜率参数，默认为 0。对于 ReLU 激活函数，通常使用默认值。
- `mode`：可以是 `'fan_in'` 或 `'fan_out'`。`'fan_in'` 模式会保持前向传播时的方差一致，而 `'fan_out'` 模式会保持反向传播时的方差一致。
- `nonlinearity`：非线性激活函数的名称，默认为 `'leaky_relu'`。常见的选项包括 `'relu'` 和 `'leaky_relu'`。

## nn.init.constant_

`nn.init.constant_`用于将张量的所有元素初始化为指定的常数值。

```python
torch.nn.init.constant_(tensor, val)
```

- `tensor`：需要初始化的张量。
- `val`：用于初始化张量的常数值。

##  nn.Module

 `nn.Module`用于返回当前模块及其所有子模块的迭代器。主要作用是遍历当前模块及其所有子模块，包括嵌套的子模块。这在需要对整个模型进行操作时非常有用，例如初始化权重、设置模式（训练或评估）等。

##  nn.init.normal_

`nn.init.normal_` 用于以正态分布（高斯分布）初始化张量的权重。以下是这个函数的详细介绍：

```python
torch.nn.init.normal_(tensor, mean=0.0, std=1.0)
```

- `tensor`：需要初始化的张量。
- `mean`：正态分布的均值，默认为 0.0。
- `std`：正态分布的标准差，默认为 1.0。

## nn.init.trunc_normal_

`nn.init.trunc_normal_` 用于用截断的正态分布（truncated normal distribution）初始化张量的值。

```python
torch.nn.init.trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0)
```

- `tensor`：要初始化的张量。
- `mean`：正态分布的均值（默认值为0.0）。
- `std`：正态分布的标准差（默认值为1.0）。
- `a`：截断范围的下界（默认值为-2.0）。
- `b`：截断范围的上界（默认值为2.0）。









## nn.MaxPool2d

`nn.MaxPool2d` 用于在二维输入上执行最大池化操作。最大池化是一种下采样操作，常用于卷积神经网络（CNN）中，以减少特征图的尺寸，同时保留最显著的特征。

```python
torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
```

- `kernel_size`：池化窗口的大小，可以是一个整数或一个包含两个整数的元组。
- `stride`：池化窗口的步幅，默认为 `kernel_size`。可以是一个整数或一个包含两个整数的元组。
- `padding`：在输入的每一边添加的零填充，默认为 0。
- `dilation`：池化窗口元素之间的间距，默认为 1。
- `return_indices`：如果为 True，会返回最大值的索引，通常用于上采样。
- `ceil_mode`：如果为 True，会使用天花板函数而不是地板函数来计算输出形状。

**上采样** 增加数据的分辨率或尺寸，而 **下采样** 减少数据的分辨率或尺寸。

**天花板函数** 将数向上取整，而 **地板函数** 将数向下取整。

## nn.BatchNorm2d

`nn.BatchNorm2d` 用于对二维输入（如图像）进行批量归一化。

```python
torch.nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
```

- `num_features`：输入的特征数量（即通道数）。
- `eps`：一个小值，用于避免除零错误，默认为 `1e-5`。
- `momentum`：用于计算运行时均值和方差的动量，默认为 `0.1`。
- `affine`：如果为 `True`，则该层有可学习的仿射参数（即缩放和偏移），默认为 `True`。
- `track_running_stats`：如果为 `True`，则该层会跟踪运行时的均值和方差，默认为 `True`。

## nn.ReLU

`nn.ReLU` 用于应用修正线性单元（ReLU）激活函数。定义为$ f(x)=max(0,x)$，它将输入中的负值置为零，正值保持不变。

```python
torch.nn.ReLU(inplace=False)
```

- `inplace`：如果为 `True`，将会进行原地操作，即直接修改输入数据而不创建新的输出，默认为 `False`。

## nn.UpsamplingBilinear2d

`nn.UpsamplingBilinear2d` 用于对二维输入进行双线性插值上采样。双线性插值通过插值计算新的像素值，从而增加图像的分辨率。

```python
torch.nn.UpsamplingBilinear2d(size=None, scale_factor=None)
```

- `size`：目标输出大小，可以是一个整数或一个包含两个整数的元组。如果指定了 `size`，则 `scale_factor` 参数将被忽略。
- `scale_factor`：上采样的比例因子(输入以比例输出)，可以是一个浮点数或一个包含两个浮点数的元组。如果指定了 `scale_factor`，则 `size` 参数将被忽略。

## nn.Identity

`nn.Identity` 它不会对输入数据进行任何操作，直接输出输入的数据。

```python
class torch.nn.Identity(*args, **kwargs)
```

`nn.Identity` 通常不需要任何参数，但它可以接受任意数量的参数（`*args` 和 `**kwargs`），这些参数不会对其行为产生影响。

## nn.functional.pad

nn.functional 通常as为F。

`nn.functional.pad` 用于对张量进行填充操作，可以在张量的各个维度上添加指定数量的填充值。

```python
torch.nn.functional.pad(input, pad, mode='constant', value=0)
```

- **`input`**：需要进行填充的输入张量。
- **`pad`**：一个包含填充数量的元组，表示在每个维度上添加的填充值。
  - 例如，在PyTorch中，张量的默认维度顺序通常是 `[B, C, H, W]`，则按照顺序`(pad_left, pad_right, pad_top, pad_bottom，pad_front,pad_back)`。
  - 若张量的维度顺序通常是 `[B, H, W, C]`，则照顺序`(pad_front,pad_back，pad_left, pad_right, pad_top, pad_bottom，)`。

- `mode`填充模式，默认值是 `'constant'`。
  - `'constant'`：用常数值进行填充。
  - `'reflect'`：用输入张量的反射值进行填充。
  - `'replicate'`：用输入张量的边缘值进行填充。
- **`value`**：在 `'constant'` 模式下使用的填充值。默认值是 `0`。

## torch.linspace

`torch.linspace` 用于生成在指定范围内均匀分布的值。

```python
torch.linspace(start, end, steps, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
```

- **`start`**：生成值的起始点。
- **`end`**：生成值的结束点。
- **`steps`**：生成值的数量。
- **`out`**（可选）：输出张量。
- **`dtype`**（可选）：生成张量的数据类型。
- **`layout`**（可选）：张量的布局，默认是 `torch.strided`。
- **`device`**（可选）：生成张量的设备（如 CPU 或 GPU）。
- **`requires_grad`**（可选）：如果为 `True`，则记录对生成张量的操作，以便进行自动微分。

## nn.ModuleList

`nn.ModuleList` 是 PyTorch 中的一个容器，用于存储子模块（子层）的有序列表。它类似于 Python 的标准列表，但专门用于存储 `nn.Module` 对象，并且会自动注册这些子模块，使其成为模型的一部分。

```python
class torch.nn.ModuleList(modules=None)
```

- **`modules`**（可选）：一个包含 `nn.Module` 对象的可迭代对象，用于初始化 `ModuleList`。

**主要方法**

- **`append(module)`**：将一个 `nn.Module` 对象添加到 `ModuleList` 的末尾。
- **`extend(modules)`**：将一个包含多个 `nn.Module` 对象的可迭代对象添加到 `ModuleList` 的末尾。
- **`__getitem__(index)`**：根据索引获取 `ModuleList` 中的子模块。
- **`__setitem__(index, module)`**：根据索引设置 `ModuleList` 中的子模块。
- **`__len__()`**：返回 `ModuleList` 中子模块的数量。

## .view

`view` 是 PyTorch 中的一个方法，用于对张量进行重塑（reshape），即改变张量的形状而不改变其数据。

```python
Tensor.view(*shape)
```

- **`*shape`**：一个整数序列，表示新的形状。可以是多个整数参数或一个包含多个整数的元组。

**注意事项**

- 新的形状必须与原始张量的总元素数量一致，否则会引发错误。
- `view` 方法返回的是一个新的张量，但它与原始张量共享相同的数据存储。

## torch.zeros

`torch.zeros` 用于创建一个包含全零的张量。

```python
torch.zeros(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
```

- **`*size`**：一个整数序列，表示张量的形状。可以是多个整数参数或一个包含多个整数的元组。
- **`out`**（可选）：输出张量。
- **`dtype`**（可选）：生成张量的数据类型。默认是 `torch.float32`。
- **`layout`**（可选）：张量的布局，默认是 `torch.strided`。
- **`device`**（可选）：生成张量的设备（如 CPU 或 GPU）。默认是当前设备。
- **`requires_grad`**（可选）：如果为 `True`，则记录对生成张量的操作，以便进行自动微分。默认是 `False`。

## .contiguous

​	`contiguous` 是 PyTorch 中的一个方法，用于确保张量在内存中是连续存储的。某些操作（如转置或切片）可能会导致张量在内存中变得不连续，这会影响某些操作的性能或兼容性。

```python
Tensor.contiguous()
```

- **`Tensor`**：返回一个新的张量，该张量在内存中是连续存储的。

## torch.unsqueeze

`torch.unsqueeze` 用于在指定位置插入一个大小为1的维度。

```python
torch.unsqueeze(input, dim)
```

- `input`：输入的张量（tensor）。
- `dim`：指定插入新维度的位置。这个参数是一个整数，表示新维度将插入到现有维度的索引位置。

## .masked_fill

`masked_fill` 是 PyTorch 中的一个方法，用于根据给定的掩码（mask）将张量中的某些元素替换为指定的值。

```python
torch.Tensor.masked_fill(mask, value)
```

- `mask`：一个布尔张量（Boolean tensor），其形状与输入张量相同。`mask` 中为 `True` 的位置对应的输入张量中的元素将被替换。
- `value`：用于替换的值。

**示例：**

```PYTHON
attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
```

`torch.roll` 用于沿指定维度滚动（循环移位）张量的元素。

```python
torch.roll(input, shifts, dims=None)
```

- `input`：输入的张量（tensor）。
- `shifts`：一个整数或整数元组，表示每个维度上滚动的位移量。如果是一个整数，则所有维度都滚动相同的位移量；正数则是从左向右移动，负数是从右向左移动，上下方向同理。
- `dims`：一个整数或整数元组，表示要滚动的维度。如果未指定，则所有维度都将被滚动。

## torch.arange

`torch.arange` 是 PyTorch 中的一个函数，用于生成一个包含等间隔值的一维张量。

```python
torch.arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
```

- `start`：起始值（默认值为0）。
- `end`：结束值（不包含在生成的张量中）。
- `step`：步长（默认值为1）。
- `out`：可选的输出张量。
- `dtype`：数据类型。
- `layout`：张量布局（默认值为 `torch.strided`）。
- `device`：设备（如 `cpu` 或 `cuda`）。
- `requires_grad`：是否需要梯度（默认值为 `False`）。

## torch.stack

`torch.stack` 用于沿着新维度连接一系列张量。

```python
torch.stack(tensors, dim=0, *, out=None)
```

- `tensors`：一个包含多个张量的序列（如列表或元组）。这些张量必须具有相同的形状。
- `dim`：新维度的索引位置（默认值为0）。
- `out`：可选的输出张量。

**示例**

假设我们有三个形状为 `(2, 3)` 的张量：

```python
import torch

a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
b = torch.tensor([[7, 8, 9],
                  [10, 11, 12]])
c = torch.tensor([[13, 14, 15],
                  [16, 17, 18]])
```

```python
# 沿着第0维连接
stacked_0 = torch.stack([a, b, c], dim=0)
print(stacked_0.shape)  # 输出: torch.Size([3, 2, 3])
print(stacked_0)
# 输出:
# tensor([[[ 1,  2,  3],
#          [ 4,  5,  6]],
#         [[ 7,  8,  9],
#          [10, 11, 12]],
#         [[13, 14, 15],
#          [16, 17, 18]]])

# 沿着第1维连接
stacked_1 = torch.stack([a, b, c], dim=1)
print(stacked_1.shape)  # 输出: torch.Size([2, 3, 3])
print(stacked_1)
# 输出:
# tensor([[[ 1,  2,  3],
#          [ 7,  8,  9],
#          [13, 14, 15]],
#         [[ 4,  5,  6],
#          [10, 11, 12],
#          [16, 17, 18]]])

# 沿着第2维连接
stacked_2 = torch.stack([a, b, c], dim=2)
print(stacked_2.shape)  # 输出: torch.Size([2, 3, 3])
print(stacked_2)
# 输出:
# tensor([[[ 1,  7, 13],
#          [ 2,  8, 14],
#          [ 3,  9, 15]],
#         [[ 4, 10, 16],
#          [ 5, 11, 17],
#          [ 6, 12, 18]]])
```

## torch.meshgrid

`torch.meshgrid` 用于生成坐标网格。它接受一系列一维张量，并返回一个多维网格，每个维度对应于输入张量中的一个。

​	

```python
torch.meshgrid(*tensors, indexing='ij')
```

- `*tensors`：一系列一维张量。
- `indexing`：索引方式，可以是 `'ij'`（默认值）或 `'xy'`。
  - `'ij'` 表示矩阵索引，**输入两个一维tensor数据，且两个tensor数据类型相同，输出两个tensor数据（两个tensor的行数为第一个输入张量的元素个数，列数为第二个输入张量的元素个数）其中第一个输出张量填充第一个输入张量中的元素，各行元素相同；第二个输出张量填充第二个输入张量中的元素，各列元素相同。**
  - `'xy'` 表示笛卡尔坐标索引。**则输出两个tensor的第一个维度对应于第二个输入的元素个数，第二个维度对应于第一个输入的元素个数。其中第一个输出张量填充第一个输入张量中的元素，各列元素相同；第二个输出张量填充第二个输入张量中的元素，各行元素相同。**

**示例**

假设我们有两个一维张量：

```python
import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5])
```

使用 `torch.meshgrid` 生成坐标网格：

```python
# 使用矩阵索引 ('ij')
grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
print(grid_x)
# 输出:
# tensor([[1, 1],
#         [2, 2],
#         [3, 3]])
print(grid_y)
# 输出:
# tensor([[4, 5],
#         [4, 5],
#         [4, 5]])

# 使用笛卡尔坐标索引 ('xy')
grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
print(grid_x)
# 输出:
# tensor([[1, 2, 3],
#         [1, 2, 3]])
print(grid_y)
# 输出:
# tensor([[4, 4, 4],
#         [5, 5, 5]])
```

## register_buffer

​	`register_buffer` 是 PyTorch 中的一个方法，用于将一个张量注册为模型的缓冲区（buffer）。缓冲区是模型的一部分，但不会被视为模型的参数，因此在调用 `model.parameters()` 时不会返回这些缓冲区。缓冲区通常用于保存一些固定的、在训练过程中不会更新的值，例如均值和方差。

```python
register_buffer(name, tensor, persistent=True)
```

- `name`：缓冲区的名称（字符串）。
- `tensor`：要注册的张量。
- `persistent`：布尔值，表示缓冲区是否应在模型的状态字典（state_dict）中持久保存。默认值为 `True`。

## nn.Softmax

`nn.Softmax` 用于对输入张量的最后一个维度应用 Softmax 函数。Softmax 函数将输入张量的元素转换为概率分布，使得输出张量的元素在 [0, 1] 范围内，并且它们的和为1。

```python
torch.nn.Softmax(dim=None)
```

- `dim`：指定应用 Softmax 函数的维度。通常是最后一个维度。

## torch.unbind

`unbind` 用于沿指定维度将张量分解成多个子张量。

```python
torch.unbind(tensor, dim=0)
```

- `tensor`：输入的张量。
- `dim`：指定分解的维度（默认值为0）。

## .item

`.item()` 是 PyTorch 中的一个方法，用于将包含单个值的张量转换为 Python 标量。

```python
tensor.item()
```

`.item()` 方法没有任何参数。

## trunc_normal_

`trunc_normal_` 是一个用于生成截断正态分布随机数的函数，通常用于初始化神经网络的权重。

```python
torch.nn.init.trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0)
```

- `tensor`: 需要初始化的张量。
- `mean`: 正态分布的均值，默认值为 0.0。
- `std`: 正态分布的标准差，默认值为 1.0。
- `a`: 截断范围的下限，默认值为 -2.0。
- `b`: 截断范围的上限，默认值为 2.0。
  - 该函数将张量 `tensor` 的值初始化为从截断正态分布中抽取的随机数。截断正态分布是指在指定范围 `[a, b]` 内的正态分布。

## apply

`apply` 是 Python 中 pandas 库的一个函数，用于对 DataFrame 或 Series 的行或列应用函数。

```python
DataFrame.apply(func, axis=0, raw=False, result_type=None, args=(), **kwds)
```

- **func**: 要应用的函数。可以是 NumPy 函数、用户定义的函数或 lambda 函数。
- **axis**: 指定函数应用的轴。`0` 或 `'index'` 表示按列应用，`1` 或 `'columns'` 表示按行应用。默认值为 `0`。
- **raw**: 如果为 `True`，则传递给函数的是 ndarray 而不是 Series。默认值为 `False`。
- **result_type**: 当 `func` 返回一个 Series 时，指定结果的类型。可以是 `'expand'`、`'reduce'` 或 `'broadcast'`。默认值为 `None`。
- **args**: 传递给 `func` 的位置参数。
- **kwds**: 传递给 `func` 的关键字参数。

## torch.nn.fuctional.interpolate

`F.interpolate` 用于对张量进行上采样或下采样的函数。支持多种插值算法，可以改变图像或特征图的大小。

```python
torch.nn.functional.interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False)
```

- **`input` (Tensor)**：输入张量，形状为 `(N, C, H, W)` 或 `(N, C, D, H, W)`，其中 `N` 是批量大小，`C` 是通道数，`H`、`W` 和 `D` 分别是高度、宽度和深度。
- **`size` (int 或 Tuple[int])**：输出的空间大小，可以是一个整数或一个包含高度和宽度的元组。
- **`scale_factor` (float 或 Tuple[float])**：空间大小的倍数。如果是元组，其长度必须与输入的空间维度数相同。
- **`mode` (str)**：插值算法，支持 ‘nearest’、‘linear’、‘bilinear’、‘bicubic’、‘trilinear’ 和 ‘area’。默认值为 ‘nearest’。
- **`align_corners` (bool, optional)**：如果设置为 `True`，输入和输出张量的角像素将对齐。仅在使用 ‘linear’、‘bilinear’、‘bicubic’ 或 ‘trilinear’ 模式时有效。默认值为 `False`。
- **`recompute_scale_factor` (bool, optional)**：是否重新计算 `scale_factor`。默认值为 `None`。
- **`antialias` (bool, optional)**：是否应用抗锯齿。仅在 ‘bilinear’ 和 ‘bicubic’ 模式下有效。默认值为 `False`。

**示例**

```python
import torch
import torch.nn.functional as F

# 输入张量 (N, C, H, W)
x = torch.randn(1, 3, 240, 240)

# 使用 scale_factor 进行上采样
upscaled_x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
print(upscaled_x.shape)  # 输出: torch.Size([1, 3, 480, 480])

# 使用 size 进行下采样
downscaled_x = F.interpolate(x, size=(120, 120), mode='bilinear', align_corners=True)
print(downscaled_x.shape)  # 输出: torch.Size([1, 3, 120, 120])
```













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

# 其余常用函数

## isinstance

`isinstance` 用于检查一个对象是否是指定类或其子类的实例返回布尔值。

```python
isinstance(object, classinfo)
```

- `object`：要检查的对象。
- `classinfo`：类或类型，或者由类或类型组成的元组。

## **kwargs

表示可变关键字参数。`kwargs` 是 “keyword arguments” 的缩写。`**kwargs` 允许你在函数中传递任意数量的关键字参数。这些参数会被收集到一个字典中，字典的键是参数名，值是参数值。

**示例**

```python
def example_function(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

# 调用函数并传递多个关键字参数
example_function(name="Alice", age=30, city="New York")
```

```
name: Alice
age: 30
city: New York
```

## \_\_call\_\_

​	在 PyTorch 中，调用一个模块实例时，例如 `self.up_concat4(feat4, feat5)`，会自动调用该模块的 `forward` 方法。`nn.Module` 类重载了 `__call__` 方法，使得调用模块实例时会执行 `forward` 方法。

```python
class unetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(unetUp, self).__init__()
        # 初始化代码

    def forward(self, x1, x2):
        # 前向传播代码
        return x

    def another_method(self, x):
        # 其他方法
        pass

# 创建 unetUp 实例
up_concat4 = unetUp(in_channels, out_channels)

# 调用实例时会自动运行 forward 方法
up4 = up_concat4(feat4, feat5)
```

如果模块中有多个方法，直接调用这些方法，而不是通过实例调用。例如：

```python
class unetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(unetUp, self).__init__()
        # 初始化代码

    def forward(self, x1, x2):
        # 前向传播代码
        return x

    def another_method(self, x):
        # 其他方法
        return x * 2

# 创建 unetUp 实例
up_concat4 = unetUp(in_channels, out_channels)

# 调用 forward 方法
up4 = up_concat4(feat4, feat5)

# 调用另一个方法
result = up_concat4.another_method(feat4)
```

## 列表生成式

列表生成式（List Comprehension）是Python中的一种简洁且强大的语法，用于创建列表。它可以用一行代码来生成一个新的列表。

```python
[expression for item in iterable if condition]
```

- **expression**：生成列表元素的表达式。
- **item**：从可迭代对象（iterable）中取出的每个元素。
- **iterable**：一个可迭代对象，如列表、元组、字符串等。
- **condition**（可选）：一个条件表达式，用于筛选元素。

**示例**

1. **生成一个包含1到10的平方的列表**：

```python
squares = [x**2 for x in range(1, 11)]
print(squares)
```

```
[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
```

2. **嵌套列表生成式**：

```python
matrix = [[j for j in range(3)] for i in range(3)]
print(matrix)
```

```
[[0, 1, 2], [0, 1, 2], [0, 1, 2]]
```

## timm.models.layers.to_2tuple

`timm.models.layers.to_2tuple` 是 `timm` 库中的一个辅助函数，用于将输入转换为一个包含两个相同元素的元组。

```python
from timm.models.layers import to_2tuple

def to_2tuple(x):
    return (x, x) if isinstance(x, int) else x
```

- `x`: 可以是一个整数或一个包含两个元素的元组。
  - 如果 `x` 是一个整数，函数会将其转换为一个包含两个相同整数的元组。
  - 如果 `x` 已经是一个包含两个元素的元组，函数会直接返回该元组。

## rearrange

`rearrange` 是 `einops` 库中的一个函数，用于对张量（tensor）进行重排和变换。

```python
rearrange(tensor, pattern, **axes_lengths)
```

- `tensor`: 需要重排的张量。
- `pattern`: 描述重排方式的字符串模式。
- `axes_lengths`: 可选参数，用于指定轴的长度。
  - `rearrange` 函数通过指定的模式对张量的维度进行重排和变换。模式字符串使用了一种简洁的语法来描述输入和输出的维度关系。

1. **基本用法**:

   ```python
   import torch
   from einops import rearrange
   
   tensor = torch.randn(2, 3, 4)
   result = rearrange(tensor, 'b c h -> b h c')
   print(result.shape)  # 输出: torch.Size([2, 4, 3])
   ```

2. **增加维度**:

   ```python
   tensor = torch.randn(2, 3, 4)
   result = rearrange(tensor, 'b c h -> b c h 1')
   print(result.shape)  # 输出: torch.Size([2, 3, 4, 1])
   ```

3. **合并维度**:

   ```python
   tensor = torch.randn(2, 3, 4)
   result = rearrange(tensor, 'b c h -> (b c) h')
   print(result.shape)  # 输出: torch.Size([6, 4])
   ```

4. **指定轴长度**:

   ```python
   tensor = torch.randn(2, 3, 4)
   result = rearrange(tensor, 'b c h -> b (c h)', c=3, h=4)
   print(result.shape)  # 输出: torch.Size([2, 12])
   ```

​	代码中的用法

```python
x = x.view(B, H, W, C) #将之前展平的x重塑为B H W C
#将x最后一个维度拆分为三个部分，分别是 p1、p2 和 c。
#将拆分后的维度重新组合，形成新的维度顺序。
x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
```

## build_norm_layer

`build_norm_layer` 用于构建归一化层的函数,可以根据配置字典动态创建不同类型的归一化层。

```python
def build_norm_layer(cfg: Dict, num_features: int, postfix: Union[int, str] = '') -> Tuple[str, torch.nn.Module]:
```

- **`cfg` (Dict)**：归一化层的配置字典，包含以下键值对：
  - **`type` (str)**：归一化层的类型，例如 ‘BN’（BatchNorm）、‘GN’（GroupNorm）、‘LN’（LayerNorm）等。
  - **`requires_grad` (bool, optional)**：是否需要梯度更新，默认为 `True`。
  - **其他参数**：实例化归一化层所需的其他参数。
- **`num_features` (int)**：输入特征的通道数。
- **`postfix` (Union[int, str], optional)**：附加到归一化层名称后的后缀，用于区分不同的归一化层实例。

**返回值**

- **`Tuple[str, torch.nn.Module]`**：返回一个包含归一化层名称和实例的元组。例如，`('bn1', nn.BatchNorm2d)`。

**代码示例**

```python
build_norm_layer(norm_cfg, mla_channels)[1]
```

1. **`build_norm_layer(norm_cfg, mla_channels)`**：调用 `build_norm_layer` 函数，传入归一化层的配置字典 `norm_cfg` 和特征通道数 `mla_channels`。这个函数返回一个元组，例如 `('bn1', nn.BatchNorm2d)`。
2. **`[1]`**：通过索引 `[1]` 获取元组中的第二个元素，即归一化层的实例。