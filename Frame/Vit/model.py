import torch
import torch.nn as nn

# 切小方块
class PatchEmbedding(nn.Module): #继承nn.Module
    #embed_dim: 嵌入维度，卷积核的输出通道数，卷积操作后的输出维度
    #patch_size: 切小方块的大小，卷积核的大小
    #num_patches: 切成多少个方块
    def __init__(self, in_channels, patch_size, embed_dim, num_patches, drpout): #构造函数
        super(PatchEmbedding, self).__init__() #调用父类初始化
        self.patcher = nn.Sequential( #切小方块，论文中用16*16的卷积相当于切块。
            nn.Conv2d(in_channels = in_channels, out_channels = embed_dim, kernel_size = patch_size, stride = patch_size), #卷积层
            nn.Flatten(start_dim = 2) #展平
        )
        self.cls_token = nn.Parameter(torch.randn(size = (1, 1, embed_dim)), requires_grad = True)#分类标记，拼接到每个patch的向量中。
        self.position_embedding = nn.Parameter(torch.randn(size=(1, num_patches+1, embed_dim)), requires_grad=True) #位置编码
        self.dropout = nn.Dropout(drpout)

    def forward(self, x): #图像预处理    x: 输入数据即图像数据   
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.patcher(x).permute(0, 2, 1) #卷积操作，展平，维度转换
        x = torch.cat([cls_token, x], dim = 1) #拼接cls_token
        x = x + self.position_embedding #添加位置编码
        x = self.dropout(x)
        return x
    
class Vit(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_patches, dropout, num_heads, activation, num_encoders, num_classes):
        super(Vit, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_dim, num_patches, dropout) #实例化PatchEmbedding
        encoder_layer = nn.TransformerEncoderLayer(d_model = embed_dim, nhead= num_heads, activation = activation, dropout = dropout, 
                                                   batch_first= True, norm_first= True) #encoder层
        self.encoder_layers = nn.TransformerEncoder(encoder_layer, num_layers = num_encoders)#将encoder层循环num_encoders次
        self.MLP = nn.Sequential(
            nn.LayerNorm(normalized_shape = embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes)
        )#多层感知机
    
    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encoder_layers(x)
        x = self.MLP(x[:, 0, :]) #仅仅取第二维度的第一个元素，原因在笔记中有记载。
        return x