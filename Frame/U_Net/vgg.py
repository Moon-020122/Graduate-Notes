import torch.nn as nn
from torch.hub import load_state_dict_from_url


class VGG(nn.Module):
    def __init__(self, features, num_classes = 1000):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))#平均池化，输出7*7
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),#全连接层
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),#softmax分类
        )
        
    def forward(self,x):
        feat1 = self.features[:4](x)#输入 x 通过 self.features 前 4 层，得到第一个特征图 feat1。  64 64   一个层有卷积层和RELU激活层，所以代表64 64 层
        feat2 = self.features[4:9](feat1)#128 128 
        feat3 = self.features[9:16](feat2)#256 256 256 
        feat4 = self.features[16:23](feat3)
        feat5 = self.features[23:-1](feat4)#第 23 层到倒数第二层，得到第五个特征图 feat5,最后一个最大池化未使用。
        return [feat1, feat2, feat3, feat4, feat5] #后续残差连接给加强结构特征
    #初始化各种层
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')#初始化张量权重反向传播方差一致
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)#初始化偏置
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm = False, in_channels = 3):
    layers = []
    for v in cfg:
        if v == 'M':#最大池化
            layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]#最大池化层,向下取整，步长为2，卷积核大小为2*2
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size = 3, padding = 1)#输入通道数3，输出通道数v，卷积核大小3*3，padding=1 此为same填充
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace = True)]#BN层（输入特征数目）
            else:
                layers += [conv2d, nn.ReLU(inplace = True)]
            in_channels = v
    return nn.Sequential(*layers)
# 512,512,3 -> 512,512,64 -> 256,256,64(M) -> 256,256,128 -> 128,128,128(M) -> 128,128,256 -> 64,64,256(M)
# 64,64,512 -> 32,32,512(M) -> 32,32,512(输出)
cfgs = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}


def VGG16(pretrained, in_channels = 3, **kwargs):
    model = VGG(make_layers(cfgs["D"], batch_norm = False, in_channels = in_channels), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/vgg16-397923af.pth', progress = True)
        model.load_state_dict(state_dict)
    return model