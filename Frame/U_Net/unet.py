import torch
import torch.nn as nn

from resnet import resnet50
from vgg import VGG16


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        #输入通道数，输出通道数，卷积核大小3*3，padding=1 此为same填充
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)#双线性插值上采样，比例因子为2，不改变通道数
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)#拼接 在第二个维度上
        outputs = self.conv1(outputs)#卷积
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)#再卷积
        outputs = self.relu(outputs)
        return outputs
    
class Unet(nn.Module):
    def __init__ (self, num_classes = 21,
                  pretrained = False, backbone = 'vgg'):
        super(Unet, self).__init__()
        if backbone == 'vgg': #骨干网络选择vgg
            self.vgg = VGG16(pretrained = pretrained)
            in_filters = [192, 384, 768, 1024] #拼接层的输入通道数
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained)
            in_filters = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        # upsampling  实例化上采样层
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])


        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)#卷积核大小为1*1，输出通道数为num_classes，即对每一个像素点进行分类
        self.backbone = backbone
    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)#得到不同特征图
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)
        up4 = self.up_concat4(feat4, feat5)#forward方法 4和5拼接卷积激活卷积激活
        up3 = self.up_concat3(feat3, up4)#重复
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)#传递

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1) #输出最后卷积
        
        return final
    #冻结参数骨干网络 不会更新
    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True