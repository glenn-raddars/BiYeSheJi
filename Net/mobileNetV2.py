import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_bn(inchannel, outchannel, stride):
    # 卷积标准化加激活函数
    return nn.Sequential(
        nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(outchannel),
        nn.ReLU6(inplace=True)
    )

def conv_1_1_bn(inchannel, outchannel):
    return nn.Sequential(
        nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(outchannel),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inchannel, outchannel, stride, expand_ratio) -> None: # expand_ratio是否需要进行升维
        super().__init__()
        self.stride = stride
        assert stride in [1, 2] # stride只可能是1，2否则抛出错误

        hidden_dim = round(expand_ratio * inchannel) # 四舍五入保留整数部分，还是一个浮点数

        self.use_res_connect = self.stride == 1 and inchannel == outchannel

        if expand_ratio == 1: # 不升维
            self.conv = nn.Sequential( # 不会有特征图尺寸变化
                nn.Conv2d(in_channels=hidden_dim ,out_channels=hidden_dim, kernel_size=3, stride=self.stride, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True), # 限制输出最大为6
                
                # 1*1卷积降维
                nn.Conv2d(in_channels=hidden_dim, out_channels=outchannel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(outchannel)
            )

        else:
            self.conv = nn.Sequential(
                # 先进行1*1升维
                nn.Conv2d(in_channels=inchannel, out_channels=hidden_dim, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),

                # 特征卷积
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=self.stride, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),

                # 最后降维
                nn.Conv2d(in_channels=hidden_dim, out_channels=outchannel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        # 主干与残差边相加
        if self.use_res_connect: # 使用残差边
            return x + self.conv(x)

        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, n_class = 1000, input_size = 224, width_mult = 1.) -> None:
        super().__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        interverted_residual_setting = [
            # 第一个参数是expand_ratio，表示是否进行1*1卷积升维。第二个参数是invertedResidualBlock的out_channel。第三个参数是这个打款里面有几个inverted...模块，第四个参数是步长stride
            # PSPNet输入是473， 473， 3的图片
            # 最开始标准化卷积 -> 237, 237, 32
            [1, 16, 1, 1], # 237, 237, 32 -> 237, 237, 16
            [6, 24, 2 ,2], # 237, 237, 16 -> 119, 119, 24
            [6, 32, 3, 2], # 119, 119, 24 -> 60, 60, 32
            [6, 64, 4, 2], # 60, 60, 32 -> 30, 30, 64
            [6, 96, 3, 1], # 30, 30, 64 -> 30, 30, 96
            [6, 160, 3, 2],# 30, 30, 96 -> 15, 15, 160
            [6, 320, 1, 1],# 15, 15, 160 -> 15, 15, 320
        ]

        assert input_size % 32 == 0 # input_size必须是32的倍数
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel

        # 先标准化卷积加激活函数
        self.features = [conv_bn(3, input_channel, 2)]

        # 开始构建MobileNetV2
        for e, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                # 因为只有一开始的时候才会存在是否需要对特征层进行高和宽的压缩的情况出现，所以要判断，即一个大模块只会有一次特征图尺寸压缩
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, e))

                else:
                    self.features.append(block(input_channel, output_channel, 1, e))
                input_channel = output_channel # 一层一层传递 
            
        # 收尾工作
        self.features.append(conv_1_1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features) # 直接将列表展开

        # 分类部分
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2) # 先求维度3的均值，再求维度2的均值
        x = self.classifier(x)
        return x