import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBnReluBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel, stride=1, groups=1, dilation=1, padding=None, name=None) -> None:
        super().__init__()
        if padding==None: # 为了符合resnet的图片尺寸, 这个填充会导致图片尺寸只跟步长有关，步长为2直接减半
            padding = (kernel - 1) // 2

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=kernel, stride=stride, padding=padding, dilation=dilation, groups=groups),
            nn.BatchNorm2d(outchannel),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

class Bottleneck(nn.Module):
    # 定义resNet中的残差结构
    def __init__(self, inchannel, outchannel, stride=1, shortcut=True, dilation=1, padding=None, name=None) -> None:
        super().__init__()
        self.expansion = 4 # 最后的输出通道数会变成4倍
        # 三次卷积归一化relu操作
        self.tripleConv = [
            ConvBnReluBlock(inchannel, outchannel, kernel=1),
            ConvBnReluBlock(outchannel, outchannel, kernel=3, stride=stride, padding=padding, dilation=dilation),
            ConvBnReluBlock(outchannel, outchannel*self.expansion, kernel=1, stride=1)
        ]
        self.tripleConv = nn.Sequential(*self.tripleConv)

        if not shortcut:
            # 不能直接相加，进行维度匹配
            self.short = ConvBnReluBlock(inchannel, outchannel * self.expansion, kernel=1, stride=stride)

        self.shortcut = shortcut
        self.num_out_channels = outchannel * self.expansion
    
    def forward(self, x):
        out = self.tripleConv(x)
        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        
        finout = out + short
        return F.relu(finout)

class ResNet101(nn.Module):
    # ResNet实现
    def __init__(self, block=Bottleneck, num_class=1000) -> None:
        super().__init__()
        # 4 种bottleneck，每个数量如下
        depth = [3, 4, 23, 3]
        # 4种bottleneck输入通道数
        num_channels = [64, 256, 512, 1024]
        # 4种bottleneck第一个卷积核的输出通道数，最终输出会变成4倍
        num_filters = [64, 128, 256, 512]

        # 刚开始的卷积操作3->64
        self.conv = ConvBnReluBlock(3, 64, 7, stride=2)
        # 全局池化
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        # 能否直接相加shortcut
        l1_shortcut = False

        # 第一种bottleneck，一共三个
        self.layer1 = self._make_layers(
            block=block,
            inchannel=num_channels[0],
            outchannel=num_filters[0],
            depth=depth[0],
            stride=1,
            shortcut=l1_shortcut
        )

        # 第二种bottlenneck，共4个
        self.layer2 = self._make_layers(
            block=block,
            inchannel=num_channels[1],
            outchannel=num_filters[1],
            depth=depth[1],
            stride=2
        )

        # 第三种bottleneck，共23个
        self.layer3 = self._make_layers(
            block=block,
            inchannel=num_channels[2],
            outchannel=num_filters[2],
            depth=depth[2],
            stride=2,
            dilation=2
        )

        # 第四种bottleneck，共3个
        self.layer4 = self._make_layers(
            block=block,
            inchannel=num_channels[3],
            outchannel=num_filters[3],
            depth=depth[3],
            stride=2,
            dilation=4
        )

        self.last_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.out_dim = num_filters[-1] * 4
        self.linear = nn.Linear(self.out_dim, num_class)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.last_pool(x)
        x = torch.reshape(x, shape=[-1, self.out_dim])
        x = self.linear(x)

        return x

    def _make_layers(self, block, inchannel, outchannel, depth, stride, dilation=1, shortcut=False, name=None):
        if dilation > 1:
            # 如果进行了空洞卷积的操作, 则进行填充大小为空洞的大小
            padding = dilation # 也是为了不改变图片尺寸，因为它只会在3*3的卷积上做空洞卷积
        else:
            padding = None

        # 构造ResNet层
        layers = [block(
            inchannel,
            outchannel,
            stride = stride, # 只有第一层需要修改图片尺寸
            shortcut=shortcut,
            dilation=dilation,
            padding=padding
        )]

        for i in range(1, depth):# 第一层已添加
            layers.append(block(
                outchannel * 4,
                outchannel,
                stride=1,
                dilation=dilation,
                padding=padding
                # 这个时候就可以直接相加了，因为输入输出都是4倍通道数
            ))

        return nn.Sequential(*layers)

if __name__ == "__main__":
    input = torch.ones([1, 3, 224, 224], dtype=torch.float)
    net = ResNet101()
    out = net(input)
    print(out.size())