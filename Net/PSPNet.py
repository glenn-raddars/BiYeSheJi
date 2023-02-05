import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import ResNet101

class _PSPModle(nn.Module):
    def __init__(self, inchannel, pool_sizes, norm_layer) -> None:
        super().__init__()
        outchannel = inchannel // len(pool_sizes) # 每个池化后的输出通道数相等
        self.stages = nn.ModuleList([self._make_stages(inchannel, outchannel, pool_size, norm_layer) for pool_size in pool_sizes])

        # 最后阶段的整合输出
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=(inchannel + outchannel * len(pool_sizes)), out_channels=outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU6(inplace=True),
            nn.Dropout2d(0.1)
        )


    def _make_stages(self, inchannel, outchannel, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz) # 自动平均池化层，给出输出尺寸自己寻找合适的卷积核大小和步长
        # 卷积标准化加激活函数
        conv = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=1, stride=1, bias=False)
        bn = norm_layer(outchannel)
        relu = nn.ReLU6(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages]) # 对每一个平均池化后的特征图上采样，让其恢复到与原特征图相同尺寸，方便拼接
        output = self.bottleneck(torch.cat(pyramids, dim=1)) # 拼接通道数，最后收尾
        return output

class PSPNet(nn.Module):
    def __init__(self, num_class, use_aux=True) -> None:
        super().__init__()
        self.use_aux = use_aux
        masterfeature = ResNet101()
        # self.backbone = nn.Sequential(
        #     masterfeature.conv,
        #     masterfeature.pool,
        #     masterfeature.layer1,
        #     masterfeature.layer2,
        #     masterfeature.layer3,
        #     masterfeature.layer4
        # )

        self.initResNet = nn.Sequential(
            masterfeature.conv,
            masterfeature.pool,
        )
        self.layer1 = masterfeature.layer1
        self.layer2 = masterfeature.layer2
        self.layer3 = masterfeature.layer3
        self.layer4 = masterfeature.layer4 # 7*7*2048 以 224*224*3来说

        num_channels = 2048
        self.pspmodule = _PSPModle(num_channels, [1, 2, 3, 6], nn.BatchNorm2d) # 7*7*4096

        # 分类器
        self.classifier = nn.Sequential(
            self.pspmodule,
            nn.Conv2d(num_channels//4, num_class, kernel_size=1) # 7*7*512
        )

        # 辅助分支
        self.aux = nn.Sequential(
            nn.Conv2d(num_channels//2, num_channels//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels//4),
            nn.ReLU(),
            nn.Conv2d(num_channels//4, num_class, kernel_size=1)
        )

    def forward(self, x):
        h, w = x.size()[2], x.size()[3]
        # ResNet提取特征
        x = self.initResNet(x)
        x = self.layer1(x)
        x = self.layer2(x)
        aux_x = self.layer3(x)
        feature = self.layer4(aux_x)
        # PspNet语义分割
        out = self.classifier(feature)
        out = F.interpolate(out, size=(h, w), mode="bilinear",  align_corners=True)
        if self.use_aux: # 如果使用辅助分支
            aux = self.aux(aux_x) # [num_class, 1, 1]
            aux = F.interpolate(aux, size=(h, w), mode="bilinear", align_corners=True)
            return out, aux
        return out

if __name__ == "__main__":
    input = torch.ones([2, 3, 320, 480], dtype=torch.float)
    net = PSPNet(num_class=21, use_aux=True)
    out, aux = net(input)
    print(out.size())
    print(aux.size())

    