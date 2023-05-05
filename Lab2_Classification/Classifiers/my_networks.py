from torch import nn
import torch.nn.functional as F

from Utilties.setup import import_torch

import_torch()  # 导入torch位置到sys环境变量中


# 最原始的残差块
class ResBlock(nn.Module):
    expansion = 1  # 残差块中卷积层输出通道数的倍数

    def __init__(
        self,
        in_channel,
        out_channel,
        stride=1,
        downsample=None,  # 是否需要下采样
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    # 前向传播
    def forward(self, x):
        identity = x
        # 如果downsample不为None，就对输入进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)

        # 原始的实现就是在第二个卷积的relu之前，bn之后残差相加
        out = self.conv1(x)
        # print(out.shape)
        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.bn2(self.conv2(out1))
        out = self.relu(out2 + identity)

        # 替代：使用highway结构，有一个门控机制
        # out1 = self.relu(self.bn1(self.conv1(x)))
        # out2 = self.bn2(self.conv2(out1))
        # self.gate=F.sigmoid(out2)
        # out = self.relu(out2 * self.gate + identity * (1 - self.gate))

        return out
        #


# 二级残差块
class ResBlockG1(nn.Module):
    expansion = 1  # 残差块中卷积层输出通道数的倍数
    def __init__(
        self,
        block,
        block_num,
        in_channel,
        stride=1,
    ) -> None:
        super().__init__()
        self.resBlocksG1 = self._make_sublayers(
            self, block, in_channel, block_num, stride
        )
    

        # 整个convN_x块进行残差连接需要的下采样
        self.curDownsample = nn.Sequential(
            nn.Conv2d(
                self.in_channel,
                block.expansion * in_channel,
                1,
                stride,
                bias=False,
            ),
            nn.BatchNorm2d(block.expansion * in_channel),
        )

    def _make_sublayers(self, block, in_channel, block_num, stride):
        downsample = None
        # 如果需要下采样，就对identity进行下采样
        if stride != 1 or self.in_channel != block.expansion * in_channel:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channel,
                    block.expansion * in_channel,
                    1,
                    stride,
                    bias=False,
                ),
                nn.BatchNorm2d(block.expansion * in_channel),
            )

        layers = []
        layers.append(
            block(self.in_channel, block.expansion * in_channel, stride, downsample)
        )
        self.in_channel = block.expansion * in_channel
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, in_channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        identity = self.curDownsample(x)
        out = self.resBlocksG1(x)
        out = self.relu(out + identity)
        
        return out
    


# 更高级的残差块
class ResBlockG2plus(nn.Module):
    def __init__(
        self,
        block,
        block_num,
        in_channel,

    ) -> None: 
        super().__init__()

        self.resBlkG1_1 = block()   #ResBlockG1()
        self.resBlkG1_2 = block()

        # 整个convN_x块进行残差连接需要的下采样
        self.curDownsample = nn.Sequential(
            nn.Conv2d(
                self.in_channel,
                block.expansion * in_channel,
                1,
                stride,
                bias=False,
            ),
            nn.BatchNorm2d(block.expansion * in_channel),
        )

    def forward(self, x):
        identity = self.curDownsample(x)

        out = self.resBlkG1_1(x)
        out = self.resBlkG1_2(out)

        out = self.relu(out + identity)
        
        return out






class MyResNet18(nn.Module):
    def __init__(
        self,
        block,
        block_num,
        in_channel=64,
        num_classes=10,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, in_channel, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.in_channel = in_channel
        self.fc_last = nn.Linear(512, num_classes)

        # self.resBlocks=nn.ModuleList()

        self.resBlocks = nn.ModuleList(
            [
                self._make_layer(block, _in_channel, block_num[i], _stride)
                for _in_channel, i, _stride in zip(
                    [
                        self.in_channel,
                        self.in_channel * 2,
                        self.in_channel * 4,
                        self.in_channel * 8,
                    ],
                    range(4),
                    [1, 2, 2, 2],
                )
            ]
        )

        # 初始化卷积层的权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_layer(self, block, in_channel, block_num, stride=1):
        downsample = None
        # 如果需要下采样，就对identity进行下采样
        if stride != 1 or self.in_channel != block.expansion * in_channel:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channel,
                    block.expansion * in_channel,
                    1,
                    stride,
                    bias=False,
                ),
                nn.BatchNorm2d(block.expansion * in_channel),
            )

        layers = []
        layers.append(
            block(self.in_channel, block.expansion * in_channel, stride, downsample)
        )
        self.in_channel = block.expansion * in_channel
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, in_channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        # print(x.shape)
        x = self.maxpool(x)
        # print(x.shape)

        for resBlock in self.resBlocks:
            x = resBlock(x)
            # print(x.shape)

        # print('fin')
        # x =torch.flatten(x, 1)
        # x= nn.Linear(512, 10)(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc_last(x)
        return x


# class MyResNext(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()


class MyMultiHierResNet(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
