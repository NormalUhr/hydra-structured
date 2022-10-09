import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, conv_layer, in_planes, planes, stride=1, bn=True, kernel=3):
        super(BasicBlock, self).__init__()
        self.bn = bn
        if kernel == 3:
            self.conv1 = conv_layer(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=(not self.bn))
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = conv_layer(planes, planes, kernel_size=3,
                                    stride=1, padding=1, bias=(not self.bn))
        elif kernel == 2:
            self.conv1 = conv_layer(
                in_planes, planes, kernel_size=2, stride=stride, padding=1, bias=(not self.bn))
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = conv_layer(planes, planes, kernel_size=2,
                                   stride=1, padding=0, bias=(not self.bn))
        elif kernel == 1:
            self.conv1 = conv_layer(
                in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=(not self.bn))
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = conv_layer(planes, planes, kernel_size=1,
                                    stride=1, padding=0, bias=(not self.bn))
        else:
            exit("kernel not supported!")

        if self.bn:
            self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if self.bn:
                self.shortcut = nn.Sequential(
                    conv_layer(in_planes, self.expansion * planes,
                               kernel_size=1, stride=stride, bias=(not self.bn)),
                    nn.BatchNorm2d(self.expansion * planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    conv_layer(in_planes, self.expansion * planes,
                               kernel_size=1, stride=stride, bias=(not self.bn)),
                )

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        else:
            out = F.relu(self.conv1(x))
            out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet9_v1(nn.Module):
    def __init__(self, conv_layer, linear_layer, block, num_blocks=2, num_classes=10, in_planes=64, bn=True,
                 last_layer="avg"):
        super(ResNet9_v1, self).__init__()
        self.in_planes = in_planes
        self.bn = bn
        self.last_layer = last_layer
        self.conv1 = conv_layer(3, in_planes, kernel_size=3, stride=2, padding=0, bias=not self.bn)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(conv_layer, block, in_planes * 2, num_blocks, stride=2, bn=bn, kernel=3)
        self.layer2 = self._make_layer(conv_layer, block, in_planes * 2, num_blocks, stride=2, bn=bn, kernel=3)
        if self.last_layer == "avg":
            self.avg2d = nn.AvgPool2d(4)
            self.linear = linear_layer(in_planes * 2 * block.expansion, num_classes)
        elif self.last_layer == "dense":
            self.linear1 = linear_layer(in_planes * 2 * block.expansion * 16, 100)
            self.linear2 = linear_layer(100, num_classes)
        else:
            exit("last_layer type not supported!")

    def _make_layer(self, conv_layer, block, planes, num_blocks, stride, bn, kernel):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(conv_layer, self.in_planes, planes, stride, bn, kernel))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        if self.last_layer == "avg":
            out = self.avg2d(out)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        elif self.last_layer == "dense":
            out = torch.flatten(out, 1)
            out = F.relu(self.linear1(out))
            out = self.linear2(out)
        return out


class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, conv_layer, in_planes, planes, stride=1, bn=True, kernel=3):
        super(BasicBlock2, self).__init__()
        self.bn = bn
        if kernel == 3:
            self.conv1 = conv_layer(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=(not self.bn))
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = conv_layer(planes, planes, kernel_size=3,
                                   stride=1, padding=1, bias=(not self.bn))
        elif kernel == 2:
            self.conv1 = conv_layer(
                in_planes, planes, kernel_size=2, stride=stride, padding=1, bias=(not self.bn))
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = conv_layer(planes, planes, kernel_size=2,
                                   stride=1, padding=0, bias=(not self.bn))
        elif kernel == 1:
            self.conv1 = conv_layer(
                in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=(not self.bn))
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = conv_layer(planes, planes, kernel_size=1,
                                   stride=1, padding=0, bias=(not self.bn))
        else:
            exit("kernel not supported!")

        if self.bn:
            self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if self.bn:
                self.shortcut = nn.Sequential(
                    conv_layer(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=(not self.bn)),
                    nn.BatchNorm2d(self.expansion*planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    conv_layer(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=(not self.bn)),
                )

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        else:
            out = F.relu(self.conv1(x))
            out = self.conv2(out)
        out += self.shortcut(x)
        return out


class ResNet9_v4(nn.Module):
    def __init__(self, conv_layer, linear_layer, block, num_blocks=2, num_classes=10, in_planes=64, bn=True, last_layer="avg"):
        super(ResNet9_v4, self).__init__()
        self.in_planes = in_planes
        self.bn = bn
        self.last_layer = last_layer
        self.conv1 = conv_layer(3, in_planes, kernel_size=3,
                               stride=2, padding=0, bias=not self.bn)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(conv_layer, block, in_planes, num_blocks, stride=2, bn=bn, kernel=3)
        self.layer2 = self._make_layer(conv_layer, block, in_planes * 2, num_blocks, stride=2, bn=bn, kernel=3)
        self.layer3 = self._make_layer(conv_layer, block, in_planes * 2, num_blocks, stride=2, bn=bn, kernel=3)
        if self.last_layer == "avg":
            self.avg2d = nn.AvgPool2d(4)
            self.linear = linear_layer(in_planes * 2 * block.expansion, num_classes)
        elif self.last_layer == "dense":
            self.linear1 = linear_layer(in_planes * 2 * block.expansion * 16 // 4, 100)
            self.linear2 = linear_layer(100, num_classes)
        else:
            exit("last_layer type not supported!")

    def _make_layer(self, conv_layer, block, planes, num_blocks, stride, bn, kernel):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(conv_layer, self.in_planes, planes, stride, bn, kernel))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if self.last_layer == "avg":
            out = self.avg2d(out)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        elif self.last_layer == "dense":
            out = torch.flatten(out, 1)
            out = F.relu(self.linear1(out))
            out = self.linear2(out)
        return out


def ResNet4bpp(conv_layer, linear_layer, init_type, num_classes=10, bn=True):
    return ResNet9_v1(conv_layer, linear_layer, BasicBlock, num_blocks=2, num_classes=num_classes, in_planes=64, bn=bn, last_layer="dense")


def ResNet8bpp(conv_layer, linear_layer, init_type, num_classes=10, bn=True):
    return ResNet9_v1(conv_layer, linear_layer, BasicBlock2, num_blocks=4, num_classes=num_classes, in_planes=64, bn=bn, last_layer="dense")


def ResNet9bppp(conv_layer, linear_layer, init_type, num_classes=10, bn=True):
    return ResNet9_v4(conv_layer, linear_layer, BasicBlock2, num_blocks=3, num_classes=num_classes, in_planes=128, bn=bn, last_layer="dense")

