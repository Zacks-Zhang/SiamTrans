# -*- coding: utf-8 -*
import math

import torch
import torch.nn as nn

from videoanalyst.model.backbone.backbone_base import VOS_BACKBONES, TRACK_BACKBONES
from videoanalyst.model.common_opr.common_block import conv_bn_relu, projector
from videoanalyst.model.module_base import ModuleBase


class creat_residual_block(nn.Module):
    def __init__(self, inplanes, outplanes, stride, has_proj=False):
        super(creat_residual_block, self).__init__()
        self.has_proj = has_proj
        if self.has_proj:
            self.proj_conv = conv_bn_relu(inplanes,
                                          outplanes,
                                          stride=stride,
                                          kszie=1,
                                          pad=0,
                                          has_bn=True,
                                          has_relu=False,
                                          bias=False)

        self.conv1 = conv_bn_relu(inplanes,
                                  outplanes,
                                  stride=stride,
                                  kszie=3,
                                  pad=1,
                                  has_bn=True,
                                  has_relu=True,
                                  bias=False)
        self.conv2 = conv_bn_relu(outplanes,
                                  outplanes,
                                  stride=1,
                                  kszie=3,
                                  pad=1,
                                  has_bn=True,
                                  has_relu=False,
                                  bias=False)

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        if self.has_proj:
            residual = self.proj_conv(residual)

        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        x = self.relu(x)
        return x


class create_bottleneck(nn.Module):
    """
    Modified Bottleneck : We change the kernel size of projection conv from 1 to 3.

    """
    def __init__(self, inplanes, outplanes, stride, has_proj=False):
        super(create_bottleneck, self).__init__()
        self.has_proj = has_proj
        if self.has_proj:
            self.proj_conv = conv_bn_relu(inplanes,
                                          outplanes,
                                          stride=stride,
                                          kszie=3,
                                          pad=1,
                                          has_bn=True,
                                          has_relu=False,
                                          bias=False)

        self.conv1 = conv_bn_relu(inplanes,
                                  outplanes,
                                  stride=stride,
                                  kszie=3,
                                  pad=1,
                                  has_bn=True,
                                  has_relu=True,
                                  bias=False)
        self.conv2 = conv_bn_relu(outplanes,
                                  outplanes,
                                  stride=1,
                                  kszie=3,
                                  pad=1,
                                  has_bn=True,
                                  has_relu=True,
                                  bias=False)
        self.conv3 = conv_bn_relu(outplanes,
                                  outplanes,
                                  stride=1,
                                  kszie=3,
                                  pad=1,
                                  has_bn=True,
                                  has_relu=False,
                                  bias=False)

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        if self.has_proj:
            residual = self.proj_conv(residual)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + residual
        x = self.relu(x)
        return x


@VOS_BACKBONES.register
class ResNet50_M(ModuleBase):

    default_hyper_params = {"pretrain_model_path": ""}

    def __init__(self, block=create_bottleneck):
        super(ResNet50_M, self).__init__()
        self.block = block
        self.stage1 = nn.Sequential(
            conv_bn_relu(3,
                         32,
                         stride=2,
                         kszie=3,
                         pad=3,
                         has_bn=True,
                         has_relu=True,
                         bias=False),
            conv_bn_relu(32,
                         32,
                         stride=1,
                         kszie=3,
                         pad=1,
                         has_bn=True,
                         has_relu=True,
                         bias=False),
            conv_bn_relu(32,
                         32,
                         stride=1,
                         kszie=3,
                         pad=1,
                         has_bn=True,
                         has_relu=True,
                         bias=False), nn.MaxPool2d(3, 2, 1, ceil_mode=False))
        self.stage2 = self.__make_stage(self.block, 32, 64, 3, 1)
        self.stage3 = self.__make_stage(self.block, 64, 128, 4, 2)
        self.stage4 = self.__make_stage(self.block, 128, 256, 6, 2)
        self.stage5 = self.__make_stage(self.block, 256, 512, 3, 2)

    def __make_stage(self, block, inplane, outplane, blocks, stride):
        layers = []
        layers.append(block(inplane, outplane, stride=stride, has_proj=True))
        for i in range(1, blocks):
            layers.append(block(outplane, outplane, 1, False))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        return x5


@VOS_BACKBONES.register
class ResNet18_M(ModuleBase):

    default_hyper_params = {"pretrain_model_path": ""}

    def __init__(self, block=creat_residual_block):
        super(ResNet18_M, self).__init__()
        self.block = block
        self.stage1 = nn.Sequential(
            conv_bn_relu(3,
                         32,
                         stride=2,
                         kszie=3,
                         pad=3,
                         has_bn=True,
                         has_relu=True,
                         bias=False),
            conv_bn_relu(32,
                         32,
                         stride=1,
                         kszie=3,
                         pad=1,
                         has_bn=True,
                         has_relu=True,
                         bias=False),
            conv_bn_relu(32,
                         32,
                         stride=1,
                         kszie=3,
                         pad=1,
                         has_bn=True,
                         has_relu=True,
                         bias=False), nn.MaxPool2d(3, 2, 1, ceil_mode=False))
        self.stage2 = self.__make_stage(self.block, 32, 64, 2, 1)
        self.stage3 = self.__make_stage(self.block, 64, 128, 2, 2)
        self.stage4 = self.__make_stage(self.block, 128, 256, 2, 2)
        self.stage5 = self.__make_stage(self.block, 256, 256, 2, 2)

    def __make_stage(self, block, inplane, outplane, blocks, stride):
        layers = []
        layers.append(block(inplane, outplane, stride=stride, has_proj=True))
        for i in range(1, blocks):
            layers.append(block(outplane, outplane, 1, False))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        return x5


@VOS_BACKBONES.register
class JointEncoder(ModuleBase):

    default_hyper_params = {"pretrain_model_path": ""}

    def __init__(self, basemodel):
        super(JointEncoder, self).__init__()
        self.basemodel = basemodel
        self.projector_corr_feature = projector(256, 256)

    def forward(self, saliency_image, corr_feature):
        corr_feature = self.projector_corr_feature(corr_feature)
        x1 = self.basemodel.stage1(saliency_image)
        x2 = self.basemodel.stage2(x1)
        x3 = self.basemodel.stage3(x2)
        x4 = self.basemodel.stage4(x3) + corr_feature
        x5 = self.basemodel.stage5(x4)
        return [x5, x4, x3, x2]


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        padding = 2 - stride

        if dilation > 1:
            padding = dilation

        dd = dilation
        pad = padding
        if downsample is not None and dilation > 1:
            dd = dilation // 2
            pad = dd

        self.conv1 = nn.Conv2d(inplanes, planes,
                               stride=stride, dilation=dd, bias=False,
                               kernel_size=3, padding=pad)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        padding = 2 - stride
        if downsample is not None and dilation > 1:
            dilation = dilation // 2
            padding = dilation

        assert stride == 1 or dilation == 1, \
            "stride and dilation must have one equals to zero at least"

        if dilation > 1:
            padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out

@TRACK_BACKBONES.register
class ResNet50(ModuleBase):
    default_hyper_params = dict(
        pretrain_model_path="",
    )

    def __init__(self, transform_input=False, block=Bottleneck, layers=[3, 4, 6, 3], used_layers=[4]):
        self.inplanes = 64
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0,  # 3
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 15x15

        self.feature_size = 128 * block.expansion
        self.used_layers = used_layers
        layer3 = True #  if 3 in used_layers else False
        layer4 = True #  if 4 in used_layers else False

        if layer3:
            self.layer3 = self._make_layer(block, 256, layers[2],
                                           stride=1, dilation=2)  # 15x15, 7x7
            self.feature_size = (256 + 128) * block.expansion
        else:
            self.layer3 = lambda x: x  # identity

        if layer4:
            self.layer4 = self._make_layer(block, 512, layers[3],
                                           stride=1, dilation=4)  # 7x7, 3x3
            self.feature_size = 512 * block.expansion
        else:
            self.layer4 = lambda x: x  # identity

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        dd = dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1 and dilation == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                if dilation > 1:
                    dd = dilation // 2
                    padding = dd
                else:
                    dd = 1
                    padding = 0
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=3, stride=stride, bias=False,
                              padding=padding, dilation=dd),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride,
                            downsample, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        # conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x_ = self.relu(x)
        x = self.maxpool(x_)
        # conv2
        p1 = self.layer1(x)
        # conv3
        p2 = self.layer2(p1)
        # conv4
        p3 = self.layer3(p2)
        # conv5
        p4 = self.layer4(p3)
        out = [x_, p1, p2, p3, p4]
        out = [out[i] for i in self.used_layers]
        if len(out) == 1:
            return out[0]
        else:
            return out



if __name__ == "__main__":
    print(VOS_BACKBONES)
    resnet_m = ResNet18_M()
    image = torch.rand((1, 3, 257, 257))
    print(image.shape)
    feature = resnet_m(image)
    print(feature.shape)
    print(resnet_m.state_dict().keys())
    #print(resnet_m)
