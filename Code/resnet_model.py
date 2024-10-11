import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision

class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Convolution, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return F.elu(x, inplace=True)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(UpConv, self).__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv = Convolution(in_channels, out_channels, kernel_size, 1)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)

class GetDisparity(nn.Module):
    def __init__(self, in_channels):
        super(GetDisparity, self).__init__()
        self.conv = nn.Conv2d(in_channels, 2, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm2d(2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return 0.3 * self.sigmoid(x)

class ResnetModel(nn.Module):
    def __init__(self, in_channels, encoder='resnet18', pretrained=False):
        super(ResnetModel, self).__init__()
        resnet = getattr(torchvision.models, encoder)(pretrained=pretrained)
        self.first_conv = resnet.conv1
        self.first_bn = resnet.bn1
        self.first_relu = resnet.relu
        self.first_maxpool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.upconv6 = UpConv(512, 512, 3, 2)
        self.iconv6 = Convolution(512 + 256, 512, 3, 1)

        self.upconv5 = UpConv(512, 256, 3, 2)
        self.iconv5 = Convolution(256 + 128, 256, 3, 1)

        self.upconv4 = UpConv(256, 128, 3, 2)
        self.iconv4 = Convolution(128 + 64, 128, 3, 1)
        self.disp4_layer = GetDisparity(128)

        self.upconv3 = UpConv(128, 64, 3, 1)
        self.iconv3 = Convolution(64 + 2, 64, 3, 1)
        self.disp3_layer = GetDisparity(64)

        self.upconv2 = UpConv(64, 32, 3, 2)
        self.iconv2 = Convolution(32 + 2, 32, 3, 1)
        self.disp2_layer = GetDisparity(32)

        self.upconv1 = UpConv(32, 16, 3, 2)
        self.iconv1 = Convolution(16 + 2, 16, 3, 1)
        self.disp1_layer = GetDisparity(16)

        self.init_weights()

    def forward(self, x):
        x = self.first_conv(x)
        x = self.first_bn(x)
        x = self.first_relu(x)
        x = self.first_maxpool(x)

        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        up6 = self.upconv6(x4)
        concat6 = torch.cat((up6, x3), 1)
        iconv6 = self.iconv6(concat6)

        up5 = self.upconv5(iconv6)
        concat5 = torch.cat((up5, x2), 1)
        iconv5 = self.iconv5(concat5)

        up4 = self.upconv4(iconv5)
        concat4 = torch.cat((up4, x1), 1)
        iconv4 = self.iconv4(concat4)
        disp4 = self.disp4_layer(iconv4)

        up3 = self.upconv3(iconv4)
        concat3 = torch.cat((up3, disp4), 1)
        iconv3 = self.iconv3(concat3)
        disp3 = self.disp3_layer(iconv3)

        up2 = self.upconv2(iconv3)
        concat2 = torch.cat((up2, disp3), 1)
        iconv2 = self.iconv2(concat2)
        disp2 = self.disp2_layer(iconv2)

        up1 = self.upconv1(iconv2)
        concat1 = torch.cat((up1, disp2), 1)
        iconv1 = self.iconv1(concat1)
        disp1 = self.disp1_layer(iconv1)

        return disp1, disp2, disp3, disp4