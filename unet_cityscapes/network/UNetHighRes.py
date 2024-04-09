import torch.nn as nn
from unet_cityscapes.network.modules import DoubleConv2D, OutLayer
import torch
from torchvision.transforms import CenterCrop


class UNetHighRes(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.left_conv_1 = DoubleConv2D(in_channels, 64)
        self.left_pool_1 = nn.MaxPool2d(2)
        self.left_conv_2 = DoubleConv2D(64, 128)
        self.left_pool_2 = nn.MaxPool2d(2)
        self.left_conv_3 = DoubleConv2D(128, 256)
        self.left_pool_3 = nn.MaxPool2d(2)
        self.left_conv_4 = DoubleConv2D(256, 512)
        self.left_pool_4 = nn.MaxPool2d(2)
        self.bottom_conv = DoubleConv2D(512, 1024)
        self.right_up_4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.cropped_left_conv_4 = CenterCrop((112, 240))
        self.right_conv_4 = DoubleConv2D(1024, 512)
        self.right_up_3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.cropped_left_conv_3 = CenterCrop((216, 472))
        self.right_conv_3 = DoubleConv2D(512, 256)
        self.right_up_2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.cropped_left_conv_2 = CenterCrop((424, 936))
        self.right_conv_2 = DoubleConv2D(256, 128)
        self.right_up_1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.cropped_left_conv_1 = CenterCrop((840, 1864))
        self.right_conv_1 = DoubleConv2D(128, 64)
        self.out = OutLayer(64, out_channels)

    def forward(self, input):
        left_conv_1 = self.left_conv_1(input)
        left_pool_1 = self.left_pool_1(left_conv_1)
        left_conv_2 = self.left_conv_2(left_pool_1)
        left_pool_2 = self.left_pool_2(left_conv_2)
        left_conv_3 = self.left_conv_3(left_pool_2)
        left_pool_3 = self.left_pool_3(left_conv_3)
        left_conv_4 = self.left_conv_4(left_pool_3)
        left_pool_4 = self.left_pool_4(left_conv_4)
        bottom_conv = self.bottom_conv(left_pool_4)
        right_up_4 = self.right_up_4(bottom_conv)
        cat_4 = torch.cat((right_up_4, self.cropped_left_conv_4(left_conv_4)), dim=1)
        right_conv_4 = self.right_conv_4(cat_4)
        right_up_3 = self.right_up_3(right_conv_4)
        cat_3 = torch.cat((right_up_3, self.cropped_left_conv_3(left_conv_3)), dim=1)
        right_conv_3 = self.right_conv_3(cat_3)
        right_up_2 = self.right_up_2(right_conv_3)
        cat_2 = torch.cat((right_up_2, self.cropped_left_conv_2(left_conv_2)), dim=1)
        right_conv_2 = self.right_conv_2(cat_2)
        right_up_1 = self.right_up_1(right_conv_2)
        cat_1 = torch.cat((right_up_1, self.cropped_left_conv_1(left_conv_1)), dim=1)
        right_conv_1 = self.right_conv_1(cat_1)
        return self.out(right_conv_1)
