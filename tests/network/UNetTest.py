import unittest
from unet_cityscapes.network.UNet import UNet
import torch


class UNetTest(unittest.TestCase):
    def test_forward_low_res(self):
        unet = UNet(3, 34)
        input = torch.rand((1, 3, 512, 1024))
        output = unet.forward(input)
        self.assertEqual(output.shape, (1, 34, 512, 1024))

    def test_forward_high_res(self):
        unet = UNet(3, 34)
        input = torch.rand((1, 3, 1024, 2048))
        output = unet.forward(input)
        print(output.shape)
        self.assertEqual(output.shape, (1, 34, 1024, 2048))
