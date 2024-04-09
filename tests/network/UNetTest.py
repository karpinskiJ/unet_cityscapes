import unittest
from unet_cityscapes.network.UNetLowRes import UNetLowRes
from unet_cityscapes.network.UNetHighRes import UNetHighRes
import torch


class UNetTest(unittest.TestCase):
    def test_forward_low_res(self):
        unet = UNetLowRes(3, 34)
        input = torch.rand((1, 3, 508, 1020))
        output = unet.forward(input)
        self.assertEqual(output.shape, (1, 34, 324, 836))

    def test_forward_high_res(self):
        unet = UNetHighRes(3, 34)
        input = torch.rand((1, 3, 1024, 2048))
        output = unet.forward(input)
        print(output.shape)
        self.assertEqual(output.shape, (1, 34, 836, 1860))
