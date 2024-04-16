import torch.nn as nn


class DoubleConv2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv_3x3_with_activation = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3,padding="same"),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3,padding="same"),
            nn.ReLU()
        )

    def forward(self, input):
        return self.double_conv_3x3_with_activation(input)


class OutLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.output = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            # nn.Softmax(dim=1)
        )

    def forward(self, input):
        return self.output(input)
