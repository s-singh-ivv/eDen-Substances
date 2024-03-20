__version__ = "2.0"
__author__ = "Singh, Satnam"
__maintainer__ = "Singh, Satnam"
__contact__ = "satnam.singh@ivv.fraunhofer.de"

import torch
from torch.nn import Module
from torch.nn import Sequential
from torch.nn import (
    Conv3d,
    Dropout3d,
    MaxPool3d,
    ReLU,
    BatchNorm3d,
    ConvTranspose3d,
    LeakyReLU,
)
from torchinfo import summary
from torch.utils import checkpoint
import torch.nn.functional as F
import torch.nn as nn


i = 0


class U_net_3d(Module):
    """
    3D-UNet implementation
    Parameters:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        features (int): number of features/channels to be used
        padding (int): padding for each conv layer
        debug (bool): whether to print shapes of each intermediate layer for debug purposes
        checkpointing (bool): whether to enable checkpoint sequential implementation for memory and runtime
                            efficiency
        filter_size (int): size of adaptive filter used prior to FC layers. Also dictates FC shape
        fin_neurons (int): number of neurons in final layer
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=8,
        padding=1,
        debug=False,
        checkpointing=True,
        filter_size=8,
        fin_neurons=64,
    ):
        super(U_net_3d, self).__init__()
        self.debug = debug
        self.filter_size = filter_size
        self.fin_neurons = fin_neurons
        self.checkpointing = checkpointing
        self.block1 = Sequential(
            Conv3d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=padding,
                bias=False,
            ),
            BatchNorm3d(num_features=features),
            LeakyReLU(inplace=True),
            Conv3d(
                in_channels=features,
                out_channels=2 * features,
                kernel_size=3,
                padding=padding,
                bias=False,
            ),
            BatchNorm3d(num_features=2 * features),
            LeakyReLU(inplace=True),
        )

        self.pool1 = MaxPool3d(kernel_size=2, stride=2, padding=0, ceil_mode=False)

        self.block2 = Sequential(
            Conv3d(
                in_channels=2 * features,
                out_channels=2 * features,
                kernel_size=3,
                padding=padding,
                bias=False,
            ),
            BatchNorm3d(num_features=2 * features),
            LeakyReLU(inplace=True),
            Conv3d(
                in_channels=2 * features,
                out_channels=4 * features,
                kernel_size=3,
                padding=padding,
                bias=False,
            ),
            BatchNorm3d(num_features=4 * features),
            LeakyReLU(inplace=True),
            Dropout3d(0.2, inplace=True),
        )

        self.pool2 = MaxPool3d(kernel_size=2, stride=2, padding=0, ceil_mode=False)

        self.block3 = Sequential(
            Conv3d(
                in_channels=4 * features,
                out_channels=4 * features,
                kernel_size=3,
                padding=padding,
                bias=False,
            ),
            BatchNorm3d(num_features=4 * features),
            LeakyReLU(inplace=True),
            Conv3d(
                in_channels=4 * features,
                out_channels=8 * features,
                kernel_size=3,
                padding=padding,
                bias=False,
            ),
            BatchNorm3d(num_features=8 * features),
            LeakyReLU(inplace=True),
            Dropout3d(0.2, inplace=True),
        )

        self.pool3 = MaxPool3d(kernel_size=2, stride=2, padding=0, ceil_mode=False)

        self.block4 = Sequential(
            Conv3d(
                in_channels=8 * features,
                out_channels=8 * features,
                kernel_size=3,
                padding=padding,
                bias=False,
            ),
            BatchNorm3d(num_features=8 * features),
            ReLU(inplace=True),
            Conv3d(
                in_channels=8 * features,
                out_channels=16 * features,
                kernel_size=3,
                padding=padding,
                bias=False,
            ),
            BatchNorm3d(num_features=16 * features),
            ReLU(inplace=True),
            Dropout3d(0.2, inplace=True),
        )

        self.upconv3 = ConvTranspose3d(
            in_channels=16 * features,
            out_channels=16 * features,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )

        self.de_block3 = Sequential(
            Conv3d(
                in_channels=(8 * features) * 3,
                out_channels=8 * features,
                kernel_size=3,
                padding=padding,
                bias=False,
            ),
            BatchNorm3d(num_features=8 * features),
            LeakyReLU(0.1, inplace=True),
            Conv3d(
                in_channels=8 * features,
                out_channels=8 * features,
                kernel_size=3,
                padding=padding,
                bias=False,
            ),
            BatchNorm3d(num_features=8 * features),
            LeakyReLU(0.1, inplace=True),
        )
        self.upconv2 = ConvTranspose3d(
            in_channels=8 * features,
            out_channels=8 * features,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )

        self.de_block2 = Sequential(
            Conv3d(
                in_channels=(4 * features) * 3,
                out_channels=4 * features,
                kernel_size=3,
                padding=padding,
                bias=False,
            ),
            BatchNorm3d(num_features=4 * features),
            LeakyReLU(0.1, inplace=True),
            Conv3d(
                in_channels=4 * features,
                out_channels=4 * features,
                kernel_size=3,
                padding=padding,
                bias=False,
            ),
            BatchNorm3d(num_features=4 * features),
            LeakyReLU(0.1, inplace=True),
        )

        self.upconv1 = ConvTranspose3d(
            in_channels=4 * features,
            out_channels=4 * features,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )
        self.de_block1 = Sequential(
            Conv3d(
                in_channels=(2 * features) * 3,
                out_channels=2 * features,
                kernel_size=3,
                padding=padding,
                bias=False,
            ),
            BatchNorm3d(num_features=2 * features),
            LeakyReLU(inplace=True),
            # 64->64
            Conv3d(
                in_channels=2 * features,
                out_channels=2 * features,
                kernel_size=3,
                padding=padding,
                bias=False,
            ),
            BatchNorm3d(num_features=2 * features),
            LeakyReLU(0.1, inplace=True),
        )
        self.conv1 = Conv3d(
            in_channels=2 * features,
            out_channels=3,
            kernel_size=1,
            padding=0,
            bias=True,
        )
        self.conv = Conv3d(
            in_channels=2 * features,
            out_channels=3,
            kernel_size=1,
            padding=0,
            bias=True,
        )
        self.Madap3d = nn.AdaptiveMaxPool3d(
            output_size=(self.filter_size, self.filter_size, self.filter_size)
        )
        self.fc1BN = nn.BatchNorm3d(num_features=3)
        self.fc1 = nn.Linear(
            self.filter_size * self.filter_size * self.filter_size * 3,
            self.fin_neurons,
            bias=True,
        )
        self.fc2 = nn.Linear(self.fin_neurons, 2, bias=True)

    def forward(self, x):
        """Forward function for the network class
        Debug mode allows printing of shapes for different layers"""

        if self.checkpointing:
            enc1 = checkpoint.checkpoint_sequential(self.block1, 6, x)
            pool1 = self.pool1(enc1)
            enc2 = checkpoint.checkpoint_sequential(self.block2, 6, pool1)
            pool2 = self.pool2(enc2)
            enc3 = checkpoint.checkpoint_sequential(self.block3, 6, pool2)
            pool3 = self.pool3(enc3)
            enc4 = checkpoint.checkpoint_sequential(self.block4, 6, pool3)

        else:
            enc1 = self.block1(x)
            pool1 = self.pool1(enc1)
            enc2 = self.block2(self.pool1(enc1))
            pool2 = self.pool2(enc2)
            enc3 = self.block3(self.pool2(enc2))
            pool3 = self.pool3(enc3)
            enc4 = self.block4(self.pool3(enc3))

        if self.debug:
            print(f"Input x: {x.shape}")
            print("#### Encoder path ####")
            print(f"enc1: {enc1.shape}")
            print(f"enc2: {enc2.shape}")
            print(f"enc3: {enc3.shape}")
            print(f"enc4: {enc4.shape}")

        dec3 = self.upconv3(enc4)
        dec3_1 = dec3
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3_2 = dec3
        dec3 = self.de_block3(dec3)
        dec3_3 = dec3

        dec2 = self.upconv2(dec3)
        dec2_1 = dec2
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2_2 = dec2
        dec2 = self.de_block2(dec2)
        dec2_3 = dec2

        dec1 = self.upconv1(dec2)
        dec1_1 = dec1
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1_2 = dec1
        dec1 = self.de_block1(dec1)
        dec1_3 = dec1

        fin = self.conv(dec1)
        three_dim_x = x.repeat(1, 3, 1, 1, 1)
        prod = torch.mul(three_dim_x, fin[:, :, :, :, :])
        adap = self.Madap3d(prod)
        batch_size = adap.shape[0]
        mid = self.fc1BN(adap)
        fin2 = F.relu(self.fc1(mid.view(batch_size, -1)))
        fin2 = self.fc2(fin2)

        if self.debug:
            print("#### Decoder path ####")
            print(f"dec3_1: {dec3_1.shape}")
            print(f"dec3_2: {dec3_2.shape}")
            print(f"dec3_3: {dec3_3.shape}")
            print(f"dec2_1: {dec2_1.shape}")
            print(f"dec2_2: {dec2_2.shape}")
            print(f"dec2_3: {dec2_3.shape}")
            print(f"dec1_1: {dec1_1.shape}")
            print(f"dec1_2: {dec1_2.shape}")
            print(f"dec1_3: {dec1_3.shape}")
            print(f"Final Shape: {fin.shape}")
            print(f"Final Shape: {fin2.shape}")

        return fin, fin2, enc4  # dice/eneg ; class


activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


if __name__ == "__main__":
    device = torch.device("cuda:0")
    in_channels = 1
    u_net_3dmodel = U_net_3d(
        in_channels=1,
        features=28,
        filter_size=4,
        fin_neurons=16,
        out_channels=3,
        checkpointing=True,
        debug=True,
    ).to(device)
    print(u_net_3dmodel)
    batch_size = 2
    print(summary(u_net_3dmodel, depth=2))
