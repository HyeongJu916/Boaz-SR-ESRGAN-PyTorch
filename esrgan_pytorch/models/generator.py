# Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

model_urls = {
    "esrgan_4x4_16": "https://github.com/Lornatang/ESRGAN-PyTorch/releases/download/0.1.0/GAN_esrgan_4x4_16.pth",
    "esrgan_4x4_23": "https://github.com/Lornatang/ESRGAN-PyTorch/releases/download/0.1.0/GAN_esrgan_4x4_23.pth"
}


class Generator(nn.Module):
    def __init__(self, num_rrdb_blocks=16):
        r""" This is an esrgan model defined by the author himself.

        We use two settings for our generator – one of them contains 8 residual blocks, with a capacity similar
        to that of SRGAN and the other is a deeper model with 16/23 RRDB blocks.

        Args:
            num_rrdb_blocks (int): How many residual in residual blocks are combined. (Default: 16).

        Notes:
            Use `num_rrdb_blocks` is 16 for TITAN 2080Ti.
            Use `num_rrdb_blocks` is 23 for Tesla A100.
        """
        super(Generator, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # 16/23 ResidualInResidualDenseBlock layer
        rrdb_blocks = []
        for _ in range(num_rrdb_blocks):
            rrdb_blocks += [ResidualInResidualDenseBlock(in_channels=64, growth_channels=32, scale_ratio=0.2)]
        self.Trunk_RRDB = nn.Sequential(*rrdb_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Upsampling layers
        self.up1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.up2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Next layer after upper sampling
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # Final output layer
        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out1 = self.conv1(input)
        trunk = self.Trunk_RRDB(out1)
        out2 = self.conv2(trunk)
        out = torch.add(out1, out2)
        out = F.leaky_relu(self.up1(F.interpolate(out, scale_factor=2, mode="nearest")), 0.2, True)
        out = F.leaky_relu(self.up2(F.interpolate(out, scale_factor=2, mode="nearest")), 0.2, True)
        out = self.conv3(out)
        out = self.conv4(out)

        return out


class ResidualDenseBlock(nn.Module):
    r"""The residual block structure of traditional SRGAN and Dense model is defined"""

    def __init__(self, in_channels: int = 64, growth_channels: int = 32, scale_ratio: float = 0.2):
        """

        Args:
            in_channels (int): Number of channels in the input image. (Default: 64).
            growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32).
            scale_ratio (float): Residual channel scaling column. (Default: 0.2)
        """
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + 0 * growth_channels, growth_channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels + 1 * growth_channels, growth_channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels + 2 * growth_channels, growth_channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels + 3 * growth_channels, growth_channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv5 = nn.Conv2d(in_channels + 4 * growth_channels, in_channels, 3, 1, 1, bias=False)

        self.scale_ratio = scale_ratio

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(input)
        conv2 = self.conv2(torch.cat((input, conv1), 1))
        conv3 = self.conv3(torch.cat((input, conv1, conv2), 1))
        conv4 = self.conv4(torch.cat((input, conv1, conv2, conv3), 1))
        conv5 = self.conv5(torch.cat((input, conv1, conv2, conv3, conv4), 1))

        return conv5.mul(self.scale_ratio) + input


class ResidualInResidualDenseBlock(nn.Module):
    r"""The residual block structure of traditional ESRGAN and Dense model is defined"""

    def __init__(self, in_channels: int = 64, growth_channels: int = 32, scale_ratio: float = 0.2):
        """

        Args:
            in_channels (int): Number of channels in the input image.
            growth_channels (int): how many filters to add each layer (`k` in paper).
            scale_ratio (float): Residual channel scaling column.
        """
        super(ResidualInResidualDenseBlock, self).__init__()
        self.RDB1 = ResidualDenseBlock(in_channels, growth_channels, scale_ratio)
        self.RDB2 = ResidualDenseBlock(in_channels, growth_channels, scale_ratio)
        self.RDB3 = ResidualDenseBlock(in_channels, growth_channels, scale_ratio)

        self.scale_ratio = scale_ratio

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.RDB1(input)
        out = self.RDB2(out)
        out = self.RDB3(out)

        return out.mul(self.scale_ratio) + input


def _esrgan(arch, num_residual_block, pretrained, progress):
    model = Generator(num_residual_block)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def esrgan_4x4_16(pretrained: bool = False, progress: bool = True) -> Generator:
    r"""GAN model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1809.00219>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _esrgan("esrgan_4x4_16", 16, pretrained, progress)


def esrgan_4x4_23(pretrained: bool = False, progress: bool = True) -> Generator:
    r"""GAN model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1809.00219>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _esrgan("esrgan_4x4_23", 23, pretrained, progress)
