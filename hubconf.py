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

"""File for accessing GAN via PyTorch Hub https://pytorch.org/hub/
Usage:
    import torch
    model = torch.hub.load("Lornatang/ESRGAN-PyTorch", num_residual_block=16)
"""
import torch
from torch.hub import load_state_dict_from_url

from esrgan_pytorch import Generator

model_urls = {
    "esrgan_4x4_16": "https://github.com/Lornatang/ESRGAN-PyTorch/releases/download/0.1.0/GAN_4x4_esrgan_16.pth",
    "esrgan_4x4_23": "https://github.com/Lornatang/ESRGAN-PyTorch/releases/download/0.1.0/GAN_4x4_esrgan_23.pth",
}

dependencies = ["torch"]


def create(arch, num_rrdb_blocks, pretrained, progress):
    """ Creates a specified GAN model

    Args:
        arch (str): Arch name of model.
        num_rrdb_blocks (int): How many layers of RRDB stack.
        pretrained (bool): Load pretrained weights into the model.
        progress (bool): Show progress bar when downloading weights.

    Returns:
        PyTorch model.
    """
    model = Generator(num_rrdb_blocks)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
    return model


def esrgan_4x4_16(pretrained: bool = False, progress: bool = True) -> Generator:
    r"""GAN model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1809.00219>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return create("esrgan_4x4_16", 16, pretrained, progress)


def esrgan_4x4_23(pretrained: bool = False, progress: bool = True) -> Generator:
    r"""GAN model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1809.00219>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return create("esrgan_4x4_23", 23, pretrained, progress)
