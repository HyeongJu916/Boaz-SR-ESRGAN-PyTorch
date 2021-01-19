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
import os

import torch.utils.data.dataset
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomApply, ColorJitter
import torch
from PIL import Image

__all__ = [
    "TrainDatasetFromFolder", "check_image_file"
]

# customized + dynamic size
def train_hr_transform(crop_size):
    return Compose([

        RandomApply(torch.nn.ModuleList([
            Resize(960),
            Resize(720),
            Resize(576),
            Resize(480)		
						]), p=0.7),

        RandomCrop(crop_size),
				RandomHorizontalFlip(p=0.5),				
				#RandomVerticalFlip(p=0.5),

        RandomApply(torch.nn.ModuleList([
            ColorJitter(brightness=0.1, 
                        contrast=0.1, 
                        saturation=0.1, 
                        hue=0.05),
						]), p=0.3),

        ToTensor(),
    ])

# only downsizing
def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])

class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


def check_image_file(filename):
    r"""Filter non image files in directory.

    Args:
        filename (str): File name under path.

    Returns:
        Return True if bool(x) is True for any x in the iterable.

    """
    return any(filename.endswith(extension) for extension in ["bmp", ".png",
                                                              ".jpg", ".jpeg",
                                                              ".png", ".PNG",
                                                              ".jpeg", ".JPEG"])
