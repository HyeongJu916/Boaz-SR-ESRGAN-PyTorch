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
import argparse
import os

import cv2
import numpy as np
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, ToPILImage, transforms

import PIL
from PIL import Image
from tqdm import tqdm

from esrgan_pytorch import Generator
from esrgan_pytorch import select_device
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="ESRGAN algorithm is applied to video files.")
parser.add_argument("--file", type=str, required=True,
                    help="Test low resolution video name.")
parser.add_argument("--upscale-factor", type=int, default=4, choices=[4],
                    help="Low to high resolution scaling factor. (default:4).")
parser.add_argument("--model-path", default="./weight/ESRGAN_4x.pth", type=str, metavar="PATH",
                    help="Path to latest checkpoint for model. (default: ``./weight/ESRGAN_4x.pth``).")
parser.add_argument("--device", default="0",
                    help="device id i.e. `0` or `0,1` or `cpu`. (default: ``CUDA:0``).")

########################################################
####        Test Code for h:w = 9:16 Videos         ####
########################################################
####  Recommended Input Resolution - h:w = 180:320  ####
########################################################

args = parser.parse_args()
print(args)

# Selection of appropriate treatment equipment
device = select_device(args.device, batch_size=4)

# Construct SRGAN model.
model = Generator(upscale_factor=args.upscale_factor).to(device)
model.load_state_dict(torch.load(args.weights, map_location=device))

def expandFrame(frame):
    height = frame.shape[0]
    if height < 1080:
        h = 1080
        w = 1920
        frame = cv2.resize(frame, (w, h), interpolation = Image.BILINEAR)
    return frame

def text_on_frame(frame):
    frame = cv2.putText(frame,  
                f'ESRGAN x4 Upscale',  
                org = (25, 100),
                fontFace = cv2.FONT_HERSHEY_DUPLEX,  
                fontScale = 1.5, color = (0, 255, 255), thickness = 2)
    frame = cv2.putText(frame,  
                f'{input_h}x{input_w} -> {4*input_h}x{4*input_w}',  
                org = (25, 150),
                fontFace = cv2.FONT_HERSHEY_DUPLEX,  
                fontScale = 1.5, color = (0, 255, 255), thickness = 2)
    return frame

def make_bar(frame, bar_size):
    frame = frame[ bar_size:-(bar_size), : , : ]
    frame = cv2.copyMakeBorder(frame,
                                top= bar_size, bottom=bar_size,
                                left=0, right = 0,
                                borderType=cv2.BORDER_CONSTANT,
                                value=[0, 0, 0] # black
                                )
    return frame



# Set model eval mode
model.eval()


UPSCALE_FACTOR = args.upscale_factor
VIDEO_NAME = args.file
input_w = 320
input_h = 180
sr_video_size = (input_w * UPSCALE_FACTOR, input_h * UPSCALE_FACTOR)
save_video_size = sr_video_size
if (input_h * UPSCALE_FACTOR) < 1080 :
    save_video_size = (1920, 1080)

#model = Generator(UPSCALE_FACTOR).eval()
if torch.cuda.is_available():
    model = model.cuda()

videoCapture = cv2.VideoCapture(VIDEO_NAME)
fps = videoCapture.get(cv2.CAP_PROP_FPS)
frame_numbers = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)

output_sr_name  = VIDEO_NAME.split('.')[0] + '_out_srf_' + f"{input_h}pX{str(UPSCALE_FACTOR)}" + '_concat' + '.mp4'
sr_video_writer = cv2.VideoWriter(output_sr_name, cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'), fps, save_video_size)

success, frame = videoCapture.read()
#test_bar = tqdm(range(int(frame_numbers)), desc='[processing video and saving result videos]')
test_bar = tqdm(range(int(fps*58)), desc='[processing video and saving result videos]')
for index in test_bar:
      if success:
        
        frame = PIL.Image.fromarray(frame)
        frame = frame.resize(sr_video_size,      Image.BILINEAR)
        frame = frame.resize((input_w, input_h), Image.BILINEAR)
        frame = np.array(frame)

        # Upscale(resize) the LR_img before cropping
        lr_img = cv2.resize(frame, sr_video_size,
                            interpolation = cv2.INTER_NEAREST)
        # Cropping half of the LR_img
        h, w   = lr_img.shape[:2]
        pad_size = 2
        lr_img = lr_img[ :, int(w*0.25):int(w*0.75)-pad_size, : ] # 좌측 반
        lr_img = cv2.copyMakeBorder(lr_img,
                                    top=0, bottom=0,
                                    left=0, right = pad_size,
                                    borderType=cv2.BORDER_CONSTANT,
                                    value=[255, 255, 255] # 흰색
                                    )
                
        image = transforms.ToTensor()(frame).unsqueeze(0)        
        
        if torch.cuda.is_available():
            image = image.cuda()
        
        out = model(image)
        out = out.cpu()
        out = out.clamp(0,1)
        out_img = out.data[0].numpy()
        out_img *= 255.0
        out_img = (np.uint8(out_img)).transpose((1, 2, 0))

        # hr_img cropping
        out_img = out_img[ : , int(w*0.25):int(w*0.75) , : ]        
        # Concatenate lr & hr imgs
        concat_img = cv2.hconcat([lr_img, out_img])
        # Expand by frame height by 720p, if smaller 
        concat_img = expandFrame(concat_img)
        # Make bar on top & bottom      
        bar = int(save_video_size[1] * 0.2)
        concat_img = make_bar(concat_img, bar)

        # Write Text on Frame
        concat_img = text_on_frame(concat_img) 

  

        sr_video_writer.write(concat_img) 
        success, frame = videoCapture.read()

videoCapture.release()
sr_video_writer.release()
print(f"Video Saved : {output_sr_name}")
