import os
import time
import logging


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from utils.ssim import SSIM


class Loss(nn.Module):

    def __init__(self, losses):
        super().__init__()
        self.loss_funcs = []

        for loss in losses:
            if 'ssim' in loss:
                win_size = int(loss[loss.rfind('m') + 1:])
                self.ssim = SSIM(data_range=1.0,
                                 win_size=win_size,
                                 nonnegative_ssim=True)
                self.loss_funcs.append(self._ssim)
            elif loss=='mse':
                self.mse=nn.MSELoss()
                self.loss_funcs.append(self._mse)

    def _ssim(self,x,y):
        x,y=denormalize(x),denormalize(y)
        return 1-self.ssim(x,y)

    def _mse(self,x,y):
        return self.mse(x,y)

    def forward(self,x,y):
        loss=0
        for loss_func in self.loss_funcs:
            loss=loss+loss_func(x,y)

        return loss


def normalize(img):
    return (img-0.5)*2

def denormalize(img):
    return (img+1)/2

def shuffle_pixel(img,p=0.3):
    if p==0: return img.clone()

    *_,h,w=img.shape
    out=img.clone()
    original_index=torch.arange(h*w)
    shuffle_index=torch.randperm(h*w)
    shuffle_index=torch.where(torch.rand(*original_index.shape)>p,original_index,shuffle_index)

    out=out.view(3,-1)[:,shuffle_index].view(*out.shape)
    return out

def log(iter_cnt, iter_num, start_time, start_time_inloop, scale_cur, scale_num,loss):
    desc = f"[{scale_cur + 1}|{scale_num}][{iter_cnt}|{iter_num}]"

    time_log = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    time_log_inloop = time.strftime("%M:%S", time.gmtime(time.time() - start_time_inloop))
    logging.info(f"{time_log} | {time_log_inloop} | Iteration {desc} | Loss {loss}")
    start_time_inloop = time.time()
    
def log_imgs(save_dir,imgs,desc):
    log_img = torch.cat([denormalize(interp(img.detach(), imgs[-1].shape[-2:])) for img in imgs][::-1], dim=-2)

    os.makedirs(save_dir, exist_ok=True)
    path = f"{save_dir}/{desc}.png"

    save_image(log_img, path, padding=10)

def interp(x,img_shape):
    if isinstance(img_shape,(tuple,list)):
        return F.interpolate(x,size=img_shape,mode='bicubic',align_corners=True).clamp(-1,1)
    elif isinstance(img_shape,int):
        h,w=x.shape[-2:]
        self_scale=img_shape/max(h,w)
        h,w=int(h*self_scale),int(w*self_scale)
        return F.interpolate(x,size=(h,w),mode='bicubic',align_corners=True).clamp(-1,1)
    else:
        raise Exception