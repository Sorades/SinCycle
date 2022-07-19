from copy import deepcopy
import logging
import os
from time import time

import torch
from torchvision.utils import save_image
from utils.runner_utils import Loss, shuffle_pixel, log, log_imgs, denormalize
from networks.network import Network
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


class Runner:

    def __init__(self, img_holder, save_dir, losses, iter_per_scale,opt_iter,
                 pixel_shuffle_p, img_ch, net_ch, lr, betas, scale_num,
                 resampler, model_name):
        self.img = img_holder.imgs # multiscale 的图像list
        self.save_dir = save_dir # train directory

        self.loss = Loss(losses)
        self.scale_num = scale_num

        self.iter_per_scale = iter_per_scale
        self.pixel_shuffle_p = pixel_shuffle_p
        self.opt_iter=opt_iter

        self.net_list = [Network(img_ch, net_ch).cuda()] # 串联网络

        self.lr = lr    # 学习率
        self.betas = betas
        self.opt = Adam(self.net_list[-1].parameters(), lr, betas)
        self.opt_sch = CosineAnnealingLR(self.opt, self.iter_per_scale)

        self.resampler = resampler
        self.model_name = model_name

    def train(self):
        """SinIR模型训练
        """
        start_time = time()
        start_time_inloop = time()

        for scale in range(self.scale_num):
            train_loss = 0
            for iter_cnt in range(1, self.iter_per_scale + 1):
                out = self._forward(img=self.img, scale_to=scale, scale_from=0)
                loss = self._calc_loss(out, scale)
                self._step(loss)

                train_loss = (train_loss *
                              (iter_cnt - 1) + float(loss)) / iter_cnt

                if iter_cnt % (self.iter_per_scale // 4) == 0:
                    log(iter_cnt, self.iter_per_scale, start_time,
                        start_time_inloop, scale, self.scale_num, train_loss)
                    start_time_inloop = time()

            if scale < self.scale_num - 1:
                self._grow_network()

        self.save()

    def optimize(self, opt_img,desc=None):
        """SinCycle方法优化

        Args:
            opt_img (list): 转换状态的multiscale 图像
            desc (str, optional): 图像保存时的描述. Defaults to None.
        """
        self.net_list[0].requires_grad_(True)
        for scale in range(self.scale_num):
            opt_loss = 0

            self.opt = Adam(self.net_list[scale].parameters(), self.lr,
                            self.betas)
            self.opt_sch = CosineAnnealingLR(self.opt, self.opt_iter)

            for iter_cnt in range(1, self.opt_iter + 1):
                out_opt = self._forward(img=opt_img,
                                        scale_to=scale,
                                        scale_from=0)
                out_rec=self._forward(img=self.img,scale_from=0,scale_to=scale)

                loss = self._calc_loss(out_opt, scale)+self._calc_loss(out_rec,scale)
                self._step(loss)

                opt_loss = (opt_loss * (iter_cnt - 1) + float(loss)) / iter_cnt

            logging.info(f"Model {self.model_name} training | Scale {scale}/{self.scale_num} | Loss {opt_loss}")
            desc=f"{desc}_IT[{scale}|{self.scale_num}]"
            log_imgs(save_dir=f"{self.save_dir}/opt_results/",imgs=out_opt,desc=desc)

        self.save(path=f"{self.save_dir}/opt_model/")

    def infer(self, infer_img, desc):
        """推断

        Args:
            infer_img (list): multiscale image
            desc (str): 图像保存时的描述

        Returns:
            Image: 结果图像
        """
        with torch.no_grad():
            logging_imgs = []
            for i_from in range(self.scale_num):
                out = self._forward(img=infer_img,
                                    scale_to=self.scale_num - 1,
                                    scale_from=i_from,
                                    infer=True)

                logging_imgs.append(out[-1])

            grid = denormalize(torch.cat(logging_imgs, dim=0))
            os.makedirs(f"{self.save_dir}/infer_result/",exist_ok=True)
            save_image(grid, f"{self.save_dir}/infer_result/{desc}.png", nrow=3)

        return grid[-1]

    def save(self,path=None):
        """保存模型
        """
        if path:
            os.makedirs(path)
            torch.save({'net': [net.state_dict() for net in self.net_list]},f"{path}/{self.model_name}.torch")
            logging.info(f"Saved to {path}/{self.model_name}.torch")
        torch.save({'net': [net.state_dict() for net in self.net_list]},
                   f"{self.save_dir}/{self.model_name}.torch")
        logging.info(f"Saved to {self.save_dir}/{self.model_name}.torch")

    def load(self):
        """加载模型
        """
        saved = torch.load(f"{self.save_dir}/{self.model_name}.torch")
        for _ in range(self.scale_num - 1):
            self._grow_network()
        for net, sn in zip(self.net_list, saved['net']):
            net.load_state_dict(sn)
        logging.info(f"Loaded from {self.save_dir}/{self.model_name}.torch")
        self.net_list[0].requires_grad_(True)

    def _forward(self, img, scale_to, scale_from=0, infer=False):
        """串联网络的前向传播

        Args:
            img (list): multiscale images
            scale_to (int): 初始scale
            scale_from (int, optional): 目标scale. Defaults to 0.
            infer (bool, optional): 是否推断中. Defaults to False.

        Returns:
            torch.Tensor: 激活值
        """
        x = img[scale_from]
        out = []

        for scale in range(scale_from, scale_to + 1):
            net = self.net_list[scale]

            x = net(x)
            out.append(x)

            if not infer:
                with torch.no_grad():
                    x = shuffle_pixel(x, p=self.pixel_shuffle_p)

            if scale < scale_to:
                shape = img[scale + 1].shape[-2:]
                if x.shape[-2:] == shape:
                    x = x.detach()
                else:
                    with torch.no_grad():
                        x = self.resampler(x, shape)

        return out

    def _calc_loss(self, out, scale):
        """计算损失

        Args:
            out (torch.Tensor): 激活值
            scale (int): 在scale层计算损失

        Returns:
            torch.Tensor: 损失值
        """
        return self.loss(out[scale], self.img[scale])

    def _step(self, loss):
        """反向传播优化

        Args:
            loss (torch.Tensor): 损失值
        """
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.opt_sch.step()

    def _grow_network(self):
        """增加串联的网络个数
        """
        self.net_list.append(deepcopy(self.net_list[-1]))
        self.net_list[-2].requires_grad_(False)
        self.opt = Adam(self.net_list[-1].parameters(), self.lr, self.betas)
        self.opt_sch = CosineAnnealingLR(self.opt, self.iter_per_scale)
