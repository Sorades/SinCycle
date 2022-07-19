import torch.nn as nn
from utils.functions import _fspecial_gauss_1d, ssim


class SSIM(nn.Module):

    def __init__(self,
                 data_range=255,
                 size_average=True,
                 win_size=11,
                 win_sigma=1.5,
                 channel=3,
                 K=(0.01, 0.03),
                 nonnegative_ssim=False):
        super(SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size,
                                      win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X, Y):
        return ssim(X,Y,
                    data_range=self.data_range,
                    size_average=self.size_average,
                    win=self.win,
                    K=self.K,
                    nonnegative_ssim=self.nonnegative_ssim)

