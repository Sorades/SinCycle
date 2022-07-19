import torch
import torch.nn.functional as F


def _fspecial_gauss_1d(size, sigma):
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)

def _ssim(X,Y,
          data_range,
          win,
          K=(0.01,0.03)):
    """计算X和Y的ssim index

    Args:
        X (torch.Tenor): images
        Y (torch.Tensor): images
        data_range (float or int): 像素值范围，一般 1.0 或 255
        win (torch.Tensor): 1-D gauss keenel
        K (tuple, optional): 常量 (K1,K2). 如果结果为负值或NaN则表明需要将K2的值上调. Defaults to (0.01,0.03).
    Return:
        torch.Tensor: ssim results.
    """
    K1,K2=K
    compensation=1.0

    C1=(K1*data_range)**2
    C2=(K2*data_range)**2

    win=win.to(X.device,dtype=X.dtype)

    mu1=gaussian_filter(X,win)
    mu2=gaussian_filter(Y,win)

    mu1_mu2=mu1*mu2
    mu1_sq,mu2_sq=mu1.pow(2),mu2.pow(2)

    sigma1_sq=compensation*(gaussian_filter(X*X,win)-mu1_sq)
    sigma2_sq=compensation*(gaussian_filter(Y*Y,win)-mu2_sq)
    sigma12=compensation*(gaussian_filter(X*Y,win)-mu1_mu2)

    cs_map=(2*sigma12+C2)/(sigma1_sq+sigma2_sq+C2)
    ssim_map=((2*mu1_mu2+C1)/(mu1_sq+mu2_sq+C1))*cs_map

    ssim_per_channel=torch.flatten(ssim_map,2).mean(-1)
    cs=torch.flatten(cs_map,2).mean(-1)

    return ssim_per_channel,cs



def ssim(X,
         Y,
         data_range=255,
         size_average=True,
         win_size=11,
         win_sigma=1.5,
         win=None,
         K=(0.01, 0.03),
         nonnegative_ssim=False):
    """interface of ssim

    Args:
        X (torch.Tensor): (N,C,H,W)
        Y (torch.Tensor): (N,C,H,W)
        data_range (float or int, optional): 像素值范围. Defaults to 255.
        size_average (bool, optional): 为真则所有图像的ssim会做均值，称为标量. Defaults to True.
        win_size (int, optional): gauss 核的大小. Defaults to 11.
        win_sigma (float, optional): 正态分布的 sigma. Defaults to 1.5.
        win (torch.Tensor, optional): 1-D gauss kernel, 如果为空，则会根据win_size核win_sigma生成一个核. Defaults to None.
        K (list or tuple, optional): 常量 (K1,K2). 如果结果为负值或NaN则表明需要将K2的值上调. Defaults to (0.01,0.03).
        nonnegative_ssim (bool, optional): 强制使ssim为非负. Defaults to False.
    Return:
        torch.Tensor: ssim results.
    """
    if len(X.shape) != 4:
        raise ValueError('Input images should be 4-d tensors.')
    if not X.type() == Y.type():
        raise ValueError('Input images should have the same dtype.')
    if not X.shape == Y.shape:
        raise ValueError('Input images should have the same shape.')

    if win is not None:
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError('Window size should be odd.')

    if win is None:
        win=_fspecial_gauss_1d(win_size,win_sigma)
        win=win.repeat(X.shape[1],1,1,1)

    ssim_per_channel,_=_ssim(X,Y,
                             data_range=data_range,
                             win=win,
                             K=K)

    if nonnegative_ssim:
        ssim_per_channel=torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)

def gaussian_filter(input,win):
    """Blur input with 1-D kernel

    Args:
        input (torch.Tensor): a batch of tensors to be blured.
        win (torch.Tensor): 1-D gauss kernel.
    Return:
        torch.Tensor: blured tensors.
    """
    C=input.shape[1]
    out=F.conv2d(input,win,stride=1,padding=0,groups=C)
    out=F.conv2d(out,win.transpose(2,3),stride=1,padding=0,groups=C)
    return out

