from copy import deepcopy
from PIL import Image
from torchvision.transforms import ToTensor

from utils.runner_utils import normalize


class ImageHolder:

    def __init__(self,
                 img_path,
                 resampler,
                 img_shape=None,
                 scale_factor=1,
                 scale_num=2) -> None:

        self.resampler = resampler

        self.img = self.load(img_path, img_shape).cuda()

        self.scale_factor = scale_factor
        self.scale_num = scale_num

        self.imgs = self._multiscale(self.img)

        self.img_path=img_path

    def load(self, img_path, img_shape):
        img = Image.open(img_path).convert('RGB')
        img = normalize(ToTensor()(img))

        if img_shape is not None:
            return self.resampler(img.unsqueeze(0), img_shape)
        else:
            return img.unsqueeze(0)

    def copy_config(self,img):
        holder=deepcopy(self)
        holder.img=img.unsqueeze(0)
        holder.imgs=holder._multiscale(holder.img)

        return holder

    def _multiscale(self, img):
        self_scale = 1
        imgs = [img]
        *_, h, w = img.shape
        for _ in range(self.scale_num - 1):
            self_scale = self_scale * self.scale_factor
            img = self.resampler(
                img, (round(h * self_scale), round(w * self_scale)))
            imgs.append(img)

        return imgs[::-1]