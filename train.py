import logging
import os
import argparse
import sys
from glob import glob

from torchvision.utils import save_image
from infer import infer_main

from runners.SinIR_runner import Runner
from loaders.loader import ImageHolder
from utils.parser import Parser
from utils.runner_utils import denormalize, interp

def arg_parse():
    parser=argparse.ArgumentParser()

    parser.add_argument('gpu',default='0',type=str)
    parser.add_argument('--yaml','-y',type=str,required=True)
    p=parser.parse_args()

    return p.gpu,p.yaml

def set_logging(save_dir,gpu):
    os.makedirs(save_dir)

    log_format=f"%(asctime)s GPU {gpu}: {os.path.basename(save_dir)} | %(message)s"
    logging.basicConfig(stream=sys.stdout,level=logging.INFO,format=log_format,datefmt="[%y/%m%d %H:%M:%S]")

    file_handler=logging.FileHandler(f"{save_dir}/log.txt")
    file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(file_handler)

def make_save_dir(yaml, img_shape, scale_num, scale_factor):
    save_dir = []
    save_dir.append(f"[{os.path.basename(yaml.DATASET.img_path).split('.')[0]}]")
    save_dir.append("{}x{}".format(*img_shape))
    save_dir.append(f"S{scale_num}")
    save_dir.append(f"CH{yaml.NET.net_ch}")
    save_dir.append(f"SF{scale_factor:.3f}".replace("0.", ""))

    for i, loss in enumerate(yaml.TRAIN.losses):
        if i == 0:
            loss = '[' + loss
        if i == len(yaml.TRAIN.losses) - 1:
            loss = loss + ']'
        save_dir.append(loss)

    save_dir.append(f"[{str(yaml.TRAIN.iter_per_scale)}|{str(yaml.TRAIN.pixel_shuffle_p)}]")

    folders = glob(f"{os.getcwd()}/outs/*]")
    cnt = len(glob(f"{os.getcwd()}/outs/*]"))
    for f in folders:
        try:
            cnt = max(cnt, int(os.path.basename(f)[:3]))
        except Exception:
            continue
    cnt += 1
    return f"{os.getcwd()}/outs/{cnt:03d}_{'_'.join(save_dir)}_[{'_'.join(yaml.DESC)}]", cnt

def update_scale_params(parser, scale_factor, scale_num):
    parser.update_mem("DATASET", "scale_factor", value=scale_factor)
    parser.update_mem("DATASET", "scale_num", value=scale_num)

def train_main():
    gpu,yaml=arg_parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    yaml_path = f"{os.getcwd()}/configs/train/{yaml}.yaml"

    parser = Parser(yaml_path)
    args_yaml = parser.C

    resampler=interp

    img_holder = ImageHolder(**args_yaml.DATASET, resampler=resampler)

    update_scale_params(parser, img_holder.scale_factor, img_holder.scale_num)

    save_dir, cnt = make_save_dir(args_yaml,
                                  img_holder.img.shape[-2:],
                                  img_holder.scale_num,
                                  img_holder.scale_factor)

    set_logging(save_dir, gpu)

    save_image(denormalize(img_holder.img), f"{save_dir}/{os.path.basename(args_yaml.DATASET.img_path)}".replace('.jpg', '.png'))

    parser.dump(f"{save_dir}/args.yaml")
    logging.info(args_yaml)
    logging.info(save_dir)

    runner = Runner(img_holder,
                    save_dir,
                    **args_yaml.TRAIN,
                    **args_yaml.NET,
                    **args_yaml.OPT,
                    scale_num=img_holder.scale_num,
                    resampler=resampler)
    runner.train()

    if args_yaml.INFER_YAML is not None:
        for iy in args_yaml.INFER_YAML:
            infer_main(infer_dir_num=cnt,gpu=gpu,rerun=False,yaml=iy)

if __name__=="__main__":
    train_main()