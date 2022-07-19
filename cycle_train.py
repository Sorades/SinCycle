import argparse
import logging
import os
import sys
from glob import glob
from loaders.loader import ImageHolder

from torchvision.utils import save_image
from runners.Cycle_runner import CycleRunner
from runners.SinIR_runner import Runner
from utils.parser import Parser
from utils.runner_utils import interp,denormalize

def arg_parse():
    """参数解析

    Returns:
        tuple: gpu & yaml config
    """
    parser=argparse.ArgumentParser()

    parser.add_argument('gpu',default='0',type=str) # gpu配置
    parser.add_argument('--yaml','-y',type=str,required=True)   # yaml文件路径
    p=parser.parse_args()

    return p.gpu,p.yaml

def set_logging(save_dir,gpu):
    """配置log

    Args:
        save_dir (str): log保存文件夹路径
        gpu (int): gpu index
    """
    os.makedirs(save_dir,exist_ok=True)

    log_format=f"%(asctime)s GPU {gpu}: {os.path.basename(save_dir)} | %(message)s"
    logging.basicConfig(stream=sys.stdout,level=logging.INFO,format=log_format,datefmt="[%y/%m%d %H:%M:%S]")

    file_handler=logging.FileHandler(f"{save_dir}/log.txt")
    file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(file_handler)

def make_save_dir(yaml, scale_num, scale_factor):
    """每次训练新建文件夹，返回路径

    Args:
        yaml (YamlStructure): parser.C
        scale_num (int): 下采样的数量
        scale_factor (float): 缩放因子

    Returns:
        str: 根据规则生成的路径名
    """
    save_dir = []
    raw=os.path.basename(yaml.INPUTS.raw['img_path']).split('.')[0]
    ref=os.path.basename(yaml.INPUTS.ref['img_path']).split('.')[0]
    save_dir.append(f"[{raw}_x_{ref}]")
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


def main():
    #region 参数导入
    gpu,yaml=arg_parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    # 指定yaml文件路径
    yaml_path = f"{os.getcwd()}/configs/{yaml}.yaml"

    parser = Parser(yaml_path)
    args_yaml = parser.C

    resampler=interp # 指定resampler方法

    # 加载raw和ref图像
    raw_holder=ImageHolder(**args_yaml.INPUTS.raw,resampler=resampler)
    ref_holder=ImageHolder(**args_yaml.INPUTS.ref,resampler=resampler)

    # 生成输出文件夹
    save_dir,_=make_save_dir(args_yaml,raw_holder.scale_num,raw_holder.scale_factor)
    # save_dir="/home/sora/code/Python/underwater/SinCycle/outs/001_[blur_fishxyellow_fish]_S2_CH128_SF1.000_[ssim11_mse]_[500|0.005]_[test_modify]"
    # save_dir="/home/sora/code/Python/underwater/SinCycle/outs/004_[blur_fish_x_yellow_fish]_S2_CH128_SF1.000_[ssim11_mse]_[200|0.005]_[test_modify]"

    # 输出log信息
    set_logging(save_dir,gpu)

    # 保存图像与参数
    save_image(denormalize(raw_holder.img),f"{save_dir}/raw-image.png")
    save_image(denormalize(ref_holder.img),f"{save_dir}/ref-image.png")

    parser.dump(f"{save_dir}/args.yaml")

    logging.info(args_yaml)
    logging.info(save_dir)
    #endregion

    # 配置runner，进行训练
    runner=CycleRunner(raw_holder=raw_holder,
                       ref_holder=ref_holder,
                       save_dir=save_dir,
                       args=args_yaml,
                       resampler=resampler)

    runner.train()

def test():
        #region 参数导入
    gpu,yaml=arg_parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    # 指定yaml文件路径
    yaml_path = f"{os.getcwd()}/configs/{yaml}.yaml"

    parser = Parser(yaml_path)
    args_yaml = parser.C

    resampler=interp # 指定resampler方法

    # 加载raw和ref图像
    raw_holder=ImageHolder(**args_yaml.INPUTS.raw,resampler=resampler)
    ref_holder=ImageHolder(**args_yaml.INPUTS.ref,resampler=resampler)

    # 生成输出文件夹
    save_dir,cnt=make_save_dir(args_yaml,raw_holder.scale_num,raw_holder.scale_factor)
    # save_dir="/home/sora/code/Python/underwater/SinCycle/outs/001_[blur_fishxyellow_fish]_S2_CH128_SF1.000_[ssim11_mse]_[500|0.005]_[test_modify]"

    # 输出log信息
    set_logging(save_dir,gpu)

    # 保存图像与参数
    save_image(denormalize(raw_holder.img),f"{save_dir}/raw-image.png")
    save_image(denormalize(ref_holder.img),f"{save_dir}/ref-image.png")

    parser.dump(f"{save_dir}/args.yaml")

    logging.info(args_yaml)
    logging.info(save_dir)
    #endregion

    runner=CycleRunner(raw_holder=raw_holder,
                       ref_holder=ref_holder,
                       save_dir=save_dir,
                       args=args_yaml,
                       resampler=resampler)

    runner.test()

if __name__=="__main__":
    main()