
from utils.runner_utils import interp
from loaders.loader import ImageHolder
from runners.SinIR_runner import Runner


class CycleRunner:
    def __init__(self,raw_holder,ref_holder,save_dir,args,resampler) -> None:
        self.raw_holder=raw_holder
        self.ref_holder=ref_holder

        self.epoch=args.OPTIMIZE.epoch

        self.raw_runner=Runner(raw_holder,save_dir,
                               **args.TRAIN,
                               **args.NET,
                               **args.OPT,
                               opt_iter=args.OPTIMIZE.opt_iter,
                               scale_num=raw_holder.scale_num,
                               resampler=resampler,
                               model_name='raw')
        self.ref_runner=Runner(ref_holder,save_dir,
                               **args.TRAIN,
                               **args.NET,
                               **args.OPT,
                               opt_iter=args.OPTIMIZE.opt_iter,
                               scale_num=ref_holder.scale_num,
                               resampler=resampler,
                               model_name='ref')

    def train(self):
        # 得到两个风格模型
        # self.raw_runner.train()
        # self.ref_runner.train()
        # 加载已有模型
        self.raw_runner.load()
        self.ref_runner.load()

        # 原始模型推断
        self.ref_runner.infer(infer_img=self.raw_holder.imgs,desc="result_raw")

        for i in range(self.epoch):
            # 分别进行迁移，得到水下状态的正常光照图像ref2raw和正常光照状态下的水下图像raw2ref
            ref2raw=self.raw_runner.infer(infer_img=self.ref_holder.imgs,desc=f"ref2raw_EP[{i}|{self.epoch}]")
            raw2ref=self.ref_runner.infer(infer_img=self.raw_holder.imgs,desc=f"raw2ref_EP[{i}|{self.epoch}]")

            # 生成的图像构造imageholder
            ref2raw_holder=self.ref_holder.copy_config(ref2raw)
            raw2ref_holder=self.raw_holder.copy_config(raw2ref)

            # 使用迁移结果反向迁移回原状态，用结果图像和原图像进行PSNR和SSIM进行约束优化网络
            self.ref_runner.optimize(opt_img=ref2raw_holder.imgs,desc=f"ref2ref_EP[{i}|{self.epoch}]")
            self.raw_runner.optimize(opt_img=raw2ref_holder.imgs,desc=f"raw2raw_EP[{i}|{self.epoch}]")

            # 每轮优化后进行一次推断
            self.ref_runner.infer(infer_img=self.raw_holder.imgs,desc=f"result_EP[{i}|{self.epoch}]")

    def test(self):
        self.raw_runner.train()
        self.ref_runner.train()
