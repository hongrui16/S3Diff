
import os
import gc
import tqdm
import math
import lpips
import pyiqa
import argparse
import clip
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from pathlib import Path

from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
# from tqdm.auto import tqdm

import diffusers

from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler


import sys


from utils import misc as misc
from utils.wavelet_color import wavelet_color_fix, adain_color_fix
from utils import util_image
from utils.util_image import ImageSpliterTh


from src.de_net import DEResNet
from src.s3diff_tile import S3Diff
from src.my_utils.testing_utils import parse_args_paired_testing, PlainDataset, lr_proc
from src.my_utils.utils import instantiate_from_config


def main(args):
    config = OmegaConf.load(args.base_config)

    if args.pretrained_path is None:
        from huggingface_hub import hf_hub_download
        pretrained_path = hf_hub_download(repo_id="zhangap/S3Diff", filename="s3diff.pkl")
    else:
        pretrained_path = args.pretrained_path

    if args.sd_path is None:
        from huggingface_hub import snapshot_download
        sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")
    else:
        sd_path = args.sd_path

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    if args.seed is not None:
        set_seed(args.seed)

    
    os.makedirs(args.output_dir, exist_ok=True)

    net_sr = S3Diff(lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae,
                     sd_path=sd_path, pretrained_path=pretrained_path, args=args,
                     device = device,
                     )
    net_sr.set_eval()

    net_de = DEResNet(num_in_ch=3, num_degradation=2)
    net_de.load_model(args.de_net_path)
    net_de = net_de.to(device)
    net_de.eval()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_sr.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers not available. Try `pip install xformers`.")

    if args.gradient_checkpointing:
        net_sr.unet.enable_gradient_checkpointing()

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    net_sr, net_de = accelerator.prepare(net_sr, net_de)
        
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    if device == 'cpu':
        weight_dtype = torch.float32
    
    net_sr.to(accelerator.device, dtype=weight_dtype)
    net_de.to(accelerator.device, dtype=weight_dtype)

    if not args.input_dir is None:
        input_image_list = sorted(
            sum([list(Path(args.input_dir).glob(ext)) for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']], [])
        )
    elif not args.img_path is None:
        input_image_list = [Path(args.img_path)]
    else:
        raise ValueError("Please provide either --input_dir or --img_path.")
    print(f'num images: {len(input_image_list)}')
    for img_path in tqdm.tqdm(input_image_list):
        im_lr = util_image.imread(img_path, chn='rgb', dtype='float32')  # HWC float32
        im_lr = util_image.img2tensor(im_lr).to(device)                      # 1CHW float32
        im_lr = im_lr.to(memory_format=torch.contiguous_format).to(weight_dtype)
    
        ori_h, ori_w = im_lr.shape[2:]
        if config.sf != 1:
            im_lr = F.interpolate(
                im_lr,
                size=(ori_h // config.sf, ori_w // config.sf),
                mode='bilinear',
                align_corners=False
            )
        else:
            im_lr_resize = torch.clamp(im_lr * 2 - 1.0, -1.0, 1.0)

        im_lr_resize = im_lr_resize.to(weight_dtype)

        resize_h, resize_w = im_lr_resize.shape[2:]
        pad_h = (math.ceil(resize_h / 64)) * 64 - resize_h
        pad_w = (math.ceil(resize_w / 64)) * 64 - resize_w
        if pad_h > 0 or pad_w > 0:
            im_lr_resize = im_lr_resize.float()  # 确保是 float32
            im_lr_resize = F.pad(im_lr_resize, pad=(0, pad_w, 0, pad_h), mode='replicate')
            im_lr_resize = im_lr_resize..to(weight_dtype)

    
        with torch.no_grad():
            deg_score = net_de(im_lr)
            B = im_lr_resize.shape[0]
            pos_tag_prompt = [args.pos_prompt] * B
            neg_tag_prompt = [args.neg_prompt] * B
    
            x_tgt_pred = accelerator.unwrap_model(net_sr)(
                im_lr_resize, deg_score,
                pos_prompt=pos_tag_prompt,
                neg_prompt=neg_tag_prompt
            )
    
            # remove padding
            x_tgt_pred = x_tgt_pred[:, :, :resize_h, :resize_w]
            out_img = (x_tgt_pred * 0.5 + 0.5).cpu().detach()
    
        out_pil = transforms.ToPILImage()(out_img[0])
    
        # if args.align_method != 'nofix':
        #     # ⬇️ crop the padded input to match output size
        #     ref_tensor = im_lr_resize[0, :, :resize_h, :resize_w].cpu()
        #     ref_pil = transforms.ToPILImage()(ref_tensor)
    
        #     if out_pil.size != ref_pil.size:
        #         ref_pil = ref_pil.resize(out_pil.size, Image.BICUBIC)
    
        #     if args.align_method == 'wavelet':
        #         out_pil = wavelet_color_fix(out_pil, ref_pil)
        #     elif args.align_method == 'adain':
        #         out_pil = adain_color_fix(out_pil, ref_pil)
    
        fname = img_path.stem + '_S3Diff.png'
        out_pil.save(os.path.join(args.output_dir, fname))

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    args = parse_args_paired_testing()
    main(args)
