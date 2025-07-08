import os
import re
import requests
import sys
import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig, get_peft_model

from src.model import make_1step_sched, my_lora_fwd
from basicsr.archs.arch_util import default_init_weights
from src.my_utils.vaehook import VAEHook, perfcount
from src.de_net import DEResNet

def get_layer_number(module_name):
    base_layers = {
        'down_blocks': 0,
        'mid_block': 4,
        'up_blocks': 5
    }

    if module_name == 'conv_out':
        return 9

    base_layer = None
    for key in base_layers:
        if key in module_name:
            base_layer = base_layers[key]
            break

    if base_layer is None:
        return None

    additional_layers = int(re.findall(r'\.(\d+)', module_name)[0]) #sum(int(num) for num in re.findall(r'\d+', module_name))
    final_layer = base_layer + additional_layers
    return final_layer

def tokenize_prompts(sd_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").to(device)
    text_encoder.eval()

    pos_prompt = "A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting."
    neg_prompt = "oil painting, cartoon, blur, dirty, messy, low quality, deformation, low resolution, oversmooth"

    pos_ids = tokenizer(pos_prompt, max_length=tokenizer.model_max_length,
                        padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
    neg_ids = tokenizer(neg_prompt, max_length=tokenizer.model_max_length,
                        padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        pos_prompt_enc = text_encoder(pos_ids)[0].squeeze(0).cpu()
        neg_prompt_enc = text_encoder(neg_ids)[0].squeeze(0).cpu()

    # 保存到 checkpoint 中（或保存为单独 .pt）
    torch.save({'pos_prompt_enc': pos_prompt_enc, 'neg_prompt_enc': neg_prompt_enc}, 'prompt_encodings.pt')

def merge_lora_weights(model, adapter_name="vae_skip"):
    for name, module in model.named_modules():
        if not hasattr(module, "base_layer"):
            continue
        if not hasattr(module, "lora_A"):
            continue
        if not hasattr(module, "de_mod"):
            continue

        lora_A = module.lora_A[adapter_name]
        lora_B = module.lora_B[adapter_name]
        scaling = module.scaling[adapter_name]
        de_mod = module.de_mod[0]  # [r, r]

        # 1. 获取两个矩阵的权重
        A = lora_A.weight.data  # [r, in]  or  [r, in, 1, 1]
        B = lora_B.weight.data  # [out, r] or  [out, r, 1, 1]

        if isinstance(module.base_layer, nn.Linear):
            # 2. 合并 linear 的 LoRA 权重
            merged = B @ (de_mod @ A) * scaling  # [out, in]
            module.base_layer.weight.data += merged

        elif isinstance(module.base_layer, nn.Conv2d):
            # 处理 conv2d，要求 A/B 是 conv with kernel_size=1
            A = A.squeeze(-1).squeeze(-1)  # [r, in]
            B = B.squeeze(-1).squeeze(-1)  # [out, r]
            merged = (B @ (de_mod @ A)).unsqueeze(-1).unsqueeze(-1) * scaling  # [out, in, 1, 1]
            module.base_layer.weight.data += merged

        else:
            raise NotImplementedError("Only Linear and Conv2d are supported.")

        # 3. 清理结构（确保 ONNX 不再引用它们）
        del module.lora_A
        del module.lora_B
        del module.lora_dropout
        del module.scaling
        del module.use_dora
        del module.de_mod
        module.forward = module.base_layer.forward  # 恢复原始 forward

        print(f"[LoRA] Merged and cleaned: {name}")


class S3Diff_network(torch.nn.Module):
    def __init__(self, pretrained_path = None, de_net_path = None, sd_path = './sd_turbo', 
                 lora_rank_unet = 32, lora_rank_vae = 16, block_embedding_dim = 64,
                num_train_timesteps: int = 1000,
                beta_start: float = 0.00085,
                beta_end: float = 0.012,
                clip_sample_range: float = 1.0,
                device = 'cpu',
                   **kwargs):
        super().__init__()



        self.prediction_type = "epsilon"
        self.clip_sample = True
        self.clip_sample_range = 1.0
        self.thresholding = False

        self.num_train_timesteps = 1000
        self.num_inference_steps = 1
        beta_start = 0.0001
        beta_end = 0.02
        self.enlarge_ratio = 1

        
        # pos_prompt = "A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting."
        # neg_prompt = "oil painting, cartoon, blur, dirty, messy, low quality, deformation, low resolution, oversmooth"



        ####<<<<<<<<<<<<<<<<<<<<<<<<-------------scheduling start---------------------
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end


        self.clip_sample_range = clip_sample_range


        betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        one = torch.tensor(1.0, device= device)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("one", one)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        num_inference_steps = 1
        self.num_inference_steps = num_inference_steps
        
        # step_ratio = self.num_train_timesteps / self.num_inference_steps
        # # creates integer timesteps by multiplying by ratio
        # # casting to int to avoid issues when num_inference_step is power of 3
        # timesteps = np.round(np.arange(self.num_train_timesteps, 0, -step_ratio)).astype(np.int64)
        # timesteps -= 1
        # print('timesteps', timesteps)
        # self.timesteps = torch.from_numpy(timesteps).to(device)
        self.timesteps = torch.tensor([999], device=device).long()
        ###########--------------scheduling end----------------->>>>>>>>>>>>

        if os.path.exists(pretrained_path):
            pretrained_ckpt = torch.load(pretrained_path, map_location='cpu')
        else:
            pretrained_ckpt = {}
        


        encodings = torch.load("prompt_encodings.pt")
        self.register_buffer("pos_prompt_enc", encodings["pos_prompt_enc"])
        self.register_buffer("neg_prompt_enc", encodings["neg_prompt_enc"])
        

        self.guidance_scale = 1.07

        vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")

        target_modules_vae = r"^encoder\..*(conv1|conv2|conv_in|conv_shortcut|conv|conv_out|to_k|to_q|to_v|to_out\.0)$"
        target_modules_unet = [
            "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
            "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
        ]

        num_embeddings = 64
        self.W = nn.Parameter(torch.randn(num_embeddings), requires_grad=False)

        self.vae_de_mlp = nn.Sequential(
            nn.Linear(num_embeddings * 4, 256),
            nn.ReLU(True),
        )

        self.unet_de_mlp = nn.Sequential(
            nn.Linear(num_embeddings * 4, 256),
            nn.ReLU(True),
        )

        self.vae_block_mlp = nn.Sequential(
            nn.Linear(block_embedding_dim, 64),
            nn.ReLU(True),
        )

        self.unet_block_mlp = nn.Sequential(
            nn.Linear(block_embedding_dim, 64),
            nn.ReLU(True),
        )

        self.vae_fuse_mlp = nn.Linear(256 + 64, lora_rank_vae ** 2)
        self.unet_fuse_mlp = nn.Linear(256 + 64, lora_rank_unet ** 2)

        default_init_weights([self.vae_de_mlp, self.unet_de_mlp, self.vae_block_mlp, self.unet_block_mlp, \
            self.vae_fuse_mlp, self.unet_fuse_mlp], 1e-5)

        # vae
        self.vae_block_embeddings = nn.Embedding(6, block_embedding_dim)
        self.unet_block_embeddings = nn.Embedding(10, block_embedding_dim)

        vae_lora_config = LoraConfig(r=pretrained_ckpt["rank_vae"], init_lora_weights="gaussian", target_modules=pretrained_ckpt["vae_lora_target_modules"])
        vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        _sd_vae = vae.state_dict()
        for k in pretrained_ckpt["state_dict_vae"]:
            _sd_vae[k] = pretrained_ckpt["state_dict_vae"][k]
        vae.load_state_dict(_sd_vae)

        unet_lora_config = LoraConfig(r=pretrained_ckpt["rank_unet"], init_lora_weights="gaussian", target_modules=pretrained_ckpt["unet_lora_target_modules"])
        unet.add_adapter(unet_lora_config)
        _sd_unet = unet.state_dict()
        for k in pretrained_ckpt["state_dict_unet"]:
            _sd_unet[k] = pretrained_ckpt["state_dict_unet"][k]
        unet.load_state_dict(_sd_unet)

        _vae_de_mlp = self.vae_de_mlp.state_dict()
        for k in pretrained_ckpt["state_dict_vae_de_mlp"]:
            _vae_de_mlp[k] = pretrained_ckpt["state_dict_vae_de_mlp"][k]
        self.vae_de_mlp.load_state_dict(_vae_de_mlp)

        _unet_de_mlp = self.unet_de_mlp.state_dict()
        for k in pretrained_ckpt["state_dict_unet_de_mlp"]:
            _unet_de_mlp[k] = pretrained_ckpt["state_dict_unet_de_mlp"][k]
        self.unet_de_mlp.load_state_dict(_unet_de_mlp)

        _vae_block_mlp = self.vae_block_mlp.state_dict()
        for k in pretrained_ckpt["state_dict_vae_block_mlp"]:
            _vae_block_mlp[k] = pretrained_ckpt["state_dict_vae_block_mlp"][k]
        self.vae_block_mlp.load_state_dict(_vae_block_mlp)

        _unet_block_mlp = self.unet_block_mlp.state_dict()
        for k in pretrained_ckpt["state_dict_unet_block_mlp"]:
            _unet_block_mlp[k] = pretrained_ckpt["state_dict_unet_block_mlp"][k]
        self.unet_block_mlp.load_state_dict(_unet_block_mlp)

        _vae_fuse_mlp = self.vae_fuse_mlp.state_dict()
        for k in pretrained_ckpt["state_dict_vae_fuse_mlp"]:
            _vae_fuse_mlp[k] = pretrained_ckpt["state_dict_vae_fuse_mlp"][k]
        self.vae_fuse_mlp.load_state_dict(_vae_fuse_mlp)

        _unet_fuse_mlp = self.unet_fuse_mlp.state_dict()
        for k in pretrained_ckpt["state_dict_unet_fuse_mlp"]:
            _unet_fuse_mlp[k] = pretrained_ckpt["state_dict_unet_fuse_mlp"][k]
        self.unet_fuse_mlp.load_state_dict(_unet_fuse_mlp)

        self.W = nn.Parameter(pretrained_ckpt["w"], requires_grad=False)

        embeddings_state_dict = pretrained_ckpt["state_embeddings"]
        self.vae_block_embeddings.load_state_dict(embeddings_state_dict['state_dict_vae_block'])
        self.unet_block_embeddings.load_state_dict(embeddings_state_dict['state_dict_unet_block'])
        

        self.lora_rank_unet = lora_rank_unet
        self.lora_rank_vae = lora_rank_vae
        self.target_modules_vae = target_modules_vae
        self.target_modules_unet = target_modules_unet

        self.vae_lora_layers = []
        for name, module in vae.named_modules():
            if 'base_layer' in name:
                self.vae_lora_layers.append(name[:-len(".base_layer")])
                
        for name, module in vae.named_modules():
            if name in self.vae_lora_layers:
                module.forward = my_lora_fwd.__get__(module, module.__class__)

        self.unet_lora_layers = []
        for name, module in unet.named_modules():
            if 'base_layer' in name:
                self.unet_lora_layers.append(name[:-len(".base_layer")])

        for name, module in unet.named_modules():
            if name in self.unet_lora_layers:
                module.forward = my_lora_fwd.__get__(module, module.__class__)

        self.unet_layer_dict = {name: get_layer_number(name) for name in self.unet_lora_layers}

        unet.to(device)
        vae.to(device)

        self.unet, self.vae = unet, vae


        self.deres_net = DEResNet(num_in_ch = 3, num_degradation = 2)
        self.deres_net.load_state_dict(torch.load(de_net_path, map_location='cpu'))
        self.deres_net.eval()   
        self.deres_net.to(device)

        self.set_eval()
        # all_weights_path = 's3diff_all.pt'
        # if not os.path.exists(all_weights_path):
        #     ## save all weights
        #     torch.save(self.state_dict(), all_weights_path)



    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.vae_de_mlp.eval()
        self.unet_de_mlp.eval()
        self.vae_block_mlp.eval()
        self.unet_block_mlp.eval()
        self.vae_fuse_mlp.eval()
        self.unet_fuse_mlp.eval()

        self.vae_block_embeddings.requires_grad_(False)
        self.unet_block_embeddings.requires_grad_(False)

        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        # self.text_encoder.requires_grad_(False)
        self.deres_net.requires_grad_(False)

    def forward(self, im_lr):
        '''
        
        Args:            im_lr: [1, 3, 256, 256], 0~1.0, the input size is up to 768x768 without titling. 为了统一, 转成-1~1.0
        Returns:         output_image: [1, 3, 256, 256], 0~1.0
        '''
        B = im_lr.shape[0]
        device = im_lr.device

        neg_prompt_enc = self.neg_prompt_enc.unsqueeze(0).expand(B, -1, -1).to(device)
        pos_prompt_enc = self.pos_prompt_enc.unsqueeze(0).expand(B, -1, -1).to(device)
        
        # im_lr_resize = torch.clamp(im_lr*2 -1.0, -1.0, 1.0)


        deg_score = self.deres_net(im_lr)
        
    
        # degradation fourier embedding
        deg_proj = deg_score[..., None] * self.W[None, None, :] * 2 * np.pi
        deg_proj = torch.cat([torch.sin(deg_proj), torch.cos(deg_proj)], dim=-1)
        deg_proj = torch.cat([deg_proj[:, 0], deg_proj[:, 1]], dim=-1)

        # degradation mlp forward
        vae_de_c_embed = self.vae_de_mlp(deg_proj)
        unet_de_c_embed = self.unet_de_mlp(deg_proj)

        # block embedding mlp forward
        vae_block_c_embeds = self.vae_block_mlp(self.vae_block_embeddings.weight)
        unet_block_c_embeds = self.unet_block_mlp(self.unet_block_embeddings.weight)

        vae_embeds = self.vae_fuse_mlp(torch.cat([vae_de_c_embed.unsqueeze(1).repeat(1, vae_block_c_embeds.shape[0], 1), \
            vae_block_c_embeds.unsqueeze(0).repeat(vae_de_c_embed.shape[0],1,1)], -1))
        unet_embeds = self.unet_fuse_mlp(torch.cat([unet_de_c_embed.unsqueeze(1).repeat(1, unet_block_c_embeds.shape[0], 1), \
            unet_block_c_embeds.unsqueeze(0).repeat(unet_de_c_embed.shape[0],1,1)], -1))

        for layer_name, module in self.vae.named_modules():
            if layer_name in self.vae_lora_layers:
                split_name = layer_name.split(".")
                if split_name[1] == 'down_blocks':
                    block_id = int(split_name[2])
                    vae_embed = vae_embeds[:, block_id]
                elif split_name[1] == 'mid_block':
                    vae_embed = vae_embeds[:, -2]
                else:
                    vae_embed = vae_embeds[:, -1]
                module.de_mod = vae_embed.reshape(-1, self.lora_rank_vae, self.lora_rank_vae)

        for layer_name, module in self.unet.named_modules():
            if layer_name in self.unet_lora_layers:
                split_name = layer_name.split(".")
                if split_name[0] == 'down_blocks':
                    block_id = int(split_name[1])
                    unet_embed = unet_embeds[:, block_id]
                elif split_name[0] == 'mid_block':
                    unet_embed = unet_embeds[:, 4]
                elif split_name[0] == 'up_blocks':
                    block_id = int(split_name[1]) + 5
                    unet_embed = unet_embeds[:, block_id]
                else:
                    unet_embed = unet_embeds[:, -1]
                module.de_mod = unet_embed.reshape(-1, self.lora_rank_unet, self.lora_rank_unet)

        lq_latent = self.vae.encode(im_lr).latent_dist.sample() * self.vae.config.scaling_factor

        ## add tile function

        pos_model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=pos_prompt_enc).sample
        neg_model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=neg_prompt_enc).sample
        model_pred = neg_model_pred + self.guidance_scale * (pos_model_pred - neg_model_pred)
        

        ### x_denoised = self.step(model_pred, self.timesteps, lq_latent) replace with the following one-step sampling
        alpha_t = self.alphas_cumprod[999]
        beta_t = 1.0 - alpha_t
        x_denoised = (lq_latent - beta_t.sqrt() * model_pred) / alpha_t.sqrt()

        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image



if __name__ == "__main__":
    import argparse
    import cv2
    parser = argparse.ArgumentParser(description="S3Diff Network")
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to the lora weights')
    parser.add_argument('--de_net_path', type=str, default='de_net.pth', help='Path to the degradation network')
    parser.add_argument('--sd_path', type=str, default='./sd_turbo', help='Path to the Stable Diffusion model')
    parser.add_argument('--img_path', type=str, default=None, help='Path to the input image')
    parser.add_argument('--use_fp16', action='store_true', help='Use FP16 for inference')
    parser.add_argument('--onnx_dir', type=str, default='./onnx_models', help='Directory to save the ONNX model')


    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    export_fp16 = args.use_fp16 or device.type == 'cuda'

    ### generate prompt encodings if not exist
    if not os.path.exists("prompt_encodings.pt"):
        print("Generating prompt encodings...")
        # pos_prompt = "A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting."
        # neg_prompt = "oil painting, cartoon, blur, dirty, messy, low quality, deformation, low resolution, oversmooth"
        tokenize_prompts(args.sd_path)


    model = S3Diff_network(
        pretrained_path=args.pretrained_path,
        de_net_path=args.de_net_path,
        sd_path=args.sd_path,
        device=device
    ).to(device)

    parent_dir = os.path.dirname(args.pretrained_path)

    merge_lora_weights(model.vae, adapter_name="vae_skip")
    merge_lora_weights(model.unet, adapter_name="default")

    model.eval()
    # model.half()  # or keep float32 for now
    for p in model.parameters():
        p.requires_grad = False

    if export_fp16:
        model = model.half()
        dummy_input = torch.rand(1, 3, 256, 256).half().to(device)  # if using half
        precision = 'fp16'
        datatype = 'float16'
    else:
        model = model.float()
        dummy_input = torch.rand(1, 3, 256, 256).float().to(device)  # if using float
        precision = 'fp32'
        datatype = 'float32'

    onnx_dir = os.path.join(args.onnx_dir, precision)
    os.makedirs(onnx_dir, exist_ok=True)
    onnx_path = os.path.join(onnx_dir, f"s3diff_{precision}.onnx")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["im_lr"],
            output_names=["out_im"],
            opset_version=17,
            do_constant_folding=True,
            # dynamic_axes={"im_lr": {0: "batch"}, "out_im": {0: "batch"}},
        )

    if args.img_path is not None:   
        img = cv2.imread(args.img_path)
        
        img = cv2.resize(img, (512, 512))
        lr_img = img[128:384, 128:384, :]
        print('lr_img shape:', lr_img.shape)
        cv2.imwrite(f"input_lr_image.png", lr_img)
        lr_img_rgb = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        lr_img_tensor = torch.from_numpy(lr_img_rgb) / 255.0
        lr_img_tensor = lr_img_tensor * 2.0 - 1.0  # Convert to range [-1, 1]
        lr_img_tensor = lr_img_tensor.to(datatype)
        lr_img_tensor = lr_img_tensor.permute(2, 0, 1).unsqueeze(0)
        lr_img_tensor = lr_img_tensor.to(device)

        hr_img = model(lr_img_tensor)

        hr_img = hr_img.detach()
        hr_img = hr_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        hr_img = ((hr_img * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_RGB2BGR)
        print('hr_img shape:', hr_img.shape)
        cv2.imwrite(f"output_hr_image_{precision}.png", hr_img)
        composed_img = np.hstack((lr_img, hr_img))
        # cv2.imshow("Input and Output", composed_img)
        cv2.imwrite(f"compose_image_{precision}.png", composed_img)
        print(f"Output image saved as compose_image_{precision}.png")