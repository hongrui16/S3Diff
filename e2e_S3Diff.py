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


class S3Diff_network(torch.nn.Module):
    def __init__(self, pretrained_path = None, de_net_path = None, sd_path = './sd_turbo', 
                 lora_rank_unet = 32, lora_rank_vae = 16, block_embedding_dim = 64,
                num_train_timesteps: int = 1000,
                beta_start: float = 0.0001,
                beta_end: float = 0.02,
                clip_sample_range: float = 1.0,
                device = 'cpu',
                   **kwargs):
        super().__init__()

        self.latent_tiled_size = 96
        self.latent_tiled_overlap = 32
        self.vae_decoder_tiled_size = 224
        self.vae_encoder_tiled_size = 1024

        self.prediction_type = "epsilon"
        self.clip_sample = True
        self.clip_sample_range = 1.0
        self.thresholding = False

        self.num_train_timesteps = 1000
        self.num_inference_steps = 1
        beta_start = 0.0001
        beta_end = 0.02
        self.enlarge_ratio = 1

        
        pos_prompt = "A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting."
        neg_prompt = "oil painting, cartoon, blur, dirty, messy, low quality, deformation, low resolution, oversmooth"



        ####<<<<<<<<<<<<<<<<<<<<<<<<-------------scheduling start---------------------
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end


        self.clip_sample_range = clip_sample_range


        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32, device = device)


        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0, device= device)


        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        num_inference_steps = 1
        self.num_inference_steps = num_inference_steps
        
        self.timesteps = torch.tensor([999]).long().to(device)
        ###########--------------scheduling end----------------->>>>>>>>>>>>

        if os.path.exists(pretrained_path):
            pretrained_ckpt = torch.load(pretrained_path, map_location='cpu')
        else:
            pretrained_ckpt = {}
        

        if "pos_prompt_enc" in pretrained_ckpt and "neg_prompt_enc" in pretrained_ckpt:
            self.register_buffer("pos_prompt_enc", pretrained_ckpt["pos_prompt_enc"])
            self.register_buffer("neg_prompt_enc", pretrained_ckpt["neg_prompt_enc"])

        elif isinstance(pos_prompt, str) and isinstance(neg_prompt, str):
            self.tokenizer = AutoTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder")
            self.text_encoder.eval()
            self.text_encoder.to(device)

            pos_ids = self.tokenizer(pos_prompt, max_length=self.tokenizer.model_max_length,
                                    padding="max_length", truncation=True, return_tensors="pt").input_ids
            neg_ids = self.tokenizer(neg_prompt, max_length=self.tokenizer.model_max_length,
                                    padding="max_length", truncation=True, return_tensors="pt").input_ids
            
            pos_ids = pos_ids.to(device)
            neg_ids = neg_ids.to(device)

            with torch.no_grad():
                pos_prompt_enc = self.text_encoder(pos_ids)[0].squeeze(0).to(device)
                neg_prompt_enc = self.text_encoder(neg_ids)[0].squeeze(0).to(device)

            self.register_buffer("pos_prompt_enc", pos_prompt_enc)
            self.register_buffer("neg_prompt_enc", neg_prompt_enc)

        else:
            raise ValueError("pos_prompt and neg_prompt must be strings or precomputed encodings.")
        

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
        self.timesteps = torch.tensor([999], device=device).long()
        self.text_encoder.requires_grad_(False)

        # vae tile
        self._init_tiled_vae(encoder_tile_size=self.vae_encoder_tiled_size, decoder_tile_size=self.vae_decoder_tiled_size)

        self.deres_net = DEResNet(num_in_ch = 3, num_degradation = 2)
        self.deres_net.load_state_dict(torch.load(de_net_path, map_location='cpu'))
        self.deres_net.eval()   
        self.deres_net.to(device)

        self.set_eval()
        all_weights_path = 's3diff_all.pt'
        if not os.path.exists(all_weights_path):
            ## save all weights
            torch.save(self.state_dict(), all_weights_path)


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
        self.text_encoder.requires_grad_(False)
        self.deres_net.requires_grad_(False)

    @perfcount
    @torch.no_grad()
    def forward(self, im_lr):
        ### input: im_lr, [B, 3, 256, 256], 0~1.0
        ### output: output_image, [1, 3, 256, 256], -1.0~1.0

        B = im_lr.shape[0]
        device = im_lr.device
        
        # ori_h, ori_w = im_lr.shape[-2], im_lr.shape[-1]

        # im_lr_resize = F.interpolate(im_lr, size=(ori_h*self.enlarge_ratio, ori_w*self.enlarge_ratio), mode='bilinear', align_corners=False)
        im_lr_resize = im_lr 
        im_lr_resize = torch.clamp(im_lr_resize*2 -1.0, -1.0, 1.0)
        deg_score = self.deres_net(im_lr_resize)

        neg_prompt_enc = self.neg_prompt_enc.unsqueeze(0).expand(B, -1, -1).to(device)
        pos_prompt_enc = self.pos_prompt_enc.unsqueeze(0).expand(B, -1, -1).to(device)

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

        lq_latent = self.vae.encode(im_lr_resize).latent_dist.sample() * self.vae.config.scaling_factor

        ## add tile function
        _, _, h, w = lq_latent.size()
        tile_size, tile_overlap = (self.latent_tiled_size, self.latent_tiled_overlap)
        if h * w <= tile_size * tile_size:
            # print(f"[Tiled Latent]: the input size is tiny and unnecessary to tile.")
            pos_model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=pos_prompt_enc).sample
            neg_model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=neg_prompt_enc).sample
            model_pred = neg_model_pred + self.guidance_scale * (pos_model_pred - neg_model_pred)
        else:
            # print(f"[Tiled Latent]: the input size is {c_t.shape[-2]}x{c_t.shape[-1]}, need to tiled")
            # tile_weights = self._gaussian_weights(tile_size, tile_size, 1).to()
            tile_size = min(tile_size, min(h, w))
            tile_weights = self._gaussian_weights(tile_size, tile_size, 1).to(im_lr_resize.device)

            grid_rows = 0
            cur_x = 0
            while cur_x < lq_latent.size(-1):
                cur_x = max(grid_rows * tile_size-tile_overlap * grid_rows, 0)+tile_size
                grid_rows += 1

            grid_cols = 0
            cur_y = 0
            while cur_y < lq_latent.size(-2):
                cur_y = max(grid_cols * tile_size-tile_overlap * grid_cols, 0)+tile_size
                grid_cols += 1

            input_list = []
            noise_preds = []
            for row in range(grid_rows):
                noise_preds_row = []
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        # extract tile from input image
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    # input tile dimensions
                    input_tile = lq_latent[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                    input_list.append(input_tile)

                    if len(input_list) == 1 or col == grid_cols-1:
                        input_list_t = torch.cat(input_list, dim=0)
                        # predict the noise residual
                        pos_model_pred = self.unet(input_list_t, self.timesteps, encoder_hidden_states=pos_prompt_enc).sample
                        neg_model_pred = self.unet(input_list_t, self.timesteps, encoder_hidden_states=neg_prompt_enc).sample
                        model_out = neg_model_pred + self.guidance_scale * (pos_model_pred - neg_model_pred)
                        input_list = []
                    noise_preds.append(model_out)

            # Stitch noise predictions for all tiles
            noise_pred = torch.zeros(lq_latent.shape, device=lq_latent.device)
            contributors = torch.zeros(lq_latent.shape, device=lq_latent.device)
            # Add each tile contribution to overall latents
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        # extract tile from input image
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row*grid_cols + col] * tile_weights
                    contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
            # Average overlapping areas with more than 1 contributor
            noise_pred /= contributors
            model_pred = noise_pred

        x_denoised = self.step(model_pred, self.timesteps, lq_latent, return_dict=True)
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image

    def save_model(self, outf):
        sd = {}
        sd["unet_lora_target_modules"] = self.target_modules_unet
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip_conv" in k}
        sd["state_dict_vae_de_mlp"] = {k: v for k, v in self.vae_de_mlp.state_dict().items()}
        sd["state_dict_unet_de_mlp"] = {k: v for k, v in self.unet_de_mlp.state_dict().items()}
        sd["state_dict_vae_block_mlp"] = {k: v for k, v in self.vae_block_mlp.state_dict().items()}
        sd["state_dict_unet_block_mlp"] = {k: v for k, v in self.unet_block_mlp.state_dict().items()}
        sd["state_dict_vae_fuse_mlp"] = {k: v for k, v in self.vae_fuse_mlp.state_dict().items()}
        sd["state_dict_unet_fuse_mlp"] = {k: v for k, v in self.unet_fuse_mlp.state_dict().items()}
        sd["w"] = self.W

        sd["state_embeddings"] = {
                    "state_dict_vae_block": self.vae_block_embeddings.state_dict(),
                    "state_dict_unet_block": self.unet_block_embeddings.state_dict(),
                }
        sd['betas'] = self.betas
        # sd['alphas'] = self.alphas
        # sd['alphas_cumprod'] = self.alphas_cumprod
        # sd['final_alpha_cumprod'] = self.final_alpha_cumprod
        # sd['timesteps'] = self.timesteps
        sd['pos_prompt_enc'] = self.pos_prompt_enc
        sd['neg_prompt_enc'] = self.neg_prompt_enc

        torch.save(sd, outf)



    def _set_latent_tile(self,
        latent_tiled_size = 96,
        latent_tiled_overlap = 32):
        self.latent_tiled_size = latent_tiled_size
        self.latent_tiled_overlap = latent_tiled_overlap
    
    def _init_tiled_vae(self,
            encoder_tile_size = 256,
            decoder_tile_size = 256,
            fast_decoder = False,
            fast_encoder = False,
            color_fix = False,
            vae_to_gpu = True):
        # save original forward (only once)
        if not hasattr(self.vae.encoder, 'original_forward'):
            setattr(self.vae.encoder, 'original_forward', self.vae.encoder.forward)
        if not hasattr(self.vae.decoder, 'original_forward'):
            setattr(self.vae.decoder, 'original_forward', self.vae.decoder.forward)

        encoder = self.vae.encoder
        decoder = self.vae.decoder

        self.vae.encoder.forward = VAEHook(
            encoder, encoder_tile_size, is_decoder=False, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)
        self.vae.decoder.forward = VAEHook(
            decoder, decoder_tile_size, is_decoder=True, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights), (nbatches, self.unet.config.in_channels, 1, 1))


    
    def _get_variance(self, t, predicted_variance=None, variance_type=None):
        prev_t = self.previous_timestep(t)

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        # we always take the log of variance, so clamp it to ensure it's not 0
        variance = torch.clamp(variance, min=1e-20)


        variance = variance
        # for rl-diffuser https://arxiv.org/abs/2205.09991

        return variance



    def step(
        self,
        model_output: torch.Tensor,
        timestep: int, ### timestep = 999 in one-step sampling
        sample: torch.Tensor,
        generator=None,
        return_dict: bool = True,):
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        t = timestep

        prev_t = self.previous_timestep(t)

        predicted_variance = None
        # print('predicted_variance', predicted_variance)
        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        # if self.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5) ## prediction_type == "epsilon"

        # 3. Clip or threshold "predicted x_0"
        pred_original_sample = pred_original_sample.clamp(
            -self.clip_sample_range, self.clip_sample_range
        )

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        # return pred_prev_sample

        # 6. Add noise
        variance = 0
        # print('t', t)
        if t > 0:
            device = model_output.device
            layout = torch.strided
            variance_noise = torch.randn(model_output.shape, generator=generator,
                                          device=device, dtype=model_output.dtype, layout=layout).to(device)
            # print('variance_noise', variance_noise.shape)
            variance = (self._get_variance(t) ** 0.5) * variance_noise
            # print('variance', variance.shape)

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample
    



    def previous_timestep(self, timestep):

        num_inference_steps = (
            self.num_inference_steps if self.num_inference_steps else self.num_train_timesteps
        )
        prev_t = timestep - self.num_train_timesteps // num_inference_steps

        return prev_t
    

if __name__ == "__main__":
    import argparse
    import cv2
    parser = argparse.ArgumentParser(description="S3Diff Network")
    parser.add_argument('--pretrained_path', type=str, default='s3diff_all.pt', help='Path to the pretrained model')
    parser.add_argument('--de_net_path', type=str, default='de_net.pth', help='Path to the degradation network')
    parser.add_argument('--sd_path', type=str, default='./sd_turbo', help='Path to the Stable Diffusion model')
    parser.add_argument('--img_path', type=str, default=None, help='Path to the input image')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = S3Diff_network(
        pretrained_path=args.pretrained_path,
        de_net_path=args.de_net_path,
        sd_path=args.sd_path,
        device=device
    ).to(device)

    if args.img_path is not None:   
        img = cv2.imread(args.img_path)
        
        img = cv2.resize(img, (512, 512))
        lr_img = img[128:384, 128:384, :]
        print('lr_img shape:', lr_img.shape)
        cv2.imwrite("input_lr_image.png", lr_img)
        lr_img_rgb = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        lr_img_tensor = torch.from_numpy(lr_img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        lr_img_tensor = lr_img_tensor.to(device)

        hr_img = model(lr_img_tensor)
        hr_img = hr_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        hr_img = ((hr_img * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_RGB2BGR)
        print('hr_img shape:', hr_img.shape)
        cv2.imwrite("output_hr_image.png", hr_img)
        composed_img = np.hstack((lr_img, hr_img))
        # cv2.imshow("Input and Output", composed_img)
        cv2.imwrite("output_image.png", composed_img)
        print("Output image saved as output_image.png")