# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchvision.utils import save_image, make_grid
from masked_diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from masked_diffusion.models import XMDPT_S_2, XMDPT_B_2, XMDPT_L_2
import os
from huggingface_hub import snapshot_download
from masked_diffusion.my_dataset import load_my_data
from einops import rearrange

from torch.cuda.amp import autocast
from diffusers import StableDiffusionPipeline

# Setup PyTorch:
torch.manual_seed(0)
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
num_sampling_steps = 50
cfg_scale = 2.0
pow_scale = 0.01 # large pow_scale increase the diversity, small pow_scale increase the quality.
# model_path = 'mdt_xl2_v1_ckpt.pt'
# #### hugging face
# ckpt_model_path = snapshot_download("shgao/MDT-XL2")
# model_path = os.path.join(ckpt_model_path, "mdt_xl2_v1_ckpt.pt")
# #### hugging face
# model_path = 'logs/XMDPT_L_2/ema_0.9999_300000.pt'
# model_path = 'logs/XMDPT_B_2/ema_0.9999_300000.pt'
model_path = 'logs/XMDPT_S_2/ema_0.9999_300000.pt'

# Load model:
image_size = 256
assert image_size in [256], "We provide pre-trained models for 256x256 resolutions for now."
latent_size = image_size // 8
# model = XMDPT_L_2(input_size=latent_size, decode_layer=2).to(device)
# model = XMDPT_B_2(input_size=latent_size, decode_layer=2).to(device)
model = XMDPT_S_2(input_size=latent_size, decode_layer=2).to(device)

state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
model.load_state_dict(state_dict)

model.eval()
diffusion = create_diffusion(str(num_sampling_steps))
# vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
model_name = 'model_zoo/diffuser_finetune_vae'
vae = AutoencoderKL.from_pretrained(model_name).to(device)

# ### Chillout model gives better human face
# model_id = "emilianJR/chilloutmix_NiPrunedFp32Fix"
# model_id = "NickKolok/Realistic_Vision_V60_B1-conv" #, # NickKolok/Realistic_Vision_V60_B1-conv, SG161222/RealVisXL_V3.0_Turbo
# pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
# vae = pipe.vae
# ### Chillout model gives better human face

### start PXT
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
dino_v2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to(device)
dino_v2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
dino_v2_vitg14.eval()
dino_v2_vitb14.eval()
num_samples = 8
## test dataloader
dl_val = load_my_data(data_dir='./datasets/deepfashion/',
                    batch_size=num_samples,
                    is_inference=True,
                    labels_required=True)
iter = 0
for batch, cond in dl_val:
    print(f'Iteration {iter:03d}')
    img_s = batch['source_image'].cuda()
    img_t = batch['target_image'].cuda()
    pose_t = batch['target_skeleton'].cuda()
    dino_tgt_pose = batch['dino_pose_tgt'].cuda()
    dino_src_img = batch['dino_src'].cuda()
    with torch.no_grad():
        with autocast(dtype=torch.bfloat16):
            tmp = dino_v2_vitb14.get_intermediate_layers(dino_tgt_pose.to(device), 1, return_class_token=True)[0]
            dino_tgt_pose = torch.cat((tmp[1].unsqueeze(1), tmp[0]), dim=1)
            tmp = dino_v2_vitg14.get_intermediate_layers(dino_src_img.to(device), 1, return_class_token=True)[0]
            dino_src_img = torch.cat((tmp[1].unsqueeze(1), tmp[0]), dim=1)
        
        src_img = vae.encode(img_s, return_dict=True)[0].sample()*0.18215
    # Labels to condition the model
    class_labels = [0]*num_samples
    # Create sampling noise
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)

    model_kwargs = dict(cfg_scale=cfg_scale, scale_pow=pow_scale, 
                        dino_tgt_pose=dino_tgt_pose,
                        dino_src_img=dino_src_img,
                        src_img=src_img,
                        )
    
    ### use DDIM, PXT modified
    # DDIM solver
    samples = diffusion.ddim_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    
    samples = vae.decode(samples / 0.18215).sample
    
    recon_target = vae.encode(img_t)[0].sample()*0.18215
    recon_target = vae.decode(recon_target/0.1825).sample

    # Save and display images:
    all_images = torch.stack((img_s, img_t, recon_target, pose_t*2-1, samples),dim=1) # pose_t*2-1 to convert to -1 to 1, otherwise it is grayscale as in paper
    all_images = rearrange(all_images, 'r b ... -> (b r) ...')
    grid = make_grid(all_images, nrow = 8)
    img_saving_dir = 'test_img/' + 'XMDPT_S_2'# 'XMDPT_S_2', 'XMDPT_B_2', 'XMDPT_L_2'
    os.makedirs(img_saving_dir, exist_ok=True)
    image_name = os.path.join(img_saving_dir, f'gen_img_{iter:03d}.png')
    save_image(grid, image_name, normalize=True, value_range=(-1, 1))
    iter += 1
    if iter == 10: # you can generate more images by increasing the number
        break
