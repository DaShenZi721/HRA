from oldm.hack import disable_verbosity
disable_verbosity()

import sys
import os
import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import json
import argparse

file_path = os.path.abspath(__file__)
parent_dir = os.path.abspath(os.path.dirname(file_path) + '/..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from PIL import Image
from pytorch_lightning import seed_everything
from oldm.model import create_model, load_state_dict
from oldm.ddim_hacked import DDIMSampler
from oft import inject_trainable_oft, inject_trainable_oft_conv, inject_trainable_oft_extended, inject_trainable_oft_with_norm
from householder import inject_trainable_householder, inject_trainable_householder_conv, inject_trainable_householder_extended, inject_trainable_householder_with_norm
from lora import inject_trainable_lora

from dataset.utils import return_dataset


def process(input_image, prompt, hint_image, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
    with torch.no_grad():
        # img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = input_image.shape

        #detected_map = apply_canny(input_image, low_threshold, high_threshold)
        #detected_map = HWC3(detected_map)

        # control = torch.from_numpy(hint_image.copy()).float().cuda() / 255.0
        control = torch.from_numpy(hint_image.copy()).float().cuda()
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        # if config.save_memory:
        #     model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        # if config.save_memory:
        #     model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        # if config.save_memory:
        #     model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    # return [255 - hint_image] + results
    return [input_image] + [hint_image] + results

parser = argparse.ArgumentParser()

parser.add_argument('--d', type=int, help='the index of GPU', default=0)

parser.add_argument('--l', type=int, default=8)
parser.add_argument('--add_orth', type=str, default='gramschmidt')
# none, gramschmidt
parser.add_argument('--r', type=int, default=4)
parser.add_argument('--eps', 
                    type=float, 
                    choices=[1e-3, 2e-5, 7e-6],
                    default=7e-6,
                    )
parser.add_argument('--coft', action="store_true", default=True)
parser.add_argument('--block_share', action="store_true", default=False)
parser.add_argument('--img_ID', type=int, default=1)
parser.add_argument('--num_samples', type=int, default=8)
parser.add_argument('--batch', type=int, default=20)
parser.add_argument('--sd_locked', action="store_true", default=True)
parser.add_argument('--only_mid_control', action="store_true", default=False)
parser.add_argument('--num_gpus', type=int, default=8)
# parser.add_argument('--time_str', type=str, default=datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"))
parser.add_argument('--time_str', type=str, default='2024-03-18-10-55-21-089985')
parser.add_argument('--epoch', type=int, default=19)
parser.add_argument('--control', 
                    type=str, 
                    help='control signal. Options are [segm, sketch, densepose, depth, canny, landmark]', 
                    default="canny")

args = parser.parse_args()

if __name__ == '__main__':
    # Configs
    epoch = args.epoch
    control = args.control
    _, dataset, data_name, logger_freq, max_epochs = return_dataset(control, full=True)

    # specify the experiment name
    # experiment = './log/image_log_oft_{}_{}_eps_{}_pe_diff_mlp_r_{}_{}gpu'.format(data_name, control, args.eps, args.r, args.num_gpus)
    
    num_gpus = args.num_gpus
    time_str = args.time_str
    # experiment = 'log/image_log_oft_{}_{}_eps_{}_pe_diff_mlp_r_{}_{}gpu_{}'.format(data_name, control, args.eps, args.r, num_gpus, time_str)
    experiment = './log/image_log_householder_gramschmidt_COCO_canny_eps_7e-06_pe_diff_mlp_l_8_8gpu_2024-05-19-21-22-24-466032'
    # experiment = './log/image_log_oft_ADE20K_segm_eps_0.001_pe_diff_mlp_r_4_8gpu_2024-03-25-21-04-17-549433/train_with_norm'
    
    assert args.control in experiment
    
    if 'train_with_norm' in experiment:
        epoch = 4
    else:
        if 'COCO' in experiment:
            epoch = 0
        else:
            epoch = 19
            
    resume_path = os.path.join(experiment, f'model-epoch={epoch:02d}.ckpt')
    sd_locked = args.sd_locked
    only_mid_control = args.only_mid_control
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Result directory
    result_dir = os.path.join(experiment, 'results', str(epoch))
    os.makedirs(result_dir, exist_ok=True)
    source_dir = os.path.join(experiment, 'source', str(epoch))
    os.makedirs(source_dir, exist_ok=True)
    hint_dir = os.path.join(experiment, 'hints', str(epoch))
    os.makedirs(hint_dir, exist_ok=True)

    model = create_model('./configs/oft_ldm_v15.yaml').cpu()
    model.model.requires_grad_(False)

    if 'householder' in experiment:
        unet_lora_params, train_names = inject_trainable_householder(model.model, l=args.l, eps=args.eps, add_orth=args.add_orth)
    elif 'lora' in experiment:
        unet_lora_params, train_names = inject_trainable_lora(model.model, rank=args.r, network_alpha=None)
    else:
        if 'train_with_norm' in experiment:
            unet_opt_params, train_names = inject_trainable_oft_with_norm(model.model, r=args.r, eps=args.eps, is_coft=args.coft, block_share=args.block_share)
        else:   
            unet_lora_params, train_names = inject_trainable_oft(model.model, r=args.r, eps=args.eps, is_coft=args.coft, block_share=args.block_share)
        # unet_lora_params, train_names = inject_trainable_oft_conv(model.model, r=args.r, eps=args.eps, is_coft=args.coft, block_share=args.block_share)
        # unet_lora_params, train_names = inject_trainable_oft_extended(model.model, r=args.r, eps=args.eps, is_coft=args.coft, block_share=args.block_share)
    
    
    model.load_state_dict(load_state_dict(resume_path, location='cuda'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    # pack = range(0, len(dataset), args.batch)
    # formatted_data = {}
    # for index in range(args.batch):
    #     # import ipdb; ipdb.set_trace()
    #     start_point = pack[args.img_ID]
    #     idx = start_point + index
    
    # num_pack = len(dataset) // args.num_gpus
    # start_idx = args.d * num_pack
    # end_idx = (args.d + 1) * num_pack if args.d < args.num_gpus - 1 else len(dataset)
    
    # for idx in range(start_idx, end_idx):
    
    # canny
    img_list = [378, 441, 0, 31, 115, 182, 59, 60, 66, 269, ]
    # landmark
    # img_list = [139, 179, 197, 144, 54, 71, 76, 98, 100, 277, ]
    # segm
    # img_list = [14, 667, 576, 1387, 1603, 1697, 987, 1830, 1232, 1881, ]
    
    for idx in img_list:
    
        data = dataset[idx]
        input_image, prompt, hint = data['jpg'], data['txt'], data['hint']
        # input_image, hint = input_image.to(device), hint.to(device)
        
        if not os.path.exists(os.path.join(result_dir, f'result_{idx}_0.png')):
            result_images = process(
                input_image=input_image, 
                prompt=prompt,
                hint_image=hint,
                a_prompt="",
                n_prompt="",
                num_samples=args.num_samples,
                image_resolution=512,
                ddim_steps=50,
                guess_mode=False,
                strength=1,
                scale=9.0,
                seed=-1,
                eta=0.0,
                low_threshold=100,
                high_threshold=200,
            )
            for i, image in enumerate(result_images):
                if i == 0:
                    image = ((image + 1) * 127.5).clip(0, 255).astype(np.uint8)
                    pil_image = Image.fromarray(image)
                    output_path = os.path.join(source_dir, f'image_{idx}.png')
                    pil_image.save(output_path)
                elif i == 1:
                    image = (image * 255).clip(0, 255).astype(np.uint8)
                    # Convert numpy array to PIL Image
                    pil_image = Image.fromarray(image)
                    # Save PIL Image to file
                    output_path = os.path.join(hint_dir, f'hint_{idx}.png')
                    pil_image.save(output_path)
                else:
                    n = i - 2
                    # Convert numpy array to PIL Image
                    pil_image = Image.fromarray(image)
                    # Save PIL Image to file
                    output_path = os.path.join(result_dir, f'result_{idx}_{n}.png')
                    pil_image.save(output_path)

        # formatted_data[f"item{idx}"] = {
        #     "image_name": f'result_{idx}.png',
        #     "prompt": prompt
        # }

    # with open(os.path.join(experiment, 'results_{}.json'.format(img_ID)), 'w') as f:
    #     json.dump(formatted_data, f)

        
