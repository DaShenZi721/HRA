from oldm.hack import disable_verbosity
disable_verbosity()

import os
import sys
import torch
from datetime import datetime

file_path = os.path.abspath(__file__)
parent_dir = os.path.abspath(os.path.dirname(file_path) + '/..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from oldm.logger import ImageLogger
from oldm.model import create_model, load_state_dict
from dataset.utils import return_dataset

from oft import inject_trainable_oft, inject_trainable_oft_conv, inject_trainable_oft_extended, inject_trainable_oft_with_norm
from hra import inject_trainable_hra
from lora import inject_trainable_lora

import argparse


parser = argparse.ArgumentParser()

# HRA
parser.add_argument('--hra_r', type=int, default=8)
parser.add_argument('--hra_apply_GS', action='store_true', default=False)
parser.add_argument('--hra_lamb', type=float, default=0.0)

# OFT
parser.add_argument('--oft_r', type=int, default=4)
parser.add_argument('--oft_eps', type=float, default=7e-6)
parser.add_argument('--oft_coft', action="store_true", default=True)
parser.add_argument('--oft_block_share', action="store_true", default=False)

parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_samples', type=int, default=1)
parser.add_argument('--plot_frequency', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=9e-4)
parser.add_argument('--sd_locked', action="store_true", default=True)
parser.add_argument('--only_mid_control', action="store_true", default=False)
parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count())
parser.add_argument('--resume_path', 
                    type=str, 
                    default='./models/hra_half_init_l_8.ckpt',
                    )
parser.add_argument('--time_str', type=str, default=datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"))
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--control', 
                    type=str, 
                    help='control signal. Options are [segm, sketch, densepose, depth, canny, landmark]', 
                    default="segm")

args = parser.parse_args()


if __name__ == "__main__":
    # specify the control signal and dataset
    control = args.control
    
    # create dataset
    train_dataset, val_dataset, data_name, logger_freq, max_epochs = return_dataset(control) # , n_samples=n_samples)

    # Configs
    resume_path = args.resume_path
    
    batch_size = args.batch_size
    num_samples = args.num_samples
    plot_frequency = args.plot_frequency
    learning_rate = args.learning_rate
    sd_locked = args.sd_locked
    only_mid_control = args.only_mid_control
    num_gpus = args.num_gpus
    time_str = args.time_str
    num_workers = args.num_workers
    
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    print(f'data_name: {data_name}\nlogger_freq: {logger_freq}\nmax_epochs: {max_epochs}')
    
    if 'oft' in args.resume_path:
        experiment = 'oft_{}_{}_eps_{}_pe_diff_mlp_r_{}_{}gpu_{}'.format(data_name, control, args.oft_eps, args.oft_r, num_gpus, time_str)
    elif 'hra' in args.resume_path:
        if args.hra_apply_GS:
            experiment = 'hra_apply_GS_{}_{}_pe_diff_mlp_r_{}_{}gpu_{}'.format(data_name, control, args.hra_r, num_gpus, time_str)
        else:
            experiment = 'hra_{}_{}_pe_diff_mlp_r_{}_lambda_{}_lr_{}_{}gpu_{}'.format(data_name, control, args.hra_r,  args.hra_lamb, args.learning_rate, num_gpus, time_str)
    elif 'lora' in args.resume_path:
        experiment = 'lora_{}_{}_pe_diff_mlp_r_{}_{}gpu_{}'.format(data_name, control, args.r, num_gpus, time_str)

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./configs/oft_ldm_v15.yaml').cpu()
    model.model.requires_grad_(False)
    print(f'Total parameters not requiring grad: {sum([p.numel() for p in model.model.parameters() if p.requires_grad == False])}')

    # inject trainable oft parameters
    if 'oft' in args.resume_path:
        unet_lora_params, train_names = inject_trainable_oft(model.model, r=args.oft_r, eps=args.oft_eps, is_coft=args.oft_coft, block_share=args.oft_block_share)
    elif 'hra' in args.resume_path:
        unet_lora_params, train_names = inject_trainable_hra(model.model, r=args.hra_r, apply_GS=args.hra_apply_GS)
    elif 'lora' in args.resume_path:
        unet_lora_params, train_names = inject_trainable_lora(model.model, rank=args.r, network_alpha=None)
    
    print(f'Total parameters requiring grad: {sum([p.numel() for p in model.model.parameters() if p.requires_grad == True])}')

    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    checkpoint_callback = ModelCheckpoint(
        dirpath='log/image_log_' + experiment,
        filename='model-{epoch:02d}',
        save_top_k=-1,
        save_last=True,
        every_n_epochs=1,
        monitor=None,  # No specific metric to monitor for saving
    )

    # Misc
    train_dataloader = DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, num_workers=num_workers, batch_size=1, shuffle=False)

    logger = ImageLogger(
        val_dataloader=val_dataloader,
        batch_frequency=logger_freq, 
        experiment=experiment, 
        plot_frequency=plot_frequency,
        num_samples=num_samples,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gpus=num_gpus, 
        precision=32, 
        callbacks=[logger, checkpoint_callback],
    )

    # Train!
    last_model_path = 'log/image_log_' + experiment + '/last.ckpt'
    if os.path.exists(last_model_path):
        trainer.fit(model, train_dataloader, ckpt_path=last_model_path)
    else:
        trainer.fit(model, train_dataloader)
