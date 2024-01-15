import os
import os.path as osp
import sys
sys.path.insert(0, '/'.join(osp.realpath(__file__).split('/')[:-4]))
import json
import math
import random
import torch
import logging
import datetime
import itertools
import numpy as np
from PIL import Image
import torch.optim as optim
from einops import rearrange
import wandb
import torch.cuda.amp as amp
from importlib import reload
from copy import deepcopy, copy
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from lora_diffusion import inject_trainable_lora, extract_lora_ups_down,inject_trainable_lora_extended

# from peft import LoraConfig
# from peft.utils import get_peft_model_state_dict

import utils.transforms as data
from utils.util import to_device
from ..modules.config import cfg
from utils.seed import setup_seed
from utils.optim import AnnealingLR
from utils.multi_port import find_free_port
from utils.distributed import generalized_all_gather, all_reduce
from utils.registry_class import ENGINE, MODEL, DATASETS, EMBEDDER, AUTO_ENCODER, DISTRIBUTION, VISUAL, DIFFUSION, PRETRAIN


@ENGINE.register_function()
def train_t2v_share_diffusion_enterance(cfg_update,  **kwargs):
    for k, v in cfg_update.items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v

    if not 'MASTER_ADDR' in os.environ:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']= find_free_port()
    cfg.pmi_rank = int(os.getenv('RANK', 0)) # 0
    cfg.pmi_world_size = int(os.getenv('WORLD_SIZE', 1))
    setup_seed(cfg.seed)

    if cfg.debug:
        cfg.gpus_per_machine = 1
        cfg.world_size = 1
    else:
        cfg.gpus_per_machine = torch.cuda.device_count()
        cfg.world_size = cfg.pmi_world_size * cfg.gpus_per_machine

    if cfg.world_size == 1:
        worker(0, cfg)
    else:
        mp.spawn(worker, nprocs=cfg.gpus_per_machine, args=(cfg, ))
    return cfg

def save_lora_weight(
    model,
    path,
    target_replace_module
):
    weights = []
    for _up, _down in extract_lora_ups_down(
        model, target_replace_module=target_replace_module
    ):
        weights.append(_up.weight.to("cpu").to(torch.float16))
        weights.append(_down.weight.to("cpu").to(torch.float16))

    torch.save(weights, path)


def worker(gpu, cfg):
    '''
    Training worker for each gpu
    '''
    cfg.gpu = gpu
    cfg.rank = cfg.pmi_rank * cfg.gpus_per_machine + gpu

    shared_diffusion_steps = cfg.shared_diffusion_steps if cfg.shared_diffusion_steps \
    else None
    if not cfg.debug:
        torch.cuda.set_device(gpu)
        torch.backends.cudnn.benchmark = True
        dist.init_process_group(backend='nccl', world_size=cfg.world_size, rank=cfg.rank)

    # [Log] Save logging
    log_dir = generalized_all_gather(cfg.log_dir)[0]
    exp_name = osp.basename(cfg.cfg_file).split('.')[0]
    cfg.log_dir = osp.join(cfg.log_dir, exp_name)
    os.makedirs(cfg.log_dir, exist_ok=True)
    if cfg.rank == 0:
        log_file = osp.join(cfg.log_dir, 'log.txt')
        cfg.log_file = log_file
        reload(logging)
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(filename=log_file),
                logging.StreamHandler(stream=sys.stdout)])
        logging.info(cfg)
        logging.info(f'Save all the file in to dir {cfg.log_dir}')
        logging.info(f"Going into i2v_img_fullid_vidcom function on {gpu} gpu")


    config={
    "learning_rate": 0.003,
    }
    wandb.init(config=config,
            project="i2v_shared_diff",
            dir=str(log_dir),
            job_type="training",
            reinit=True)
    # [Diffusion]  build diffusion settings
    diffusion = DIFFUSION.build(cfg.Diffusion)

    # shared_settting
    if shared_diffusion_steps:
        shared_diffusion_steps_list = []
        for i in range(len(shared_diffusion_steps)-1):
            shared_diffusion_steps_list.append([shared_diffusion_steps[i], shared_diffusion_steps[i+1]])

    # [Dataset] imagedataset and videodataset
    len_frames = len(cfg.frame_lens)
    len_fps = len(cfg.sample_fps)
    cfg.max_frames = cfg.frame_lens[cfg.rank % len_frames]
    cfg.batch_size = cfg.batch_sizes[str(cfg.max_frames)]
    cfg.sample_fps = cfg.sample_fps[cfg.rank % len_fps]

    if cfg.rank == 0:
        logging.info(f'Currnt worker with max_frames={cfg.max_frames}, batch_size={cfg.batch_size}, sample_fps={cfg.sample_fps}')

    train_trans = data.Compose([
        data.CenterCropWide(size=cfg.resolution),
        data.ToTensor(),
        data.Normalize(mean=cfg.mean, std=cfg.std)])
    vit_trans = data.Compose([
        data.CenterCropWide(size=(cfg.resolution[0], cfg.resolution[0])) if cfg.resolution[0]>cfg.vit_resolution[0] else data.CenterCropWide(size=cfg.vit_resolution),
        data.Resize(cfg.vit_resolution),
        data.ToTensor(),
        data.Normalize(mean=cfg.vit_mean, std=cfg.vit_std)])

    if cfg.max_frames == 1:
        cfg.sample_fps = 1
        dataset = DATASETS.build(cfg.img_dataset, transforms=train_trans, vit_transforms=vit_trans)
    else:
        dataset = DATASETS.build(cfg.vid_dataset, sample_fps=cfg.sample_fps, transforms=train_trans, vit_transforms=vit_trans, max_frames=cfg.max_frames)

    sampler = DistributedSampler(dataset, num_replicas=cfg.world_size, rank=cfg.rank) if (cfg.world_size > 1 and not cfg.debug) else None
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        prefetch_factor=cfg.prefetch_factor)
    rank_iter = iter(dataloader)

    # [Model] embedder
    clip_encoder = EMBEDDER.build(cfg.embedder)
    clip_encoder.model.to(gpu)
    _, _, zero_y = clip_encoder(text="")
    _, _, zero_y_negative = clip_encoder(text=cfg.negative_prompt)
    zero_y, zero_y_negative = zero_y.detach(), zero_y_negative.detach()

    # [Model] auotoencoder
    autoencoder = AUTO_ENCODER.build(cfg.auto_encoder)
    autoencoder.eval() # freeze
    for param in autoencoder.parameters():
        param.requires_grad = False
    autoencoder.cuda()

    # [Model] UNet
    model = MODEL.build(cfg.UNet, zero_y=zero_y_negative)
    model = model.to(gpu)
    # turn off all of the gradients of unet, except for the trainable LoRA params.
    # if cfg.if_lora:
    #     unet_lora_config = LoraConfig(
    #         r=cfg.lora_rank,
    #         lora_alpha=cfg.lora_rank,
    #         init_lora_weights="gaussian",
    #         target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    #     )
    #     model.add_adapter(unet_lora_config)

    #     # if cfg.mixed_precision == "fp16":
    #     #     # only upcast trainable parameters (LoRA) into fp32
    #     #     cast_training_params(model, dtype=torch.float32)

    #     lora_layers = filter(lambda p: p.requires_grad, model.parameters())

    resume_step = 1
    model, resume_step = PRETRAIN.build(cfg.Pretrain, model=model)
    # del model.out
    torch.cuda.empty_cache()

    if cfg.use_ema:
        ema = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        ema = type(ema)([(k, ema[k].data.clone()) for k in list(ema.keys())[cfg.rank::cfg.world_size]])

    # optimizer
    # if cfg.lora:
    #     optimizer = optim.AdamW(params=lora_layers.parameters(),
    #         lr=cfg.lr, weight_decay=cfg.weight_decay)
    # else:
    if cfg.if_lora:
        model.requires_grad_(False)
        unet_lora_params, train_names = inject_trainable_lora_extended(
            model,target_replace_module=None)
        #
        # for _up, _down in extract_lora_ups_down(model):
        #     logging.info(f'Before training: Unet First Layer lora up{_up.weight.data}')
        #     logging.info(f'Before training: Unet First Layer lora down{_down.weight.data}')
        #     break
        optimizer = optim.AdamW(itertools.chain(*unet_lora_params),
                    lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        optimizer = optim.AdamW(params=model.parameters(),
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = amp.GradScaler(enabled=cfg.use_fp16)
    if cfg.use_fsdp:
        config = {}
        config['compute_dtype'] = torch.float32
        config['mixed_precision'] = True
        model = FSDP(model, **config)
    else:
        model = DistributedDataParallel(model, device_ids=[gpu]) if not cfg.debug else model.to(gpu)

    # scheduler
    scheduler = AnnealingLR(
        optimizer=optimizer,
        base_lr=cfg.lr,
        warmup_steps=cfg.warmup_steps,  # 10
        total_steps=cfg.num_steps,      # 200000
        decay_mode=cfg.decay_mode)      # 'cosine'

    # [Visual]
    viz_num = min(cfg.batch_size, 8)
    visual_func = VISUAL.build(
        cfg.visual_train,
        cfg_global=cfg,
        viz_num=viz_num,
        diffusion=diffusion,
        autoencoder=autoencoder)

    for step in range(resume_step, cfg.num_steps + 1):
        model.train()

        try:
            batch = next(rank_iter)
        except StopIteration:
            rank_iter = iter(dataloader)
            batch = next(rank_iter)

        batch = to_device(batch, gpu, non_blocking=True)
        ref_frame, _, video_data, captions, video_key = batch
        batch_size, frames_num, _, _, _ = video_data.shape
        video_data = rearrange(video_data, 'b f c h w -> (b f) c h w')

        fps_tensor =  torch.tensor([cfg.sample_fps] * batch_size, dtype=torch.long, device=gpu)
        video_data_list = torch.chunk(video_data, video_data.shape[0]//cfg.chunk_size,dim=0)
        with torch.no_grad():
            decode_data = []
            for chunk_data in video_data_list:
                latent_z = autoencoder.encode_firsr_stage(chunk_data, cfg.scale_factor).detach()
                decode_data.append(latent_z) # [B, 4, 32, 56]
            video_data = torch.cat(decode_data,dim=0)
            video_data = rearrange(video_data, '(b f) c h w -> b c f h w', b = batch_size) # [B, 4, 16, 32, 56]
        if cfg.if_lora:
            index = random.randint(0, 3)
        else:
            index = random.randint(0, 4)
        shart_timesteps = shared_diffusion_steps_list[index][0] * 1000
        end_timesteps = shared_diffusion_steps_list[index][1] * 1000
        # shart_timesteps,end_timesteps = map(float,)
        # opti_timesteps = getattr(cfg, 'opti_timesteps', cfg.Diffusion.schedule_param.num_timesteps)
        t_round = torch.randint(math.floor(shart_timesteps), math.floor(end_timesteps), (batch_size, ), dtype=torch.long, device=gpu) # 8

        # preprocess
        with torch.no_grad():
            _, _, y_words = clip_encoder(text=captions) # bs * 1 *1024 [B, 1, 1024]
            y_words_0 = y_words.clone()
            try:
                y_words[torch.rand(y_words.size(0)) < cfg.p_zero, :] = zero_y_negative
            except:
                pass

        # forward
        model_kwargs = {'y': y_words,  'fps': fps_tensor}
        if cfg.use_fsdp:
            loss = diffusion.loss(x0=video_data,
                t=t_round, model=model, model_kwargs=model_kwargs,
                use_div_loss=cfg.use_div_loss)
            loss = loss.mean()
        else:
            with amp.autocast(enabled=cfg.use_fp16):# cfg.use_div_loss: False    loss: [80]
                loss, double_frame_flag = diffusion.shared_diff_loss(
                        x0=video_data,
                        t=t_round,
                        model=model,
                        model_kwargs=model_kwargs,
                        use_div_loss=cfg.use_div_loss,
                        index =index) if shared_diffusion_steps else \
                        diffusion.loss(
                        x0=video_data,
                        t=t_round,
                        model=model,
                        model_kwargs=model_kwargs,
                        use_div_loss=cfg.use_div_loss) # cfg.use_div_loss: False    loss: [80]
                loss = loss.mean()

        # backward
        if cfg.use_fsdp:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 0.05)
            optimizer.step()
        else:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            # for name, param in model.named_parameters():
            #     if param.grad is None:
            #         print(name)
            scaler.step(optimizer)
            scaler.update()

        if not cfg.use_fsdp:
            scheduler.step()

        # ema update
        if cfg.use_ema:
            temp_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            for k, v in ema.items():
                v.copy_(temp_state_dict[k].lerp(v, cfg.ema_decay))

        all_reduce(loss)
        loss = loss / cfg.world_size

        if cfg.rank == 0 and step % cfg.log_interval == 0: # cfg.log_interval: 100
            logging.info(f'Step: {step}/{cfg.num_steps} Loss: {loss.item():.3f} scale: {scaler.get_scale():.1f} LR: {scheduler.get_lr():.7f} Double_frame_flag: {double_frame_flag}')
            wandb.log({
                "Loss":loss.item(),
                "scale":scaler.get_scale(),
                "LR":scheduler.get_lr()})
            if double_frame_flag:
                wandb.log({"double_Loss":loss.item()})
            else:
                wandb.log({'non_double_Loss':loss.item()})
            # for _up, _down in extract_lora_ups_down(model):
            #     logging.info(f'First Unet Layers Up Weight is now :{_up.weight.data}')
            #     logging.info(f'First Unet Layers Down Weight is now :{_down.weight.data}')
            #     break

        # Visualization
        # if step == resume_step or step == cfg.num_steps or step % cfg.viz_interval == 0:
        #     with torch.no_grad():
        #         try:
        #             visual_kwards = [
        #                 {
        #                     'y': y_words_0[:viz_num],
        #                     'fps': fps_tensor[:viz_num],
        #                 },
        #                 {
        #                     'y': zero_y_negative.repeat(y_words_0.size(0), 1, 1),
        #                     'fps': fps_tensor[:viz_num],
        #                 }
        #             ]
        #             input_kwards = {
        #                 'model': model, 'video_data': video_data[:viz_num], 'step': step,
        #                 'ref_frame': ref_frame[:viz_num], 'captions': captions[:viz_num]}
        #             visual_func.run(visual_kwards=visual_kwards, **input_kwards)
        #         except Exception as e:
        #             logging.info(f'Save videos with exception {e}')

        # Save checkpoint
        if step == cfg.num_steps or step % cfg.save_ckp_interval == 0 or step == resume_step:
            os.makedirs(osp.join(cfg.log_dir, 'checkpoints'), exist_ok=True)
            if cfg.use_ema:
                local_ema_model_path = osp.join(cfg.log_dir, f'checkpoints/ema_{step:08d}_rank{cfg.rank:04d}.pth')
                save_dict = {
                    'state_dict': ema.module.state_dict() if hasattr(ema, 'module') else ema,
                    'step': step}
                torch.save(save_dict, local_ema_model_path)
                if cfg.rank == 0:
                    logging.info(f'Begin to Save ema model to {local_ema_model_path}')
            if cfg.rank == 0:
                local_model_path = osp.join(cfg.log_dir, f'checkpoints/non_ema_{step:08d}.pth')
                logging.info(f'Begin to Save model to {local_model_path}')
                if cfg.if_lora:
                    save_lora_weight(model, local_model_path,target_replace_module= None)
                else:
                    save_dict = {
                        'state_dict': model.module.state_dict() if not cfg.debug else model.state_dict(),
                        'step': step}
                    torch.save(save_dict, local_model_path)
                logging.info(f'Save model to {local_model_path}')

    if cfg.rank == 0:
        logging.info('Congratulations! The training is completed!')
    wandb.finish()
    # synchronize to finish some processes
    if not cfg.debug:
        torch.cuda.synchronize()
        dist.barrier()

