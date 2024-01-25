from typing import Literal, Union, Dict
import os
import shutil
import fire
import sys
import copy

from diffusers import StableDiffusionPipeline
from safetensors.torch import safe_open, save_file
from utils.config import Config
from tools.modules.config import cfg
from utils.registry_class import INFER_ENGINE, MODEL, EMBEDDER, AUTO_ENCODER, DIFFUSION,PRETRAIN

# sys.path.append("../")
import torch
from lora_diffusion import *
# from lora_diffusion.lora import (
#     tune_lora_scale,
#     patch_pipe,
#     collapse_lora,
#     monkeypatch_remove_lora,
#     monkeypatch_or_replace_lora_extended,
# )


def _text_lora_path(path: str) -> str:
    assert path.endswith(".pt"), "Only .pt files are supported"
    return ".".join(path.split(".")[:-1] + ["text_encoder", "pt"])


def add(
    path_1: str,
    path_2: str,
    output_path: str,
    alpha_1: float = 0.5,
    alpha_2: float = 0.5,
    with_text_lora: bool = False,
):
    print(
        f"Merging UNET/CLIP from {path_1} with LoRA from {path_2} to {output_path}. Merging ratio : {alpha_1}."
    )
    print("LoRA : Patching Unet")
    u_model = path_1
    copied_model = copy.deepcopy(u_model)
    lora = torch.load(path_2)
    monkeypatch_or_replace_lora_extended(
        copied_model,
        lora,
        target_replace_module=None,
    )

    collapse_lora(copied_model, alpha_1)

    monkeypatch_remove_lora(copied_model)
    # torch.save(u_model, output_path)

def main():
    cfg_update = Config(load=True)
    cfg_update=cfg_update.cfg_dict
    for k, v in cfg_update.items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v

    cfg.pmi_rank = int(os.getenv('RANK', 0)) # 0
    cfg.pmi_world_size = int(os.getenv('WORLD_SIZE', 1))
    # setup_seed(cfg.seed)

    clip_encoder = EMBEDDER.build(cfg.embedder)
    clip_encoder.model.to(0)
    _, _, zero_y = clip_encoder(text="")
    _, _, zero_y_negative = clip_encoder(text=cfg.negative_prompt)
    zero_y, zero_y_negative = zero_y.detach(), zero_y_negative.detach()


    model = MODEL.build(cfg.UNet, zero_y=zero_y_negative)
    model, resume_step = PRETRAIN.build(cfg.Pretrain, model=model)
    model = model.to(0)

    add(model,
        '../i2vgen-xl/workspace/experiments/t2v_train_shared_diff/checkpoints/non_ema_00002000.pt',
        './lora_model_0000000.pt',1)


if __name__ == "__main__":
    main()

