# pylint: disable=E1101,C0415,W0718,R0801
# scripts/train_stage2.py
"""
This is the main training script for stage 2 of the project. 
It imports necessary packages, defines necessary classes and functions, and trains the model using the provided configuration.

The script includes the following classes and functions:

1. Net: A PyTorch model that takes noisy latents, timesteps, reference image latents, face embeddings, 
   and face masks as input and returns the denoised latents.
2. get_attention_mask: A function that rearranges the mask tensors to the required format.
3. get_noise_scheduler: A function that creates and returns the noise schedulers for training and validation.
4. process_audio_emb: A function that processes the audio embeddings to concatenate with other tensors.
5. log_validation: A function that logs the validation information using the given VAE, image encoder, 
   network, scheduler, accelerator, width, height, and configuration.
6. train_stage2_process: A function that processes the training stage 2 using the given configuration.
7. load_config: A function that loads the configuration file from the given path.

The script also includes the necessary imports and a brief description of the purpose of the file.
"""

import argparse
import copy
import logging
import math
import os
import random
import time
import warnings
from datetime import datetime
from typing import List, Tuple
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目的根目录（假设scripts文件夹与hallo文件夹同级）
project_root = os.path.abspath(os.path.join(current_dir, '..'))
# 将项目根目录添加到sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import diffusers
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange, repeat
from omegaconf import OmegaConf
from torch import nn
from tqdm.auto import tqdm
from accelerate.utils import ProjectConfiguration
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path

from hallo.animate.face_animate_yu import FaceAnimatePipeline
from hallo.datasets.audio_processor import AudioProcessor
from hallo.datasets.image_processor import ImageProcessor
# from hallo.datasets.talk_video_svd import TalkingVideoDataset
from hallo.datasets.talk_video_sonic_whisper import TalkingVideoDataset
# from hallo.models.audio_proj import AudioProjModel
from hallo.models.audio_proj_whisper import AudioProjModel

from hallo.models.audio_proj_sonic import AudioProjModel as AudioProjModel_sonic
from hallo.models.audio_to_bucket import Audio2bucketModel

from hallo.utils.util import (compute_snr, delete_additional_ckpt,
                              import_filename, init_output_dir,
                              load_checkpoint, save_checkpoint,
                              seed_everything, tensor_to_video)

import diffusers
from diffusers import StableVideoDiffusionPipeline
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from hallo.animate.face_animate_static import StaticPipeline
from hallo.datasets.mask_image import FaceMaskDataset
from hallo.models.face_locator import FaceLocator
from hallo.models.image_proj import ImageProjModel
from hallo.models.diffuser.unet_spatio_temporal_condition_audio import UNetSpatioTemporalConditionModel
from hallo.models.unet_2d_condition import UNet2DConditionModel
from hallo.models.unet_3d import UNet3DConditionModel
from hallo.utils.util import (compute_snr, delete_additional_ckpt,
                              import_filename, init_output_dir,
                              load_checkpoint, move_final_checkpoint,
                              save_checkpoint, seed_everything)

from hallo.models.whisper_local.audio2feature import load_audio_model

from packaging import version
from transformers import CLIPImageProcessor
from transformers import WhisperModel


warnings.filterwarnings("ignore")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")

class Net(nn.Module):
    """
    The Net class defines a neural network model that combines a reference UNet2DConditionModel,
    a denoising UNet3DConditionModel, a face locator, and other components to animate a face in a static image.

    Args:
        reference_unet (UNet2DConditionModel): The reference UNet2DConditionModel used for face animation.
        denoising_unet (UNet3DConditionModel): The denoising UNet3DConditionModel used for face animation.
        face_locator (FaceLocator): The face locator model used for face animation.
        reference_control_writer: The reference control writer component.
        reference_control_reader: The reference control reader component.
        imageproj: The image projection model.
        audioproj: The audio projection model.

    Forward method:
        noisy_latents (torch.Tensor): The noisy latents tensor.
        timesteps (torch.Tensor): The timesteps tensor.
        ref_image_latents (torch.Tensor): The reference image latents tensor.
        face_emb (torch.Tensor): The face embeddings tensor.
        audio_emb (torch.Tensor): The audio embeddings tensor.
        mask (torch.Tensor): Hard face mask for face locator.
        full_mask (torch.Tensor): Pose Mask.
        face_mask (torch.Tensor): Face Mask
        lip_mask (torch.Tensor): Lip Mask
        uncond_img_fwd (bool): A flag indicating whether to perform reference image unconditional forward pass.
        uncond_audio_fwd (bool): A flag indicating whether to perform audio unconditional forward pass.

    Returns:
        torch.Tensor: The output tensor of the neural network model.
    """
    def __init__(
        self,
        denoising_unet: UNetSpatioTemporalConditionModel,
        face_locator: FaceLocator,
        imageproj,
        audioproj,
        audio2bucket,
        audio2token,
    ):
        super().__init__()
        self.denoising_unet = denoising_unet
        self.face_locator = face_locator
        self.imageproj = imageproj
        self.audioproj = audioproj
        self.audio2bucket = audio2bucket
        self.audio2token = audio2token

    # model_pred = net(
    #     noisy_latents=noisy_latents,
    #     timesteps=timesteps,
    #     ref_image_latents=ref_image_latents,
    #     face_emb=image_prompt_embeds,
    #     audio_emb=batch["audio_tensor"].to(
    #         dtype=weight_dtype),
    #     uncond_img_fwd=uncond_img_fwd,
    #     uncond_audio_fwd=uncond_audio_fwd
    # )
    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        face_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        added_time_ids,
        uncond_img_fwd: bool = False,
        uncond_audio_fwd: bool = False,
    ):
        """
        simple docstring to prevent pylint error
        """
        face_emb = self.imageproj(face_emb)
        # mask = mask.to(device="cuda")
        # mask_feature = self.face_locator(mask)
        audio_emb = audio_emb.to(
            device=self.audioproj.device, dtype=self.audioproj.dtype)
        audio_emb = self.audioproj(audio_emb)

        if uncond_audio_fwd:
            audio_emb = torch.zeros_like(audio_emb).to(
                device=audio_emb.device, dtype=audio_emb.dtype
            )

        # print("**0101\n\n face_emb ", face_emb.shape)
        # print("**0101\n\n audio_emb ", audio_emb.shape)
        #  face_emb  torch.Size([2, 4, 1024])
        #  audio_emb  torch.Size([2, 14, 32, 768])
        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=face_emb,
            audio_embedding=audio_emb,
            added_time_ids=added_time_ids,
        ).sample

        return model_pred


def get_attention_mask(mask: torch.Tensor, weight_dtype: torch.dtype) -> torch.Tensor:
    """
    Rearrange the mask tensors to the required format.

    Args:
        mask (torch.Tensor): The input mask tensor.
        weight_dtype (torch.dtype): The data type for the mask tensor.

    Returns:
        torch.Tensor: The rearranged mask tensor.
    """
    if isinstance(mask, List):
        _mask = []
        for m in mask:
            _mask.append(
                rearrange(m, "b f 1 h w -> (b f) (h w)").to(weight_dtype))
        return _mask
    mask = rearrange(mask, "b f 1 h w -> (b f) (h w)").to(weight_dtype)
    return mask


def get_noise_scheduler(cfg: argparse.Namespace) -> Tuple[DDIMScheduler, DDIMScheduler]:
    """
    Create noise scheduler for training.

    Args:
        cfg (argparse.Namespace): Configuration object.

    Returns:
        Tuple[DDIMScheduler, DDIMScheduler]: Train noise scheduler and validation noise scheduler.
    """

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)

    return train_noise_scheduler, val_noise_scheduler


def process_audio_emb(audio_emb: torch.Tensor) -> torch.Tensor:
    """
    Process the audio embedding to concatenate with other tensors.

    Parameters:
        audio_emb (torch.Tensor): The audio embedding tensor to process.

    Returns:
        concatenated_tensors (List[torch.Tensor]): The concatenated tensor list.
    """
    concatenated_tensors = []

    for i in range(audio_emb.shape[0]):
        vectors_to_concat = [
            audio_emb[max(min(i + j, audio_emb.shape[0] - 1), 0)]for j in range(-2, 3)]
        concatenated_tensors.append(torch.stack(vectors_to_concat, dim=0))

    audio_emb = torch.stack(concatenated_tensors, dim=0)

    return audio_emb

def save_image_batch(image_tensor, save_path):
    image_tensor = (image_tensor + 1) / 2

    os.makedirs(save_path, exist_ok=True)

    for i in range(image_tensor.shape[0]):
        img_tensor = image_tensor[i]
        
        img_array = img_tensor.permute(1, 2, 0).cpu().numpy()
        
        img_array = (img_array * 255).astype(np.uint8)
        
        image = Image.fromarray(img_array)
        image.save(os.path.join(save_path, f'motion_frame_{i}.png'))

def log_validation(
    accelerator: Accelerator,
    vae: AutoencoderKL,
    net: Net,
    scheduler: DDIMScheduler,
    width: int,
    height: int,
    clip_length: int = 24,
    generator: torch.Generator = None,
    cfg: dict = None,
    save_dir: str = None,
    global_step: int = 0,
    times: int = None,
    face_analysis_model_path: str = "",
) -> None:
    """
    Log validation video during the training process.

    Args:
        accelerator (Accelerator): The accelerator for distributed training.
        vae (AutoencoderKL): The autoencoder model.
        net (Net): The main neural network model.
        scheduler (DDIMScheduler): The scheduler for noise.
        width (int): The width of the input images.
        height (int): The height of the input images.
        clip_length (int): The length of the video clips. Defaults to 24.
        generator (torch.Generator): The random number generator. Defaults to None.
        cfg (dict): The configuration dictionary. Defaults to None.
        save_dir (str): The directory to save validation results. Defaults to None.
        global_step (int): The current global step in training. Defaults to 0.
        times (int): The number of inference times. Defaults to None.
        face_analysis_model_path (str): The path to the face analysis model. Defaults to "".

    Returns:
        torch.Tensor: The tensor result of the validation.
    """
    ori_net = accelerator.unwrap_model(net)
    denoising_unet = ori_net.denoising_unet
    face_locator = ori_net.face_locator
    imageproj = ori_net.imageproj
    audioproj = ori_net.audioproj

    generator = torch.manual_seed(42)
    tmp_denoising_unet = copy.deepcopy(denoising_unet)


    pipeline = FaceAnimatePipeline(
        vae=vae,
        denoising_unet=tmp_denoising_unet,
        face_locator=face_locator,
        image_proj=imageproj,
        scheduler=scheduler,
    )
    pipeline = pipeline.to("cuda")

    image_processor = ImageProcessor((width, height), face_analysis_model_path)
    audio_processor = AudioProcessor(
        cfg.data.sample_rate,
        cfg.data.fps,
        cfg.wav2vec_config.model_path,
        cfg.wav2vec_config.features == "last",
        os.path.dirname(cfg.audio_separator.model_path),
        os.path.basename(cfg.audio_separator.model_path),
        os.path.join(save_dir, '.cache', "audio_preprocess")
    )

    for idx, ref_img_path in enumerate(cfg.ref_img_path):
        audio_path = cfg.audio_path[idx]
        source_image_pixels, \
        source_image_face_region, \
        source_image_face_emb, \
        source_image_full_mask, \
        source_image_face_mask, \
        source_image_lip_mask = image_processor.preprocess(
            ref_img_path, os.path.join(save_dir, '.cache'), cfg.face_expand_ratio)


        audio_emb, audio_length = audio_processor.preprocess(
            audio_path, clip_length)
        # print("\n\n\naudio length is {}".format(audio_length))
        audio_emb = process_audio_emb(audio_emb)

        source_image_pixels = source_image_pixels.unsqueeze(0)
        source_image_face_region = source_image_face_region.unsqueeze(0)
        source_image_face_emb = source_image_face_emb.reshape(1, -1)
        source_image_face_emb = torch.tensor(source_image_face_emb)

        source_image_full_mask = [
            (mask.repeat(clip_length, 1))
            for mask in source_image_full_mask
        ]
        source_image_face_mask = [
            (mask.repeat(clip_length, 1))
            for mask in source_image_face_mask
        ]
        source_image_lip_mask = [
            (mask.repeat(clip_length, 1))
            for mask in source_image_lip_mask
        ]

        times = audio_emb.shape[0] // clip_length
        tensor_result = []
        generator = torch.manual_seed(42)


        for t in range(times):
            print(f"[{t+1}/{times}]")

            if len(tensor_result) == 0:
                # The first iteration
                motion_zeros = source_image_pixels.repeat(
                    cfg.data.n_motion_frames, 1, 1, 1)
                motion_zeros = motion_zeros.to(
                    dtype=source_image_pixels.dtype, device=source_image_pixels.device)
                pixel_values_ref_img = torch.cat(
                    [source_image_pixels, motion_zeros], dim=0)  # concat the ref image and the first motion frames
            elif len(tensor_result) == 1:
                motion_frames = tensor_result[-1][0]
                motion_frames = motion_frames.permute(1, 0, 2, 3)
                motion_frames = motion_frames[np.array([-13, -11,-9,-7,-5, -4,-3,-2,-1])]
                motion_frames = motion_frames * 2.0 - 1.0
                motion_frames_to_pad = source_image_pixels.repeat(3, 1, 1, 1)
                motion_frames = torch.cat([motion_frames_to_pad, motion_frames], dim=0)
                pixel_values_ref_img = torch.cat(
                    [source_image_pixels, motion_frames], dim=0)  # concat the ref image and the motion frames
            else:
                # motion_frames = tensor_result[-1][0]
                # motion_frames = motion_frames.permute(1, 0, 2, 3)
                # motion_frames = motion_frames[0 - cfg.data.n_motion_frames:]
                motion_frames_1 = tensor_result[-1][0]
                motion_frames_1 = motion_frames_1.permute(1, 0, 2, 3)
                motion_frames_2 = tensor_result[-2][0]
                motion_frames_2 = motion_frames_2.permute(1, 0, 2, 3)
                motion_frames = torch.cat([motion_frames_2, motion_frames_1], dim=0)
                motion_frames = motion_frames[np.array([-25,-21,-17,-13, -11,-9,-7,-5, -4,-3,-2,-1])]
                motion_frames = motion_frames * 2.0 - 1.0
                motion_frames = motion_frames.to(
                    dtype=source_image_pixels.dtype, device=source_image_pixels.device)
                pixel_values_ref_img = torch.cat(
                    [source_image_pixels, motion_frames], dim=0)  # concat the ref image and the motion frames

            pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)


            # here we do patch drop
            pixel_motion_values = pixel_values_ref_img[:, 1:]

            if cfg.use_mask:

                b, f, c, h, w = pixel_motion_values.shape
                rand_mask = torch.rand(h, w)
                mask = rand_mask > cfg.mask_rate
                mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)  
                mask = mask.expand(b, f, c, h, w)  

                face_mask = source_image_face_region.repeat(f, 1, 1, 1).unsqueeze(0)
                assert face_mask.shape == mask.shape
                mask = mask | face_mask.bool()

                pixel_motion_values = pixel_motion_values * mask
                pixel_values_ref_img[:, 1:] = pixel_motion_values

            assert pixel_motion_values.shape[0] == 1

            # patch drop done



            audio_tensor = audio_emb[t * clip_length: min((t + 1) * clip_length, audio_emb.shape[0])
                           ]
            audio_tensor = audio_tensor.unsqueeze(0)
            audio_tensor = audio_tensor.to(
                device=audioproj.device, dtype=audioproj.dtype)
            audio_tensor = audioproj(audio_tensor)

            pipeline_output = pipeline(
                ref_image=pixel_values_ref_img,
                audio_tensor=audio_tensor,
                face_emb=source_image_face_emb,
                face_mask=source_image_face_region,
                pixel_values_full_mask=source_image_full_mask,
                pixel_values_face_mask=source_image_face_mask,
                pixel_values_lip_mask=source_image_lip_mask,
                width=cfg.data.train_width,
                height=cfg.data.train_height,
                video_length=clip_length,
                num_inference_steps=cfg.inference_steps,
                guidance_scale=cfg.cfg_scale,
                generator=generator
            )

            tensor_result.append(pipeline_output.videos)

        tensor_result = torch.cat(tensor_result, dim=2)
        tensor_result = tensor_result.squeeze(0)
        tensor_result = tensor_result[:, :audio_length]
        audio_name = os.path.basename(audio_path).split('.')[0]
        ref_name = os.path.basename(ref_img_path).split('.')[0]
        output_file = os.path.join(save_dir,f"{global_step}_{ref_name}_{audio_name}.mp4")
        # save the result after all iteration
        print("***\n\n tensor to video , output_file ", output_file , ref_name, audio_name, audio_path)
        tensor_to_video(tensor_result, output_file, audio_path)


    # clean up
    del tmp_denoising_unet
    del pipeline
    del image_processor
    del audio_processor
    torch.cuda.empty_cache()

    return tensor_result

def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()


def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    latents = latents * vae.config.scaling_factor


    return latents




def train_stage2_process(cfg: argparse.Namespace) -> None:
    """
    Trains the model using the given configuration (cfg).

    Args:
        cfg (dict): The configuration dictionary containing the parameters for training.

    Notes:
        - This function trains the model using the given configuration.
        - It initializes the necessary components for training, such as the pipeline, optimizer, and scheduler.
        - The training progress is logged and tracked using the accelerator.
        - The trained model is saved after the training is completed.
    """

    print(f"Current working directory: {os.getcwd()}")

    # 可选：手动创建日志目录
    log_dir = "log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Created log directory: {log_dir}")

    # 配置项目和日志目录
    config = ProjectConfiguration(project_dir=".", logging_dir=log_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with="tensorboard",
        project_config=config
    )

    # if accelerator.is_main_process:
    #     log_dir = accelerator.get_tracker("tensorboard").log_dir
    #     print(f"**tensorboard \n\n TensorBoard logs are saved in: {log_dir}")
    # # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    # create output dir for training
    exp_name = cfg.exp_name
    save_dir = f"{cfg.output_dir}/{exp_name}"
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    module_dir = os.path.join(save_dir, "modules")
    validation_dir = os.path.join(save_dir, "validation")
    if accelerator.is_main_process:
        init_output_dir([save_dir, checkpoint_dir, module_dir, validation_dir])

    accelerator.wait_for_everyone()

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )

    # print("cfg.svd.unet_path" , cfg.svd.unet_path)
    # ./pretrained_models/svd_xt_1_1.safetensors

    unet = UNetSpatioTemporalConditionModel(
        sample_size = 96
    )

    from safetensors.torch import load_file
    pretrained_path = cfg.svd.pretrain + "/unet/diffusion_pytorch_model.fp16.safetensors"
    state_dict = load_file(pretrained_path)
    m,u = unet.load_state_dict(state_dict, strict=False)
    print("** missing keys : \n\n", m)
    print("** unexpected keys : \n\n", u)

    imageproj = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=512,
        clip_extra_context_tokens=4,
    ).to(device="cuda", dtype=weight_dtype)
    face_locator = FaceLocator(
        conditioning_embedding_channels=320,
    ).to(device="cuda", dtype=weight_dtype)
    # audioproj = AudioProjModel(
    #     seq_len=5,
    #     blocks=12,
    #     channels=768,
    #     intermediate_dim=512,
    #     output_dim=768,
    #     context_tokens=32,
    # ).to(device="cuda", dtype=weight_dtype)
    audioproj = AudioProjModel(
        window=50,
        channels=384,
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
    ).to(device="cuda", dtype=weight_dtype)

    whisper = WhisperModel.from_pretrained("/yuch_ws/DH/hallo2/pretrained_models/whisper-tiny/").to(device="cuda").eval()
    whisper.requires_grad_(False)

    audio2token = AudioProjModel_sonic(seq_len=10, blocks=5, channels=384, intermediate_dim=1024, output_dim=1024,
                                 context_tokens=32).to(device="cuda")
    audio2bucket = Audio2bucketModel(seq_len=50, blocks=1, channels=384, clip_channels=1024, intermediate_dim=1024,
                                     output_dim=1, context_tokens=2).to(device="cuda")


    feature_extractor = CLIPImageProcessor.from_pretrained(
        cfg.svd.pretrain, subfolder="feature_extractor"
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        cfg.svd.pretrain, subfolder="image_encoder",
        variant="fp16"
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        cfg.svd.pretrain, subfolder="vae")

    print("load done ")

    # Freeze
    imageproj.requires_grad_(False)
    face_locator.requires_grad_(False)
    audioproj.requires_grad_(False)
    audio2bucket.requires_grad_(False)
    audio2token.requires_grad_(False)

    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)


    image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # Set motion module learnable
    # trainable_modules = cfg.trainable_para
    # for name, module in denoising_unet.named_modules():
    #     if any(trainable_mod in name for trainable_mod in trainable_modules):
    #         for params in module.parameters():
    #             params.requires_grad_(True)



    # reference_control_writer = ReferenceAttentionControl(
    #     reference_unet,
    #     do_classifier_free_guidance=False,
    #     mode="write",
    #     fusion_blocks="full",
    # )
    # reference_control_reader = ReferenceAttentionControl(
    #     denoising_unet,
    #     do_classifier_free_guidance=False,
    #     mode="read",
    #     fusion_blocks="full",
    # )

    net = Net(
        unet,
        face_locator,
        imageproj,
        audioproj,
        audio2token,
        audio2bucket
    ).to(dtype=weight_dtype)

    # m,u = net.load_state_dict(
    #     torch.load(
    #         os.path.join(cfg.audio_ckpt_dir, "net-3000.pth"),
    #         map_location="cpu",
    #     ),
    #     strict=False
    # )
    #
    # logger.info(f"missing key: {m}")
    # logger.info(f"unexcepted key: {u}")

    # get noise scheduler
    train_noise_scheduler, val_noise_scheduler = get_noise_scheduler(cfg)

    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if cfg.solver.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.data.train_bs
            * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate

    # Initialize the optimizer
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            ) from exc
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # trainable_params = list(
    #     filter(lambda p: p.requires_grad, net.parameters()))
    # logger.info(f"Total trainable params {len(trainable_params)}")


    trainable_params = []
    train_param_names= []
    for name, param in unet.named_parameters():
        if "audio_modules" in name:
            trainable_params.append(param)
            train_param_names.append(name)
            param.requires_grad = True
    for name, param in audioproj.named_parameters():
            trainable_params.append(param)
            train_param_names.append(name)
            param.requires_grad = True
    for name, param in imageproj.named_parameters():
            trainable_params.append(param)
            train_param_names.append(name)
            param.requires_grad = True

    print("**Trainable params: " , train_param_names)
    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )
    # if accelerator.is_main_process:
    #     rec_txt1 = open('params_freeze.txt', 'w')
    #     rec_txt2 = open('params_train.txt', 'w')
    #     for name, para in unet.named_parameters():
    #         if para.requires_grad is False:
    #             rec_txt1.write(f'{name}\n')
    #         else:
    #             rec_txt2.write(f'{name}\n')
    #     rec_txt1.close()
    #     rec_txt2.close()

    # Scheduler
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,
    )

    # get data loader
    train_dataset = TalkingVideoDataset(
        img_size=(cfg.data.train_width, cfg.data.train_height),
        sample_rate=cfg.data.sample_rate,
        n_sample_frames=cfg.data.n_sample_frames,
        n_motion_frames=cfg.data.n_motion_frames,
        audio_margin=cfg.data.audio_margin,
        data_meta_paths=cfg.data.train_meta_paths,
        wav2vec_cfg=cfg.wav2vec_config,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.data.train_bs, shuffle=True, num_workers=16
    )

    # Prepare everything with our `accelerator`.
    (
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(
            exp_name,

            init_kwargs={"tensorboard": {
                    # "log_dir": save_dir,
                    "flush_secs": 60

                }
            },
        )
        # 打印日志目录
        print(f"TensorBoard logs are saved in: {config.logging_dir}")

        # 查找并打印实际的日志位置（如果需要）
        def find_tensorboard_logs(base_dir='.'):
            for root, dirs, files in os.walk(base_dir):
                if any(file.startswith("events.out.tfevents") for file in files):
                    return os.path.abspath(root)
            return None

        log_dir_found = find_tensorboard_logs(config.logging_dir)
        if log_dir_found:
            print(f"Found TensorBoard logs in: {log_dir_found}")
        else:
            print("No TensorBoard logs found yet.")

        logger.info(f"save config to {save_dir}")
        OmegaConf.save(
            cfg, os.path.join(save_dir, "config.yaml")
        )

    # Train!
    total_batch_size = (
        cfg.data.train_bs
        * accelerator.num_processes
        * cfg.solver.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.data.train_bs}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.solver.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        logger.info(f"Loading checkpoint from {checkpoint_dir}")
        global_step = load_checkpoint(cfg, checkpoint_dir, accelerator)
        first_epoch = global_step // num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    def _get_add_time_ids(
        fps,
        motion_bucket_id,
        noise_aug_strength,
        dtype,
        batch_size,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = unet.config.addition_time_embed_dim * \
            len(add_time_ids)
        expected_add_embed_dim = unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size, 1)
        return add_time_ids

    for _ in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        t_data_start = time.time()
        for idx, batch in enumerate(train_dataloader):
            t_data = time.time() - t_data_start
            with accelerator.accumulate(net):
                # Convert videos to latent space
                pixel_values_vid = batch["pixel_values_vid"].to(weight_dtype)
                #print("**debug 12 29 \n\n  pixel_values_vid shape is ", pixel_values_vid.shape)
                # pixel_values_vid shape is  torch.Size([4, 14, 3, 512, 512])
                #


                conditional_pixel_values = batch["pixel_values_ref_img"].to(weight_dtype)
                # reference_pixel_values = batch["pixel_values_ref_img"].to(weight_dtype)
                #
                # conditional_pixel_values = pixel_values_vid[:, 0:1, :, :, :]

                # print("\n pixel_values_vid shape is ", pixel_values_vid.shape , torch.max(pixel_values_vid), torch.min(pixel_values_vid) )
                # print("condi shape : ", conditional_pixel_values.shape, torch.max(conditional_pixel_values) , torch.min(conditional_pixel_values))
                # ([2, 25, 3, 512, 512])   -1 , 1
                # ([2, 1, 3, 512, 512])   -1 ,1

                # print("**debug 12 29 \n\n  conditional_pixel_values shape is ", conditional_pixel_values.shape)
                # conditional_pixel_values shape is  torch.Size([4, 1, 3, 512, 512])
                # print("**debug 12 29 \n\n  vae scaling factor ", vae.config.scaling_factor )
                # vae scaling factor  0.18215

                # 新增CLIP & audio embd

                clip_img = batch['clip_images']
                image_embeds = image_encoder(
                    clip_img
                ).image_embeds
                image_embeds = image_embeds.repeat_interleave(25, dim=0)

                audio_clips = batch["audio_clips"]
                audio_clips_for_bucket = batch["audio_clips_for_bucket"]

                # audio_feature = batch['audio_feature']
                # audio_len = batch['audio_len']
                print("clip_img.shape",clip_img.shape)
                print("image_embeds.shape",image_embeds.shape)
                print("audio_clips.shape",audio_clips.shape)
                print("audio_clips_for_bucket.shape", audio_clips_for_bucket.shape)
                print("aaudio2bucket dtype is ", audio2bucket.dtype)
                print("aaudio2token dtype is ", audio2token.dtype)
                image_embeds=image_embeds.to(dtype=audio2bucket.dtype)
                audio_clips_for_bucket=audio_clips_for_bucket.to(dtype=audio2bucket.dtype)
                motion_buckets = audio2bucket(audio_clips_for_bucket, image_embeds)

                audio_clips = audio_clips.to(dtype=audio2bucket.dtype)
                audio_zeros = torch.zeros_like(audio_clips)
                cond_audio_clip = audio2token(audio_clips).squeeze(0)
                uncond_audio_clip = audio2token(audio_zeros).squeeze(0)

                print("motion buckets shape : ", motion_buckets.shape)
                print("cond_audio_clip shape : ", cond_audio_clip.shape)
                print("uncond_audioclip shape :" , uncond_audio_clip.shape)

                # for i in tqdm(range(audio_len // step)):
                #     audio_clip = audio_prompts[idx]
                #     audio_clip_for_bucket = audio_clips_for_bucket[idx]
                #     motion_bucket = audio2bucket(audio_clip_for_bucket, image_embeds)
                #     motion_bucket = motion_bucket * 16 + 16
                #     motion_buckets.append(motion_bucket[0])
                #
                #     cond_audio_clip = audio2token(audio_clip).squeeze(0)
                #     uncond_audio_clip = audio2token(torch.zeros_like(audio_clip)).squeeze(0)
                #
                #     print("cond_audio_clip shape is : ", cond_audio_clip.shape)
                #     # ref_tensor_list.append(ref_img[0])
                #     audio_tensor_list.append(cond_audio_clip[0])
                #     print("cond_audio_clip[0] shape is : ", cond_audio_clip[0].shape)
                #     uncond_audio_tensor_list.append(uncond_audio_clip[0])







                # 新增结束



                latents = tensor_to_vae_latent(pixel_values_vid , vae)


                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # 向噪声中添加一个小的偏移量来增加训练过程中的多样性或稳定性。
                # if cfg.noise_offset > 0:
                #     noise += cfg.noise_offset * torch.randn(
                #         (latents.shape[0], latents.shape[1], 1, 1, 1),
                #         device=latents.device,
                #     )
                # 生成对数正态分布的条件噪声强度
                cond_sigmas = rand_log_normal(shape=[bsz, ], loc=-3.0, scale=0.5).to(latents)
                noise_aug_strength = cond_sigmas[0]  # TODO: support batch > 1
                # print("**debug 12 29 \n\n  cond_sigmasis ", cond_sigmas)
                # print("**debug 12 29 \n\n  vae scaling factor ", vae.config.scaling_factor)
                cond_sigmas = cond_sigmas[:, None, None, None, None]
                # print("**debug 12 29 \n\n  cond_sigmasis ", cond_sigmas.shape)
                conditional_pixel_values = \
                    torch.randn_like(conditional_pixel_values) * cond_sigmas + conditional_pixel_values
                conditional_latents = tensor_to_vae_latent(conditional_pixel_values, vae)
                # print("**debug 12 29 \n\n  conditional_latents  ", conditional_latents .shape)
                conditional_latents = conditional_latents[:, 0, :, :, :]
                # print("**debug 12 29 \n\n  conditional_latents 2 ", conditional_latents.shape)
                # 归一化潜在表示
                conditional_latents = conditional_latents / vae.config.scaling_factor
                #   conditional_latents   torch.Size([4, 1, 4, 64, 64])
                #   conditional_latents 2  torch.Size([4, 4, 64, 64])


                # Sample a random timestep for each image
                # P_mean=0.7 P_std=1.6
                sigmas = rand_log_normal(shape=[bsz,], loc=0.7, scale=1.6).to(latents.device)
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                sigmas = sigmas[:, None, None, None, None]
                noisy_latents = latents + noise * sigmas
                timesteps = torch.Tensor(
                    [0.25 * sigma.log() for sigma in sigmas]).to(accelerator.device)

                inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)
                # print("**debug 12 29 \n\n  time steps 1", timesteps)
                ##   time steps 1 tensor([0.2014, 0.3099, 0.0070, 0.7441], device='cuda:0')

                # Sample a random timestep for each video
                # timesteps = torch.randint(
                #     0,
                #     train_noise_scheduler.num_train_timesteps,
                #     (bsz,),
                #     device=latents.device,
                # )
                # timesteps = timesteps.long()
                ##  time steps 2 tensor([ 47, 463, 255, 706], device='cuda:0')

                #print("train_noise_scheduler.num_train_timesteps : ", train_noise_scheduler.num_train_timesteps)
                # 1000
                # print("**debug 12 29 \n\n  time steps 2", timesteps)


                uncond_audio_fwd = random.random() < cfg.uncond_audio_ratio

                # print("**debug 12 29 \n\n  pixel_values_ref_img shape is ", pixel_values_ref_img.shape)
                # pixel_values_ref_img shape is  torch.Size([4, 13, 3, 512, 512])

                image_prompt_embeds = batch["face_emb"].to(
                    dtype=imageproj.dtype, device=imageproj.device
                )

                # process motion buckets:
                motion_bucket_scale = cfg.sonic.motion_bucket_scale
                motion_buckets = torch.stack(motion_buckets, dim=0).to(device="cuda")
                print("motion buckets shape ", motion_buckets.shape)
                motion_buckets = motion_buckets.unsqueeze(0)
                motion_buckets = motion_buckets * motion_bucket_scale
                # motion_bucket = indice_slice(motion_buckets, idx_list)
                motion_bucket = torch.mean(motion_bucket, dim=1).squeeze()
                motion_bucket_id = motion_bucket[0]
                motion_bucket_id_exp = motion_bucket[1]
                added_time_ids = _get_add_time_ids(
                    24,  # fixed
                    127,  # motion_bucket_id = 127, fixed
                    noise_aug_strength,  # noise_aug_strength == cond_sigmas
                    image_prompt_embeds.dtype,
                    bsz,
                )
                # print("**1230\n\n added_time_ids:", added_time_ids)
                added_time_ids = added_time_ids.to(latents.device)
                #  added_time_ids: tensor([[2.4000e+01, 1.2700e+02, 4.0161e-02],
                #         [2.4000e+01, 1.2700e+02, 4.0161e-02],
                #         [2.4000e+01, 1.2700e+02, 4.0161e-02],
                #         [2.4000e+01, 1.2700e+02, 4.0161e-02]], dtype=torch.float16)

                # Conditioning dropout to support classifier-free guidance during inference. For more details
                # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                # if args.conditioning_dropout_prob is not None:
                #     random_p = torch.rand(
                #         bsz, device=latents.device, generator=generator)
                #     # Sample masks for the edit prompts.
                #     prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                #     prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                #     # Final text conditioning.
                #     null_conditioning = torch.zeros_like(encoder_hidden_states)
                #     encoder_hidden_states = torch.where(
                #         prompt_mask, null_conditioning.unsqueeze(1), encoder_hidden_states.unsqueeze(1))
                #     # Sample masks for the original images.
                #     image_mask_dtype = conditional_latents.dtype
                #     image_mask = 1 - (
                #             (random_p >= args.conditioning_dropout_prob).to(
                #                 image_mask_dtype)
                #             * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                #     )
                #     image_mask = image_mask.reshape(bsz, 1, 1, 1)
                #     # Final image conditioning.
                #     conditional_latents = image_mask * conditional_latents



                conditional_latents = conditional_latents.unsqueeze(1)
                # print("**debug 12 29 \n\n  cond latent 1", conditional_latents.shape)
                #   cond latent 1 torch.Size([4, 1, 4, 64, 64])
                conditional_latents = conditional_latents.repeat(1, noisy_latents.shape[1], 1, 1, 1)
                # print("**debug 12 29 \n\n  cond latent2", conditional_latents.shape)
                #   cond latent2 torch.Size([4, 14, 4, 64, 64])

                inp_noisy_latents = torch.cat(
                    [inp_noisy_latents, conditional_latents], dim=2)
                # print("**debug 12 29 \n\n  inp_noisy_latents shape", inp_noisy_latents.shape)
                #   inp_noisy_latents shape torch.Size([4, 14, 8, 64, 64])


                # Get the target for loss depending on the prediction type
                # if train_noise_scheduler.prediction_type == "epsilon":
                #     target = noise
                # elif train_noise_scheduler.prediction_type == "v_prediction":
                #     target = train_noise_scheduler.get_velocity(
                #         latents, noise, timesteps
                #     )
                # else:
                #     raise ValueError(
                #         f"Unknown prediction type {train_noise_scheduler.prediction_type}"
                #     )

                target = latents

                # print("**1230 \n\n noisy_latents:", noisy_latents.shape)
                # print("**1230 \n\n targe shape :", target.shape)
                # print("\n batch audio tensor shape is :", batch["audio_tensor"].shape)


                audio_emb = batch["audio_tensor"].to(dtype=weight_dtype)
                # print("**debug 12 29 \n\n  audio_embd shape", audio_emb.shape)
                # print("**debug 12 29 \n\n  face_embd shape", image_prompt_embeds.shape)
                #   audio_embd shape torch.Size([4, 14, 5, 12, 768])
                #   face_embd shape torch.Size([4, 512])

                inp_noisy_latents = inp_noisy_latents.to(dtype=weight_dtype)
                timesteps = timesteps.to(dtype=weight_dtype)
                # ---- Forward!!! -----
                model_pred = net(
                    noisy_latents=inp_noisy_latents,
                    timesteps=timesteps,
                    face_emb=image_prompt_embeds,
                    audio_emb= audio_emb,
                    added_time_ids=added_time_ids,
                    uncond_audio_fwd=uncond_audio_fwd
                )
                # print("idx", idx , "denoising latents" )
                # Denoise the latents
                c_out = -sigmas / ((sigmas ** 2 + 1) ** 0.5)
                c_skip = 1 / (sigmas ** 2 + 1)
                denoised_latents = model_pred * c_out + c_skip * noisy_latents
                weighing = (1 + sigmas ** 2) * (sigmas ** -2.0)

                # MSE loss
                loss = torch.mean(
                    (weighing.float() * (denoised_latents.float() -
                                         target.float()) ** 2).reshape(target.shape[0], -1),
                    dim=1,
                )
                loss = loss.mean()

                # print("idx", idx, "MSE Loss")

                #
                # if cfg.snr_gamma == 0:
                #     loss = F.mse_loss(
                #         model_pred.float(),
                #         target.float(),
                #         reduction="mean",
                #     )
                # else:
                #     snr = compute_snr(train_noise_scheduler, timesteps)
                #     if train_noise_scheduler.config.prediction_type == "v_prediction":
                #         # Velocity objective requires that we add one to SNR values before we divide by them.
                #         snr = snr + 1
                #     mse_loss_weights = (
                #         torch.stack(
                #             [snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                #         ).min(dim=1)[0]
                #         / snr
                #     )
                #     loss = F.mse_loss(
                #         model_pred.float(),
                #         target.float(),
                #         reduction="mean",
                #     )
                #     loss = (
                #         loss.mean(dim=list(range(1, len(loss.shape))))
                #         * mse_loss_weights
                #     ).mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(
                    loss.repeat(cfg.data.train_bs)).mean()
                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps
                # print("idx", idx, "train loss")
                # Backpropagate
                # accelerator.backward(loss)
                # print(f"Loss requires_grad: {loss.requires_grad}, Loss grad_fn: {loss.grad_fn}")
                try:
                    accelerator.backward(loss)
                except Exception as e:
                    import traceback
                    print("Exception in backward pass:", e)
                    print(traceback.format_exc())

                # print("idx", idx, "backward done")
                if accelerator.sync_gradients:
                    # print("idx", idx, "clip grad norm")
                    accelerator.clip_grad_norm_(
                        trainable_params,
                        cfg.solver.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # print("idx", idx, "step done")


            if accelerator.sync_gradients:
                # print("idx", idx, "progress_bar")
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # if global_step % cfg.val.validation_steps == 0 or global_step==1:
                #     if accelerator.is_main_process:
                #         generator = torch.Generator(device=accelerator.device)
                #         generator.manual_seed(cfg.seed)

                        # log_validation(
                        #     accelerator=accelerator,
                        #     vae=vae,
                        #     net=net,
                        #     scheduler=val_noise_scheduler,
                        #     width=cfg.data.train_width,
                        #     height=cfg.data.train_height,
                        #     clip_length=cfg.data.n_sample_frames,
                        #     cfg=cfg,
                        #     save_dir=validation_dir,
                        #     global_step=global_step,
                        #     times=cfg.single_inference_times if cfg.single_inference_times is not None else None,
                        #     face_analysis_model_path=cfg.face_analysis_model_path
                        # )
            # print("idx", idx, "logs")
            torch.cuda.empty_cache()
            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "td": f"{t_data:.2f}s",
            }
            t_data_start = time.time()
            progress_bar.set_postfix(**logs)

            if (
                global_step % cfg.checkpointing_steps == 0
                or global_step == cfg.solver.max_train_steps
            ):
                # save model
                # print("idx", idx, "save")
                save_path = os.path.join(
                    checkpoint_dir, f"checkpoint-{global_step}")
                if accelerator.is_main_process:
                    delete_additional_ckpt(checkpoint_dir, 30)
                accelerator.wait_for_everyone()
                accelerator.save_state(save_path)

                # save model weight
                unwrap_net = accelerator.unwrap_model(net)
                if accelerator.is_main_process:
                    save_checkpoint(
                        unwrap_net,
                        module_dir,
                        "net",
                        global_step,
                        total_limit=3,
                    )
            if global_step >= cfg.solver.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


def load_config(config_path: str) -> dict:
    """
    Loads the configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: The configuration dictionary.
    """

    if config_path.endswith(".yaml"):
        return OmegaConf.load(config_path)
    if config_path.endswith(".py"):
        return import_filename(config_path).cfg
    raise ValueError("Unsupported format for config file")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="./configs/train/svd.yaml"
    )
    args = parser.parse_args()
    torch.multiprocessing.set_start_method('spawn')
    config = load_config(args.config)
    train_stage2_process(config)
