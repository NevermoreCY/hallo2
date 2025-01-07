# pylint: disable=E1101
# scripts/inference.py

"""
This script contains the main inference pipeline for processing audio and image inputs to generate a video output.

The script imports necessary packages and classes, defines a neural network model, 
and contains functions for processing audio embeddings and performing inference.

The main inference process is outlined in the following steps:
1. Initialize the configuration.
2. Set up runtime variables.
3. Prepare the input data for inference (source image, face mask, and face embeddings).
4. Process the audio embeddings.
5. Build and freeze the model and scheduler.
6. Run the inference loop and save the result.

Usage:
This script can be run from the command line with the following arguments:
- audio_path: Path to the audio file.
- image_path: Path to the source image.
- face_mask_path: Path to the face mask image.
- face_emb_path: Path to the face embeddings file.
- output_path: Path to save the output video.

Example:
python scripts/inference.py --audio_path audio.wav --image_path image.jpg 
    --face_mask_path face_mask.png --face_emb_path face_emb.pt --output_path output.mp4
"""

import argparse
import os
import sys

import torch
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from torch import nn
from pathlib import Path
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from pydub import AudioSegment

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hallo.animate.pipeline_svd_whisper import StableVideoDiffusionPipeline
from hallo.datasets.audio_processor import AudioProcessor
from hallo.datasets.image_processor import ImageProcessor
from hallo.models.audio_proj_whisper import AudioProjModel
from hallo.models.face_locator import FaceLocator
from hallo.models.image_proj import ImageProjModel
from hallo.models.unet_2d_condition import UNet2DConditionModel
from hallo.models.unet_3d import UNet3DConditionModel
from hallo.utils.config import filter_non_none
from hallo.utils.util import tensor_to_video_batch, merge_videos
from hallo.utils.util import tensor_to_video

from hallo.models.whisper_local.audio2feature import load_audio_model
from hallo.models.diffuser.unet_spatio_temporal_condition_audio import UNetSpatioTemporalConditionModel

class Net(nn.Module):
    """
    The Net class defines a neural network model that combines a reference UNet2DConditionModel,
    a denoising UNet3DConditionModel, a face locator, and other components to animate a face in a static image.

    Args:

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
    ):
        super().__init__()
        self.denoising_unet = denoising_unet
        self.face_locator = face_locator
        self.imageproj = imageproj
        self.audioproj = audioproj

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

def process_audio_emb(audio_emb):
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
            audio_emb[max(min(i + j, audio_emb.shape[0]-1), 0)]for j in range(-2, 3)]
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


def cut_audio(audio_path, save_dir, length=60):
    audio = AudioSegment.from_wav(audio_path)

    segment_length = length * 1000 # pydub使用毫秒

    num_segments = len(audio) // segment_length + (1 if len(audio) % segment_length != 0 else 0)

    os.makedirs(save_dir, exist_ok=True)

    audio_list = [] 

    for i in range(num_segments):
        start_time = i * segment_length
        end_time = min((i + 1) * segment_length, len(audio))
        segment = audio[start_time:end_time]
        
        path = f"{save_dir}/segment_{i+1}.wav"
        audio_list.append(path)
        segment.export(path, format="wav")

    return audio_list


def inference_process(args: argparse.Namespace):
    """
    Perform inference processing.

    Args:
        args (argparse.Namespace): Command-line arguments.

    This function initializes the configuration for the inference process. It sets up the necessary
    modules and variables to prepare for the upcoming inference steps.
    """
    # 1. init config
    cli_args = filter_non_none(vars(args))
    config = OmegaConf.load(args.config)
    config = OmegaConf.merge(config, cli_args)
    source_image_paths = config.source_image
    driving_audio_paths = config.driving_audio
    # Create a directory for each source image
    for source_image_path in source_image_paths:
        save_path = os.path.join(config.save_path, Path(source_image_path).stem)
        save_seg_path = os.path.join(save_path, "seg_video")
        print("save path: ", save_path)
    
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_seg_path):
            os.makedirs(save_seg_path)

    motion_scale = [config.pose_weight, config.face_weight, config.lip_weight]

    # 2. runtime variables
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif config.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    elif config.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        weight_dtype = torch.float32

    # 3. prepare inference data
    # 3.1 prepare source image, face mask, face embeddings
    img_size = (config.data.source_image.width,
                config.data.source_image.height)
    clip_length = config.data.n_sample_frames
    face_analysis_model_path = config.face_analysis.model_path
    image_processor = ImageProcessor(img_size, face_analysis_model_path)
    # with ImageProcessor(img_size, face_analysis_model_path) as image_processor:
    #     source_image_pixels, \
    #     source_image_face_region, \
    #     source_image_face_emb, \
    #     source_image_full_mask, \
    #     source_image_face_mask, \
    #     source_image_lip_mask = image_processor.preprocess(
    #         source_image_path, save_path, config.face_expand_ratio)

    # 3.2 prepare audio embeddings
    sample_rate = config.data.driving_audio.sample_rate
    assert sample_rate == 16000, "audio sample rate must be 16000"
    fps = config.data.export_video.fps
    wav2vec_model_path = config.wav2vec.model_path
    wav2vec_only_last_features = config.wav2vec.features == "last"
    audio_separator_model_file = config.audio_separator.model_path

    # ASSUME NOT USING CUT
    # if config.use_cut:
    #     audio_list = cut_audio(driving_audio_path, os.path.join(
    #         save_path, f"seg-long-{Path(driving_audio_path).stem}"))
    #
    #     audio_emb_list = []
    #     l = 0
    #
    #     audio_processor = AudioProcessor(
    #             sample_rate,
    #             fps,
    #             wav2vec_model_path,
    #             wav2vec_only_last_features,
    #             os.path.dirname(audio_separator_model_file),
    #             os.path.basename(audio_separator_model_file),
    #             os.path.join(save_path, "audio_preprocess")
    #         )
    #
    #     for idx, audio_path in enumerate(audio_list):
    #         padding = (idx+1) == len(audio_list)
    #         emb, length = audio_processor.preprocess(audio_path, clip_length,
    #                                                  padding=padding, processed_length=l)
    #         audio_emb_list.append(emb)
    #         l += length
    #
    #     audio_emb = torch.cat(audio_emb_list)
    #     audio_length = l
    #
    # else:
    #     with AudioProcessor(
    #             sample_rate,
    #             fps,
    #             wav2vec_model_path,
    #             wav2vec_only_last_features,
    #             os.path.dirname(audio_separator_model_file),
    #             os.path.basename(audio_separator_model_file),
    #             os.path.join(save_path, "audio_preprocess")
    #         ) as audio_processor:
    #             audio_emb, audio_length = audio_processor.preprocess(driving_audio_path, clip_length)

    audio_processor = AudioProcessor(
        sample_rate,
        fps,
        wav2vec_model_path,
        wav2vec_only_last_features,
        os.path.dirname(audio_separator_model_file),
        os.path.basename(audio_separator_model_file),
        os.path.join(save_path, "audio_preprocess")
    )

    audio_model_path = "/yuch_ws/DH/hallo2/pretrained_models/whisper_tiny.pt"
    audio_guider = load_audio_model(model_path=audio_model_path, device='cuda')


    # 4. build modules
    sched_kwargs = OmegaConf.to_container(config.noise_scheduler_kwargs)
    if config.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})

    vae = AutoencoderKL.from_pretrained(config.vae.model_path)


    unet = UNetSpatioTemporalConditionModel(
        sample_size = 96
    )

    from safetensors.torch import load_file
    # modle_path = "exp_output/svd_test_3/net-21500.pth"
    # state_dict = torch.load(modle_path)
    #
    # m,u = unet.load_state_dict(state_dict, strict=False)
    # print("** missing keys : \n\n", m)
    # print("** unexpected keys : \n\n", u)

    # denoising_unet.set_attn_processor()

    face_locator = FaceLocator(conditioning_embedding_channels=320)
    image_proj = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=512,
        clip_extra_context_tokens=4,
    )
    #
    # audio_proj = AudioProjModel(
    #     seq_len=5,
    #     blocks=12,  # use 12 layers' hidden states of wav2vec
    #     channels=768,  # audio embedding channel
    #     intermediate_dim=512,
    #     output_dim=768,
    #     context_tokens=32,
    # ).to(device=device, dtype=weight_dtype)


    # whisper

    audio_proj = AudioProjModel(
        window=50,
        channels=384,
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
    ).to(device="cuda", dtype=weight_dtype)

    audio_ckpt_dir = config.audio_ckpt_dir


    # Freeze
    vae.requires_grad_(False)
    image_proj.requires_grad_(False)
    unet.requires_grad_(False)

    face_locator.requires_grad_(False)
    audio_proj.requires_grad_(False)


    net = Net(
        unet,
        face_locator,
        image_proj,
        audio_proj,
    )

    m,u = net.load_state_dict(
        torch.load(
            os.path.join(audio_ckpt_dir, f"net-1500.pth"),
            map_location="cpu",
        ),
    )
    assert len(m) == 0 and len(u) == 0, "Fail to load correct checkpoint."
    print("\n\n\n\n\n **** loaded weight from ", os.path.join(audio_ckpt_dir, "net-1500.pth"))

    # vae: AutoencoderKLTemporalDecoder,
    # unet: UNetSpatioTemporalConditionModel,
    # scheduler: EulerDiscreteScheduler,
    # feature_extractor: CLIPImageProcessor,
    # audio_guider,
    # image_proj,

    # 5. inference
    pipeline = StableVideoDiffusionPipeline(
        vae=vae,
        unet=net.denoising_unet,
        scheduler=val_noise_scheduler,
        audio_guider=audio_guider,
        image_proj=net.imageproj,

    )
    pipeline.to(device=device, dtype=weight_dtype)


    # for each reference image exmaple
    for idx, source_image_path in enumerate(source_image_paths):

        source_image_name = os.path.basename(source_image_path)[:-4]
        source_image_pixels, \
            source_image_face_region, \
            source_image_face_emb, \
            source_image_full_mask, \
            source_image_face_mask, \
            source_image_lip_mask = image_processor.preprocess(
            source_image_path, save_path, config.face_expand_ratio)

        driving_audio_path = driving_audio_paths[idx]
        driving_audio_name = os.path.basename(driving_audio_path)[:-4]


        audio_emb, audio_length = audio_processor.preprocess(driving_audio_path, clip_length)

        audio_emb = process_audio_emb(audio_emb)

        whisper_feature = audio_guider.audio2feat(driving_audio_path)
        print("whisper feature shape :", whisper_feature.shape)
        whisper_chunks = audio_guider.feature2chunks(feature_array=whisper_feature, fps=25)
        audio_frame_num = whisper_chunks.shape[0]
        audio_fea_final = torch.Tensor(whisper_chunks)
        print("audio_fea_final shape ", audio_fea_final.shape)
        audio_length = audio_fea_final.shape[0] - 1


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

        # ic(audio_emb.shape)
        # ic(audio_length)
        batch_size = 60
        start = 0

        # for each time clip
        for t in range(times):
            print(f"[{t+1}/{times}]")
            print("\n wihout motion frame, reference image shape :", source_image_pixels)
            if len(tensor_result) == 0:
                # The first iteration
                motion_zeros = source_image_pixels.repeat(
                    config.data.n_motion_frames, 1, 1, 1)
                # zero mask
                motion_zeros = torch.zeros_like(motion_zeros)
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
                # zero mask
                motion_frames_to_pad = torch.zeros_like(motion_frames_to_pad)
                motion_frames = torch.cat([motion_frames_to_pad, motion_frames], dim=0)
                pixel_values_ref_img = torch.cat(
                    [source_image_pixels, motion_frames], dim=0)  # concat the ref image and the motion frames
            else:
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

            #  [1, 3, 3, 512, 512]

            pixel_motion_values = pixel_values_ref_img[:, 1:]

            if config.use_mask:
                b, f, c, h, w = pixel_motion_values.shape
                rand_mask = torch.rand(h, w)
                mask = rand_mask > config.mask_rate
                mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                mask = mask.expand(b, f, c, h, w)

                face_mask = source_image_face_region.repeat(f, 1, 1, 1).unsqueeze(0)
                assert face_mask.shape == mask.shape
                mask = mask | face_mask.bool()

                pixel_motion_values = pixel_motion_values * mask
                pixel_values_ref_img[:, 1:] = pixel_motion_values


            assert pixel_motion_values.shape[0] == 1

            audio_tensor = audio_fea_final[
                t * clip_length: min((t + 1) * clip_length, audio_emb.shape[0]-1)
            ]
            audio_tensor = audio_tensor.unsqueeze(0)
            audio_tensor = audio_tensor.to(
                device=net.audioproj.device, dtype=net.audioproj.dtype)
            audio_tensor = net.audioproj(audio_tensor)

            print("\n\n start run pipeline ")

            #         self,
            #         image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
            #         face_emb,
            #         audio_path,
            #         video_length,
            #         height: int = 576,
            #         width: int = 1024,
            #         num_frames: Optional[int] = None,
            #         num_inference_steps: int = 25,
            #         min_guidance_scale: float = 1.0,
            #         max_guidance_scale: float = 3.0,
            #         fps: int = 7,
            #         motion_bucket_id: int = 127,
            #         noise_aug_strength: float = 0.02,
            #         decode_chunk_size: Optional[int] = None,
            #         num_videos_per_prompt: Optional[int] = 1,
            #         generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            #         latents: Optional[torch.FloatTensor] = None,
            #         output_type: Optional[str] = "pil",
            #         callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            #         callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            #         return_dict: bool = True,
            #         context_schedule="uniform",
            #
            #     ):

            pipeline_output = pipeline(
                image=pixel_values_ref_img,
                face_emb=source_image_face_emb,
                audio_path = driving_audio_path,
                video_length = 1200,
                width=img_size[0],
                height=img_size[1],
                num_inference_steps=config.inference_steps,
                max_guidance_scale=config.cfg_scale,
                generator=generator,
            )

            tensor_result.append(pipeline_output.videos)

        tensor_result = torch.cat(tensor_result, dim=2)
        tensor_result = tensor_result.squeeze(0)
        tensor_result = tensor_result[:, :audio_length]

        output_file = config.save_path + source_image_name + '_' + driving_audio_name + '.mp4'
        # save the result after all iteration
        tensor_to_video(tensor_result, output_file, driving_audio_path)


        # return output_file
        #
        #     if (t+1) % batch_size == 0 or (t+1)==times:
        #         last_motion_frame = [tensor_result[-1]]
        #         ic(len(tensor_result))
        #
        #         if start!=0:
        #             tensor_result = torch.cat(tensor_result[1:], dim=2)
        #         else:
        #             tensor_result = torch.cat(tensor_result, dim=2)
        #
        #         tensor_result = tensor_result.squeeze(0)
        #         f = tensor_result.shape[1]
        #         length = min(f, audio_length)
        #         tensor_result = tensor_result[:, :length]
        #
        #         ic(tensor_result.shape)
        #         ic(start)
        #         ic(audio_length)
        #
        #         name = Path(save_path).name
        #         output_file = os.path.join(save_seg_path, f"{name}-{t+1:06}.mp4")
        #
        #         tensor_to_video_batch(tensor_result, output_file, start, driving_audio_path)
        #         del tensor_result
        #
        #         tensor_result = last_motion_frame
        #         audio_length -= length
        #         start += length
        #
        # merge_videos(save_seg_path, os.path.join(Path(save_seg_path).parent, "merge_video.mp4"))

    return save_seg_path
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c", "--config", default="configs/inference/long_loopy.yaml")
    parser.add_argument("--source_image", type=str, required=False,
                        help="source image")
    parser.add_argument("--driving_audio", type=str, required=False,
                        help="driving audio")
    parser.add_argument(
        "--pose_weight", type=float, help="weight of pose", required=False)
    parser.add_argument(
        "--face_weight", type=float, help="weight of face", required=False)
    parser.add_argument(
        "--lip_weight", type=float, help="weight of lip", required=False)
    parser.add_argument(
        "--face_expand_ratio", type=float, help="face region", required=False)
    parser.add_argument(
        "--audio_ckpt_dir", "--checkpoint", default="/yuch_ws/DH/hallo2/exp_output/svd_whisper_test",type=str, help="specific checkpoint dir", required=False)


    command_line_args = parser.parse_args()

    save_path = inference_process(command_line_args)

