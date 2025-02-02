# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import math
import numpy as np
import PIL.Image
import torch
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.models import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from .context import get_context_scheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

from transformers import CLIPImageProcessor
from transformers import AutoFeatureExtractor,  WhisperModel
from hallo.utils.test_preprocess import process_bbox, image_audio_to_tensor, get_audio_feature

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import StableVideoDiffusionPipeline
        >>> from diffusers.utils import load_image, export_to_video

        >>> pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
        >>> pipe.to("cuda")

        >>> image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd-docstring-example.jpeg")
        >>> image = image.resize((1024, 576))

        >>> frames = pipe(image, num_frames=25, decode_chunk_size=8).frames[0]
        >>> export_to_video(frames, "generated.mp4", fps=7)
        ```
"""


def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


# Copied from diffusers.pipelines.animatediff.pipeline_animatediff.tensor2vid
def tensor2vid(video: torch.Tensor, processor: VaeImageProcessor, output_type: str = "np"):
    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    if output_type == "np":
        outputs = np.stack(outputs)

    elif output_type == "pt":
        outputs = torch.stack(outputs)

    elif not output_type == "pil":
        raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil']")

    return outputs


@dataclass
class StableVideoDiffusionPipelineOutput(BaseOutput):
    r"""
    Output class for Stable Video Diffusion pipeline.

    Args:
        frames (`[List[List[PIL.Image.Image]]`, `np.ndarray`, `torch.FloatTensor`]):
            List of denoised PIL images of length `batch_size` or numpy array or torch tensor
            of shape `(batch_size, num_frames, height, width, num_channels)`.
    """

    frames: Union[List[List[PIL.Image.Image]], np.ndarray, torch.FloatTensor]


class StableVideoDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline to generate video from an input image using Stable Video Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKLTemporalDecoder`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        image_encoder ([`~transformers.CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder ([laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)).
        unet ([`UNetSpatioTemporalConditionModel`]):
            A `UNetSpatioTemporalConditionModel` to denoise the encoded image latents.
        scheduler ([`EulerDiscreteScheduler`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images.
    """

    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        vae: AutoencoderKLTemporalDecoder,
        unet: UNetSpatioTemporalConditionModel,
        scheduler: EulerDiscreteScheduler,
        # feature_extractor: CLIPImageProcessor,
        # audio_guider,
        image_proj,
        audio2bucket,
        audio2token

    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            # feature_extractor=feature_extractor,
            audio2bucket = audio2bucket,
            image_proj = image_proj,
            audio2token = audio2token,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def _encode_image(
        self,
        image: PipelineImageInput,
        device: Union[str, torch.device],
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ) -> torch.FloatTensor:
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.image_processor.pil_to_numpy(image)
            image = self.image_processor.numpy_to_pt(image)

            # We normalize the image before resizing to match with the original implementation.
            # Then we unnormalize it after resizing.
            image = image * 2.0 - 1.0
            image = _resize_with_antialiasing(image, (224, 224))
            image = (image + 1.0) / 2.0

        # Normalize the image with for CLIP input
        image = self.feature_extractor(
            images=image,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

        return image_embeddings

    def _encode_vae_image(
        self,
        image: torch.Tensor,
        device: Union[str, torch.device],
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ):
        image = image.to(device=device)
        image_latents = self.vae.encode(image).latent_dist.mode()

        if do_classifier_free_guidance:
            negative_image_latents = torch.zeros_like(image_latents)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_latents = torch.cat([negative_image_latents, image_latents])

        # duplicate image_latents for each generation per prompt, using mps friendly method
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)

        return image_latents

    def _get_add_time_ids(
        self,
        fps: int,
        motion_bucket_id: int,
        noise_aug_strength: float,
        dtype: torch.dtype,
        batch_size: int,
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)

        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids])

        return add_time_ids

    def decode_latents(self, latents: torch.FloatTensor, num_frames: int, decode_chunk_size: int = 14):
        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]

        print("Decode latents\n\n")
        print("latents shape is ", latents.shape)
        print("num_frames is ", num_frames)
        print("decode chunk size is ", decode_chunk_size)

        latents = latents.flatten(0, 1)

        latents = 1 / self.vae.config.scaling_factor * latents

        forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
        accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())

        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                # we only pass num_frames_in if it's expected
                decode_kwargs["num_frames"] = num_frames_in

            frame = self.vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)

        print("frames shape is ", frames.shape)
        # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.float()
        return frames

    def check_inputs(self, image, height, width):
        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    def prepare_latents(
        self,
        batch_size: int,
        num_frames: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: Union[str, torch.device],
        generator: torch.Generator,
        latents: Optional[torch.FloatTensor] = None,
    ):
        shape = (
            batch_size,
            num_frames,
            num_channels_latents // 2,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            print("111")
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            print("222")
            latents = latents.to(device=device,dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        print("return latents d type ", latents.dtype)
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        if isinstance(self.guidance_scale, (int, float)):
            return self.guidance_scale >= 1
        return self.guidance_scale.max() >= 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        face_emb,
        clip_image,
        audio_path,
        video_length,
        height: int = 512,
        width: int = 512,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 25,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
        context_schedule="uniform",
        context_frames=25,
        context_stride=1,
        context_overlap=5,
        context_batch_size=1,
        weight_dtype = torch.float16,

    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                Image(s) to guide image generation. If you provide a tensor, the expected value range is between `[0, 1]`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_frames (`int`, *optional*):
                The number of video frames to generate. Defaults to `self.unet.config.num_frames`
                (14 for `stable-video-diffusion-img2vid` and to 25 for `stable-video-diffusion-img2vid-xt`).
            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to a higher quality video at the
                expense of slower inference. This parameter is modulated by `strength`.
            min_guidance_scale (`float`, *optional*, defaults to 1.0):
                The minimum guidance scale. Used for the classifier free guidance with first frame.
            max_guidance_scale (`float`, *optional*, defaults to 3.0):
                The maximum guidance scale. Used for the classifier free guidance with last frame.
            fps (`int`, *optional*, defaults to 7):
                Frames per second. The rate at which the generated images shall be exported to a video after generation.
                Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
            motion_bucket_id (`int`, *optional*, defaults to 127):
                Used for conditioning the amount of motion for the generation. The higher the number the more motion
                will be in the video.
            noise_aug_strength (`float`, *optional*, defaults to 0.02):
                The amount of noise added to the init image, the higher it is the less the video will look like the init image. Increase it for more motion.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time. Higher chunk size leads to better temporal consistency at the expense of more memory usage. By default, the decoder decodes all frames at once for maximal
                quality. For lower memory usage, reduce `decode_chunk_size`.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `pil`, `np` or `pt`.
            callback_on_step_end (`Callable`, *optional*):
                A function that is called at the end of each denoising step during inference. The function is called
                with the following arguments:
                    `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`.
                `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` of (`List[List[PIL.Image.Image]]` or `np.ndarray` or `torch.FloatTensor`) is returned.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames
        print("num_frames ", num_frames)
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, height, width)

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
        device = self._execution_device
        print("device is :", device)
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self._guidance_scale = max_guidance_scale

        # prepare audio
        whisper_feature = self.audio_guider.audio2feat(audio_path)
        print("whisper fps is ", fps )
        whisper_chunks = self.audio_guider.feature2chunks(feature_array=whisper_feature, fps=fps)

        print("\n whisper_chunks:", whisper_chunks.shape)
        # whisper_chunks: (189, 50, 384)

        # new audio feature start
        print("new audio start \n\n ")
        from transformers import WhisperModel
        wav_enc = WhisperModel.from_pretrained("/yuch_ws/DH/hallo2/pretrained_models/whisper-tiny/").eval()
        wav_enc.requires_grad_(False)
        feature_extractor = AutoFeatureExtractor.from_pretrained("/yuch_ws/DH/hallo2/pretrained_models/whisper-tiny/")


        audio_input, audio_len = get_audio_feature(audio_path, feature_extractor)
        audio_feature = audio_input[0]
        audio_feature = audio_feature.unsqueeze(0)

        window = 3000
        audio_prompts = []
        last_audio_prompts = []
        for i in range(0, audio_feature.shape[-1], window):
            audio_prompt = wav_enc.encoder(audio_feature[:, :, i:i + window],
                                           output_hidden_states=True).hidden_states
            last_audio_prompt = wav_enc.encoder(audio_feature[:, :, i:i + window]).last_hidden_state
            last_audio_prompt = last_audio_prompt.unsqueeze(-2)
            audio_prompt = torch.stack(audio_prompt, dim=2)
            audio_prompts.append(audio_prompt)
            last_audio_prompts.append(last_audio_prompt)
            # print(video_path[-15:-4], "audio_prompts chunk shape", audio_prompt.shape)
            # print(video_path[-15:-4], "last audio prompts shape", last_audio_prompt.shape)
            # lSteele_000 audio_prompts chunk shape torch.Size([1, 1500, 5, 384])
            # lSteele_000 last audio prompts shape torch.Size([1, 1500, 1, 384])

        audio_prompts = torch.cat(audio_prompts, dim=1)
        # print(video_path[-15:-4], "audio_prompts 1 shape", audio_prompts.shape)
        # lSteele_000 audio_prompts 1 shape torch.Size([1, 1500, 5, 384])
        audio_prompts = audio_prompts[:, :audio_len * 2]
        # print(video_path[-15:-4], "audio_prompts 2 shape", audio_prompts.shape)
        # lSteele_000 audio_prompts 2 shape torch.Size([1, 650, 5, 384])
        audio_prompts = torch.cat(
            [torch.zeros_like(audio_prompts[:, :4]), audio_prompts, torch.zeros_like(audio_prompts[:, :6])], 1)

        last_audio_prompts = torch.cat(last_audio_prompts, dim=1)
        last_audio_prompts = last_audio_prompts[:, :audio_len * 2]
        last_audio_prompts = torch.cat([torch.zeros_like(last_audio_prompts[:, :24]), last_audio_prompts,
                                        torch.zeros_like(last_audio_prompts[:, :26])], 1)

        print("audio_prompts:", audio_prompts.shape)
        print("last_audio_prompts:", last_audio_prompts.shape)
        print("new audio end \n\n ")
        # new audio feature end


        audio_frame_num = whisper_chunks.shape[0]
        audio_fea_final = torch.Tensor(whisper_chunks).to(dtype=self.vae.dtype, device=self.vae.device)
        audio_fea_final = audio_fea_final.unsqueeze(0)
        print("\n audio_fea_final:", audio_fea_final.shape)
        #  audio_fea_final: torch.Size([1, 189, 50, 384])
        video_length = min(video_length, audio_frame_num)
        if video_length < audio_frame_num:
            audio_fea_final = audio_fea_final[:, :video_length, :, :]

        context_scheduler = get_context_scheduler(context_schedule)
        context_queue = list(
            context_scheduler(
                0,
                num_inference_steps,
                video_length,#latents.shape[2],
                context_frames,
                context_stride,
                context_overlap,
            )
        )
        print("***\n\n\n context_queue:", len(context_queue), context_queue)
        #  context_queue: 16 [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35], [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47], [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59], [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71], [72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83], [84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95], [96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107], [108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119], [120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131], [132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143], [144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155], [156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167], [168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179], [180, 181, 182, 183, 184, 185, 186, 187, 188, 0, 1, 2]]
        # 3. Encode input image
        image_embeddings = face_emb

        # NOTE: Stable Video Diffusion was conditioned on fps - 1, which is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        fps = fps - 1

        # 4. Encode input image using VAE
        # image = self.image_processor.preprocess(image, height=height, width=width).to(device)
        noise = randn_tensor(image.shape, generator=generator, device=device, dtype=weight_dtype)
        image = image.to( device=device)
        image = image + noise_aug_strength * noise

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        image_latents = self._encode_vae_image(
            image,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )
        image_latents = image_latents.to(image_embeddings.dtype)

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        print("current motion_bucket_id is ", motion_bucket_id )
        motion_bucket_id = 250
        print("decrease motion_bucket_id to ", motion_bucket_id)
        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        # 6. Prepare timesteps
        # self.scheduler.set_timesteps(num_inference_steps, device=device)
        # timesteps = self.scheduler.timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, None)

        # 7. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        print("ww\n\n weight_dtype is ", weight_dtype)
        print(" vae_dtype is ", self.vae.dtype)
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            video_length,
            num_channels_latents,
            height,
            width,
            weight_dtype,
            device,
            generator,
            latents,
        )
        print("latents d type is ", latents.dtype)
        # 8. Prepare guidance scale
        # guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        # guidance_scale = guidance_scale.to(device, latents.dtype)
        # guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        # guidance_scale = _append_dims(guidance_scale, latents.ndim)

        guidance_scale = torch.linspace(
            min_guidance_scale,
            max_guidance_scale,
            num_inference_steps)

        print("guidance scale")
        print("do CFG", self.do_classifier_free_guidance)
        # print(guidance_scale.shape)
        # [1,25,1,1,1]
        # print(guidance_scale)
        # 1,1.2,1.31,1.41,1.52.....3.39,3.5
        self._guidance_scale = guidance_scale
        print("self._guidance_scale", self._guidance_scale)

        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)


        # prepare image embeddings

        print("image_embeddings shape is ", image_embeddings.shape)
        image_embeddings_cfg = torch.cat([torch.zeros_like(image_embeddings), image_embeddings], 0)
        print("image_embeddings cfg shape is ", image_embeddings.shape)
        print("do cfg: ", self.do_classifier_free_guidance)
        image_embeddings = image_embeddings_cfg if self.do_classifier_free_guidance else image_embeddings
        image_embeddings = image_embeddings.to(device=self.image_proj.device, dtype=self.image_proj.dtype)
        image_embeddings = self.image_proj(image_embeddings)


        # timesteps = timesteps
        print("DEBUG\n\n timesteps old", len(timesteps), timesteps.dtype, timesteps)


        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                noise_pred = torch.zeros(
                    (
                        latents.shape[0] * (2 if self.do_classifier_free_guidance else 1),
                        *latents.shape[1:],
                    ),
                    device=latents.device,
                    dtype=latents.dtype,
                )
                counter = torch.zeros(
                    (1, latents.shape[1], 1, 1, 1),
                    device=latents.device,
                    dtype=latents.dtype,
                )

                # print("***\n\n\n i.t:", i, t, t.dtype)
                # print("noise pred shape is :", noise_pred.shape)
                # print("latent shape is ", latents.shape )
                # print("counter shape is : ", counter.shape)
                #  noise pred shape is : torch.Size([2, 25, 4, 64, 64])
                #  counter shape is :  torch.Size([1, 1, 4, 1, 1])

                # context

                num_context_batches = math.ceil(len(context_queue) / context_batch_size)
                # should be same as context_queue

                global_context = []
                for j in range(num_context_batches):
                    global_context.append(
                        context_queue[
                        j * context_batch_size: (j + 1) * context_batch_size
                        ]
                    )
                # TODO add shift count here
                shift_count = 5
                for context in global_context:
                    # print("global_context:", len(global_context), global_context)
                    # context = global_context[0]
                    # print("context is ", context)
                    new_context = [[0 for _ in range(len(context[c_j]))] for c_j in range(len(context))]
                    # print("new context is ", new_context)  # all zeros

                    for c_j in range(len(context)):
                        for c_i in range(len(context[c_j])):
                            new_context[c_j][c_i] = (context[c_j][c_i] + i * shift_count) % video_length
                    # print("new context 2 is ", new_context)
                    # print("latents shape is ", latents.shape)
                    # latents shape is  torch.Size([1, 189, 4, 64, 64])

                    latent_model_input = (
                        torch.cat([latents[:, c, :] for c in new_context])
                        .to(device=device, dtype=latents.dtype)
                        .repeat(2 if self.do_classifier_free_guidance else 1, 1, 1, 1, 1)
                    )

                    # print("latent_model_input shape is :", latent_model_input.shape, latent_model_input.dtype)
                    # latent_model_input shape is : torch.Size([2, 25, 4, 64, 64])
                    c_audio_latents = torch.cat([audio_fea_final[:, c] for c in new_context]).to(device)
                    # print("c_audio_latents shape is :", c_audio_latents.shape)
                    # c_audio_latents shape is : torch.Size([1, 25, 50, 384])
                    audio_latents = torch.cat([torch.zeros_like(c_audio_latents), c_audio_latents], 0)
                    # print("audio_latents shape is :", audio_latents.shape)
                    # audio_latents shape is : torch.Size([2, 25, 50, 384])
                    # expand the latents if we are doing classifier free guidance
                    # latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                    # print("latent_model_input shape ", latent_model_input.shape , t )



                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # print("latent_model_input shape is ", latent_model_input.shape)
                    # latent_model_input shape is  torch.Size([2, 189, 4, 64, 64])
                    # print("image_latest shape is ", image_latents.shape)
                    # image_latest shape is  torch.Size([2, 25, 4, 64, 64])

                    # Concatenate image_latents over channels dimension
                    latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)





                    audio_latents =audio_latents if self.do_classifier_free_guidance else c_audio_latents
                    audio_latents = audio_latents.to(device=self.audio_proj.device,dtype = self.audio_proj.dtype)
                    audio_latents = self.audio_proj(audio_latents)



                    t = t.to(dtype=weight_dtype)
                    # print("t dtype is ", t.dtype)
                    # predict the noise residual

                    # print("**01,11\n\n dtype check before entering unet:")
                    # print("latent model", latent_model_input.dtype, latent_model_input.device)
                    # print("t", t.dtype)
                    # print("audio latents", audio_latents.dtype, audio_latents.device)
                    # print("image_embeddings_cfg", image_embeddings.dtype, image_embeddings.device)



                    pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states= image_embeddings,
                        audio_embedding=audio_latents,
                        added_time_ids=added_time_ids,
                        return_dict=False,
                    )[0]

                    # print("**jc\n\n ", new_context)
                    # print("before update count :", counter)
                    for j, c in enumerate(new_context):

                        # print(" noise_pred[:, :, c] shape is ",  noise_pred[:, c].shape)
                        # print("pred shape is ", pred.shape)
                        # print("counter[:c] shape is ", counter[:,c].shape)
                        noise_pred[:,c] = noise_pred[:, c] + pred
                        counter[:, c] = counter[:, c] + 1
                    # print("after update count : ", counter )


                # print("check counter", counter)
                # perform guidance
                if self.do_classifier_free_guidance:
                    # print("noise pred shape ", noise_pred.shape)
                    noise_pred_uncond, noise_pred_cond = (noise_pred / counter).chunk(2)
                    # print("noise pred uncond shape", noise_pred_uncond.shape , "noise_pred cond shape ", noise_pred_cond.shape)
                    # print("self.guidance scaleis ", self.guidance_scale[i].shape)
                    noise_pred = noise_pred_uncond + self.guidance_scale[i] * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)

            frames = self.decode_latents(latents, video_length, decode_chunk_size)
            print("after decode frames shape are ", frames)
            # frames = tensor2vid(frames, self.image_processor, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)


# resizing utils
# TODO: clean up later
def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out
