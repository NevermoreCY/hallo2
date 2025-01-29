# pylint: disable=R0801
"""
talking_video_dataset.py

This module defines the TalkingVideoDataset class, a custom PyTorch dataset 
for handling talking video data. The dataset uses video files, masks, and 
embeddings to prepare data for tasks such as video generation and 
speech-driven video animation.

Classes:
    TalkingVideoDataset

Dependencies:
    json
    random
    torch
    decord.VideoReader, decord.cpu
    PIL.Image
    torch.utils.data.Dataset
    torchvision.transforms

Example:
    from talking_video_dataset import TalkingVideoDataset
    from torch.utils.data import DataLoader

    # Example configuration for the Wav2Vec model
    class Wav2VecConfig:
        def __init__(self, audio_type, model_scale, features):
            self.audio_type = audio_type
            self.model_scale = model_scale
            self.features = features

    wav2vec_cfg = Wav2VecConfig(audio_type="wav2vec2", model_scale="base", features="feature")

    # Initialize dataset
    dataset = TalkingVideoDataset(
        img_size=(512, 512),
        sample_rate=16000,
        audio_margin=2,
        n_motion_frames=0,
        n_sample_frames=16,
        data_meta_paths=["path/to/meta1.json", "path/to/meta2.json"],
        wav2vec_cfg=wav2vec_cfg,
    )

    # Initialize dataloader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Fetch one batch of data
    batch = next(iter(dataloader))
    print(batch["pixel_values_vid"].shape)  # Example output: (4, 16, 3, 512, 512)

The TalkingVideoDataset class provides methods for loading video frames, masks, 
audio embeddings, and other relevant data, applying transformations, and preparing 
the data for training and evaluation in a deep learning pipeline.

Attributes:
    img_size (tuple): The dimensions to resize the video frames to.
    sample_rate (int): The audio sample rate.
    audio_margin (int): The margin for audio sampling.
    n_motion_frames (int): The number of motion frames.
    n_sample_frames (int): The number of sample frames.
    data_meta_paths (list): List of paths to the JSON metadata files.
    wav2vec_cfg (object): Configuration for the Wav2Vec model.

Methods:
    augmentation(images, transform, state=None): Apply transformation to input images.
    __getitem__(index): Get a sample from the dataset at the specified index.
    __len__(): Return the length of the dataset.
"""

import json
import random
from typing import List

import torch
from decord import VideoReader, cpu
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

import whisper
import os
import librosa
from ..models.whisper_local.audio2feature import load_audio_model
from transformers import CLIPImageProcessor
from transformers import AutoFeatureExtractor


def get_audio_feature(audio_path, feature_extractor):
    audio_input, sampling_rate = librosa.load(audio_path, sr=16000)
    assert sampling_rate == 16000

    audio_features = []
    window = 750*640
    for i in range(0, len(audio_input), window):
        audio_feature = feature_extractor(audio_input[i:i+window],
                                        sampling_rate=sampling_rate,
                                        return_tensors="pt",
                                        ).input_features
        audio_features.append(audio_feature)
    audio_features = torch.cat(audio_features, dim=-1)
    return audio_features, len(audio_input) // 640

class TalkingVideoDataset(Dataset):
    """
    A dataset class for processing talking video data.

    Args:
        img_size (tuple, optional): The size of the output images. Defaults to (512, 512).
        sample_rate (int, optional): The sample rate of the audio data. Defaults to 16000.
        audio_margin (int, optional): The margin for the audio data. Defaults to 2.
        n_motion_frames (int, optional): The number of motion frames. Defaults to 0.
        n_sample_frames (int, optional): The number of sample frames. Defaults to 16.
        data_meta_paths (list, optional): The paths to the data metadata. Defaults to None.
        wav2vec_cfg (dict, optional): The configuration for the wav2vec model. Defaults to None.

    Attributes:
        img_size (tuple): The size of the output images.
        sample_rate (int): The sample rate of the audio data.
        audio_margin (int): The margin for the audio data.
        n_motion_frames (int): The number of motion frames.
        n_sample_frames (int): The number of sample frames.
        data_meta_paths (list): The paths to the data metadata.
        wav2vec_cfg (dict): The configuration for the wav2vec model.
    """

    def __init__(
        self,
        img_size=(512, 512),
        sample_rate=16000,
        audio_margin=2,
        n_motion_frames=0,
        n_sample_frames=16,
        data_meta_paths=None,
        wav2vec_cfg=None,
        device='cuda',
        # ============ 新增的可选参数 ============
        align_instance=None,
        clip_area=1.25,
        clip_image_size=224,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.img_size = img_size
        self.audio_margin = audio_margin
        self.n_motion_frames = n_motion_frames
        self.n_sample_frames = n_sample_frames
        self.audio_type = wav2vec_cfg.audio_type
        self.audio_model = wav2vec_cfg.model_scale
        self.audio_features = wav2vec_cfg.features

        # whisper feature extractor
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("/yuch_ws/DH/hallo2/pretrained_models/whisper-tiny/")

        #old ver
        audio_model_path = "/yuch_ws/DH/hallo2/pretrained_models/whisper_tiny.pt"
        self.audio_guider = load_audio_model(model_path=audio_model_path, device=device)

        # 新增，用于CLIP处理和人脸对齐检测
        self.clip_processor = CLIPImageProcessor()
        self.align_instance = align_instance
        self.clip_area = clip_area
        self.clip_image_size = clip_image_size

        # loopy features
        self.num_segments = 3

        # Compute total motion frames required
        self.total_motion_frames = 4 + 8 + 16  # s * (r ** n_segments - 1) // (r - 1)
        self.total_abstract_frames = 4 * self.num_segments
        self.motion_indices_offset = np.array([-25, -21, -17, -13, -11, -9, -7, -5, -4, -3, -2, -1])

        vid_meta = []
        for data_meta_path in data_meta_paths:
            with open(data_meta_path, "r", encoding="utf-8") as f:
                vid_meta.extend(json.load(f))
        self.vid_meta = vid_meta
        self.length = len(self.vid_meta)
        self.pixel_transform = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
            ]
        )
        self.attn_transform_64 = transforms.Compose(
            [
                transforms.Resize(
                    (self.img_size[0] // 8, self.img_size[0] // 8)),
                transforms.ToTensor(),
            ]
        )
        self.attn_transform_32 = transforms.Compose(
            [
                transforms.Resize(
                    (self.img_size[0] // 16, self.img_size[0] // 16)),
                transforms.ToTensor(),
            ]
        )
        self.attn_transform_16 = transforms.Compose(
            [
                transforms.Resize(
                    (self.img_size[0] // 32, self.img_size[0] // 32)),
                transforms.ToTensor(),
            ]
        )
        self.attn_transform_8 = transforms.Compose(
            [
                transforms.Resize(
                    (self.img_size[0] // 64, self.img_size[0] // 64)),
                transforms.ToTensor(),
            ]
        )

    def augmentation(self, images, transform, state=None):
        """
        Apply the given transformation to the input images.
        
        Args:
            images (List[PIL.Image] or PIL.Image): The input images to be transformed.
            transform (torchvision.transforms.Compose): The transformation to be applied to the images.
            state (torch.ByteTensor, optional): The state of the random number generator. 
            If provided, it will set the RNG state to this value before applying the transformation. Defaults to None.

        Returns:
            torch.Tensor: The transformed images as a tensor. 
            If the input was a list of images, the tensor will have shape (f, c, h, w), 
            where f is the number of images, c is the number of channels, h is the height, and w is the width. 
            If the input was a single image, the tensor will have shape (c, h, w), 
            where c is the number of channels, h is the height, and w is the width.
        """
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __getitem__(self, index):

        try:
            # print("\n\n\n  Getting image at index {}".format(index))
            video_meta = self.vid_meta[index]
            video_path = video_meta["video_path"]
            mask_path = video_meta["mask_path"]
            lip_mask_union_path = video_meta.get("sep_mask_lip", None)
            face_mask_union_path = video_meta.get("sep_mask_face", None)
            full_mask_union_path = video_meta.get("sep_mask_border", None)
            face_emb_path = video_meta["face_emb_path"]
            audio_emb_path = video_meta[
                f"{self.audio_type}_emb_{self.audio_model}_{self.audio_features}"
            ]

            # here we use whisper instead of wav2vec
            audio_path = video_meta["video_path"].replace("videos", "audios")
            audio_path = audio_path.replace(".mp4", ".wav")
            # print(f"\n {index} audio path is {audio_path}")
            fps = 25

            audio_input, audio_len = get_audio_feature(audio_path, self.feature_extractor)



            whisper_feature = self.audio_guider.audio2feat(audio_path)
            # print("whisper feature shape :", whisper_feature.shape)
            whisper_chunks = self.audio_guider.feature2chunks(feature_array=whisper_feature, fps=fps)
            # print("whisper_chunks:", whisper_chunks.shape)
            audio_frame_num = whisper_chunks.shape[0]
            audio_fea_final = torch.Tensor(whisper_chunks)
            audio_fea_final = audio_fea_final.unsqueeze(0)
            # print(video_path[-15:-4], "audio_fea_final:", audio_fea_final.shape)


            tgt_mask_pil = Image.open(mask_path)
            video_frames = VideoReader(video_path, ctx=cpu(0))
            assert tgt_mask_pil is not None, "Fail to load target mask."
            assert (video_frames is not None and len(video_frames) > 0), "Fail to load video frames."
            video_length = len(video_frames)
            # print(f"{video_path[-15:-4]} {index} video length:", video_length)

            assert (
                video_length
                > self.n_sample_frames + self.total_motion_frames  + 2 * self.audio_margin
            )
            start_idx = random.randint(
                self.total_motion_frames ,
                video_length - self.n_sample_frames - self.audio_margin - 1,
            )
            # print(video_path[-15:-4], "start idx is ", start_idx)
            videos = video_frames[start_idx : start_idx + self.n_sample_frames]

            frame_list = [
                Image.fromarray(video).convert("RGB") for video in videos.asnumpy()
            ]

            face_masks_list = [Image.open(face_mask_union_path)] * self.n_sample_frames
            lip_masks_list = [Image.open(lip_mask_union_path)] * self.n_sample_frames
            full_masks_list = [Image.open(full_mask_union_path)] * self.n_sample_frames
            assert face_masks_list[0] is not None, "Fail to load face mask."
            assert lip_masks_list[0] is not None, "Fail to load lip mask."
            assert full_masks_list[0] is not None, "Fail to load full mask."


            face_emb = torch.load(face_emb_path)
            audio_emb = torch.load(audio_emb_path)

            # print(f"\n {index} Audio embedding shape: {audio_emb.shape}")

            indices = (
                torch.arange(2 * self.audio_margin + 1) - self.audio_margin
            )  # Generates [-2, -1, 0, 1, 2]
            center_indices = torch.arange(
                start_idx,
                start_idx + self.n_sample_frames,
            ).unsqueeze(1) + indices.unsqueeze(0)
            audio_tensor = audio_emb[center_indices]
            # print(f"\n {index} center_indices are {center_indices}")

            audio_tensor_whisper = audio_fea_final[:,start_idx:start_idx + self.n_sample_frames,:,:]
            # audio_fea_final: torch.Size([1, 243, 50, 384])
            # print("audio tensor whisper shape :", audio_tensor_whisper.shape)
            # [1,14,50,384]
            # whisper log mel
            # if not os.path.exists(audio_path):
            #     print(f"\n {index} Audio path does not exist: {audio_path}")
            #     mels = np.zeros((80, 3000))
            #     audio_start_index = -1
            #     audio_end_index = -1
            # else:
            #     audio, _ = librosa.load(audio_path, sr=16000)
            #     print(f"\n {index} Audio librosa shape is :", audio.shape)
            #     # audio_start_index = int(pick_start / 25 * 50)
            #     # audio_end_index = audio_start_index + int(random_pick_size / 25 * 50)
            #
            #     # do not cut here due to long-term audio context
            #     # print('audio_start_index=', audio_start_index)
            #     # print('audio_end_index=', audio_end_index)
            #     # audio = audio[audio_start_index:audio_end_index]
            #     # print('audio.shape=', audio.shape)
            #
            #     audio = whisper.pad_or_trim(
            #         audio.flatten())  # as least 30s. you can slide to your specific duration at the usage.
            #
            #     print(f"\n {index} After pad or trim audio shape is :", audio.shape)
            #     mels = whisper.log_mel_spectrogram(audio)
            #     print(f"\n {index} mels shape is :", mels.shape)

            ref_img_idx = random.randint(
                self.total_motion_frames,
                video_length - self.n_sample_frames - self.audio_margin - 1,
            )
            ref_img = video_frames[ref_img_idx].asnumpy()
            ref_img = Image.fromarray(ref_img)

            if self.n_motion_frames > 0:
                # motions = video_frames[start_idx - self.n_motion_frames : start_idx]
                motions = video_frames.get_batch(list(start_idx + self.motion_indices_offset))
                motion_list = [
                    Image.fromarray(motion).convert("RGB") for motion in motions.asnumpy()
                ]

            # transform
            state = torch.get_rng_state()
            pixel_values_vid = self.augmentation(frame_list, self.pixel_transform, state)

            pixel_values_mask = self.augmentation(tgt_mask_pil, self.cond_transform, state)
            pixel_values_mask = pixel_values_mask.repeat(3, 1, 1)

            pixel_values_face_mask = [
                self.augmentation(face_masks_list, self.attn_transform_64, state),
                self.augmentation(face_masks_list, self.attn_transform_32, state),
                self.augmentation(face_masks_list, self.attn_transform_16, state),
                self.augmentation(face_masks_list, self.attn_transform_8, state),
            ]
            pixel_values_lip_mask = [
                self.augmentation(lip_masks_list, self.attn_transform_64, state),
                self.augmentation(lip_masks_list, self.attn_transform_32, state),
                self.augmentation(lip_masks_list, self.attn_transform_16, state),
                self.augmentation(lip_masks_list, self.attn_transform_8, state),
            ]
            pixel_values_full_mask = [
                self.augmentation(full_masks_list, self.attn_transform_64, state),
                self.augmentation(full_masks_list, self.attn_transform_32, state),
                self.augmentation(full_masks_list, self.attn_transform_16, state),
                self.augmentation(full_masks_list, self.attn_transform_8, state),
            ]
            # print("pixel_values_ref_img shape is 1", ref_img.shape)
            pixel_values_ref_img = self.augmentation(ref_img, self.pixel_transform, state)
            # print("pixel_values_ref_img shape is 2", pixel_values_ref_img.shape)
            pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)
            # print("pixel_values_ref_img shape is 3", pixel_values_ref_img.shape)
            # print("self.n_motion_frames is ", self.n_motion_frames   )
            if self.n_motion_frames > 0:
                pixel_values_motion = self.augmentation(
                    motion_list, self.pixel_transform, state
                )
                pixel_values_ref_img = torch.cat(
                    [pixel_values_ref_img, pixel_values_motion], dim=0
                )

            audio_tensor_whisper = audio_tensor_whisper.squeeze(0)
            # print("\n audio_tensor shape is :", audio_tensor_whisper.shape)
            # print("pixel_values_vid shape is :", pixel_values_vid.shape)
            # print("pixel_values_ref_img shape is ", pixel_values_ref_img.shape)

            # ========== 新增： 将 ref_img 转换为 CLIP image =============
            # （下面仅示例如何做，具体是否要裁剪/对齐看你需求）

            clip_image = None
            if self.clip_processor is not None:
                # 如果 align_instance 存在，先做一个 bounding box 检测并裁剪
                if self.align_instance is not None:
                    w, h = ref_img.size
                    # 执行人脸检测
                    _, _, bboxes_list = self.align_instance(
                        np.array(ref_img)[:, :, [2, 1, 0]], maxface=True
                    )
                    if len(bboxes_list) > 0:
                        x1, y1, ww, hh = bboxes_list[0]
                        x2, y2 = x1 + ww, y1 + hh
                        # 根据 area 放大
                        ww, hh = (x2 - x1) * self.clip_area, (y2 - y1) * self.clip_area
                        cx, cy = (x2 + x1) // 2, (y2 + y1) // 2
                        x1 = max(cx - ww // 2, 0)
                        y1 = max(cy - hh // 2, 0)
                        x2 = min(cx + ww // 2, w)
                        y2 = min(cy + hh // 2, h)

                        # 裁剪这部分作为 clip 的原图
                        ref_img_clip = ref_img.crop((x1, y1, x2, y2))
                    else:
                        ref_img_clip = ref_img
                else:
                    # 如果没有人脸检测，就直接用整张图
                    ref_img_clip = ref_img

                # 再把图 resize 到合适大小（224×224）做 CLIP 处理
                ref_img_clip = ref_img_clip.resize((self.clip_image_size, self.clip_image_size), Image.LANCZOS)
                clip_image = self.clip_processor(images=ref_img_clip, return_tensors="pt").pixel_values[0]
            # ========== 新增完毕 ==========

            # print(video_path[-15:-4] ,"clip_image shape", clip_image.shape)
            # print(video_path[-15:-4], "audio_tensor shape ", audio_input[0].shape)
            # print(video_path[-15:-4], "audio_tensor_whisper_old shape", audio_tensor_whisper.shape)
            # print(video_path[-15:-4], "audio_len", audio_len)
            # lSteele_000 audio_fea_final: torch.Size([1, 327, 50, 384])
            # lSteele_000 clip_image shape torch.Size([3, 224, 224])
            # lSteele_000 audio_tensor shape  torch.Size([80, 3000])
            # lSteele_000 audio_tensor_whisper_old shape torch.Size([25, 50, 384])
            # lSteele_000 audio_len 325
            from transformers import WhisperModel
            wav_enc = WhisperModel.from_pretrained("/yuch_ws/DH/hallo2/pretrained_models/whisper-tiny/").to(device="cuda").eval()
            wav_enc.requires_grad_(False)

            audio_feature = audio_input[0]
            audio_feature = audio_feature.unsqueeze(0).to(device="cuda")
            # print(video_path[-15:-4],"audio_feature unsqueezed : ", audio_feature.shape)
            # lSteele_000 audio_feature unsqueezed :  torch.Size([1, 80, 3000])
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

            # print(video_path[-15:-4], "audio_prompts shape", audio_prompts.shape)
            # print(video_path[-15:-4], "last_audio_prompts", last_audio_prompts.shape)
            # lSteele_000 audio_prompts shape torch.Size([1, 660, 5, 384])
            # lSteele_000 last_audio_prompts torch.Size([1, 700, 1, 384])

            center_indices = np.arange(
                start_idx,
                start_idx + self.n_sample_frames,
            )
            # print(video_path[-15:-4], "center_indices : ", center_indices)
            # rowley1_003 center_indices :  [62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86]

            audio_clips = []
            audio_clips_for_bucket = []
            for i in center_indices:
                # we shift window to cover past info
                audio_clip_start_idx = i * 2  -4
                audio_clip = audio_prompts[:, audio_clip_start_idx:audio_clip_start_idx+10]
                # we shift window to cover past info
                audio_clip_for_bucket_start_idx = i * 2  - 24
                audio_clip_for_bucket = last_audio_prompts[:, audio_clip_for_bucket_start_idx:audio_clip_for_bucket_start_idx + 50]
                audio_clips.append(audio_clip)
                audio_clips_for_bucket.append(audio_clip_for_bucket)
            # print(video_path[-15:-4], "audio_clip", audio_clip.shape)
            # print(video_path[-15:-4], "audio_clip_for_bucket", audio_clip_for_bucket.shape)
            # rowley1_003 audio_clip torch.Size([1, 10, 5, 384])
            # rowley1_003 audio_clip_for_bucket torch.Size([1, 50, 1, 384])
            audio_clips = torch.cat(audio_clips, dim=0)
            audio_clips_for_bucket = torch.cat(audio_clips_for_bucket, dim=0)
            # print(video_path[-15:-4], "audio_clips", audio_clips.shape)
            # print(video_path[-15:-4], "audio_clips_for_bucket", audio_clips_for_bucket.shape)
            # rowley1_003 audio_clips torch.Size([25, 10, 5, 384])
            # rowley1_003 audio_clips_for_bucket torch.Size([25, 50, 1, 384])

                # motion_bucket = audio2bucket(audio_clip_for_bucket, image_embeds)
                # motion_bucket = motion_bucket * 16 + 16
                # motion_buckets.append(motion_bucket[0])
                #
                # cond_audio_clip = audio_pe(audio_clip).squeeze(0)
                # uncond_audio_clip = audio_pe(torch.zeros_like(audio_clip)).squeeze(0)
                #
                # ref_tensor_list.append(ref_img[0])
                # audio_tensor_list.append(cond_audio_clip[0])
                # uncond_audio_tensor_list.append(uncond_audio_clip[0])

            sample = {
                "start_idx": start_idx,
                "video_dir": video_path,
                "pixel_values_vid": pixel_values_vid,
                "pixel_values_mask": pixel_values_mask,
                "pixel_values_face_mask": pixel_values_face_mask,
                "pixel_values_lip_mask": pixel_values_lip_mask,
                "pixel_values_full_mask": pixel_values_full_mask,
                "audio_clips" : audio_clips,
                "audio_clips_for_bucket" : audio_clips_for_bucket,
                "audio_tensor": audio_tensor_whisper,
                # "audio_feature": audio_input[0],
                # "audio_len": audio_len,
                "pixel_values_ref_img": pixel_values_ref_img,
                "face_emb": face_emb,
                "clip_images": clip_image,
            }

            return sample

        except Exception as e:
            ## AssertionError: Some clips are too short for loopy motion frames
            ## RuntimeError: PytorchStreamReader failed reading zip archive: not a ZIP archive for /yuch_ws/DH_Data/hallo_dy_michael/videos/钢铁战士51029_7413034826653109504_video_00000_0.mp4
            ## FileNotFoundError: some hubert features are failed to extract due to CUDA OOM
            print(f"## Caught error in __getitem__: {e} for {video_path}")
            # If assertion fails, move to the next item
            if index + 1 >= len(self):
                raise StopIteration("Reached end of dataset")
            return self.__getitem__(index + 1)

    def __len__(self):
        return len(self.vid_meta)
