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
from transformers import CLIPImageProcessor

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

            motion_mask = True

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
            tgt_mask_pil = Image.open(mask_path)
            video_frames = VideoReader(video_path, ctx=cpu(0))
            assert tgt_mask_pil is not None, "Fail to load target mask."
            assert (video_frames is not None and len(video_frames) > 0), "Fail to load video frames."
            video_length = len(video_frames)


            if motion_mask:

                assert (
                        video_length
                        > self.n_sample_frames + 2 * self.audio_margin
                )
                # we can start with 0, just use mask
                start_idx = random.randint(
                    self.audio_margin,
                    video_length - self.n_sample_frames - self.audio_margin - 1,
                )

            else:

                assert (
                        video_length
                        > self.n_sample_frames + self.total_motion_frames + 2 * self.audio_margin
                )
                start_idx = random.randint(
                    self.total_motion_frames,
                    video_length - self.n_sample_frames - self.audio_margin - 1,
                )

            # extract video frames
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
            indices = (
                torch.arange(2 * self.audio_margin + 1) - self.audio_margin
            )  # Generates [-2, -1, 0, 1, 2]
            center_indices = torch.arange(
                start_idx,
                start_idx + self.n_sample_frames,
            ).unsqueeze(1) + indices.unsqueeze(0)



            audio_tensor = audio_emb[center_indices]

            ref_img_idx = random.randint(
                self.total_motion_frames,
                video_length - self.n_sample_frames - self.audio_margin - 1,
            )
            ref_img = video_frames[ref_img_idx].asnumpy()
            ref_img = Image.fromarray(ref_img)

            zero_img = Image.new("RGB", (self.img_size[0], self.img_size[1]))

            test_out_dir = "/yuch_ws/DH/hallo2/test_dir/"
            video_name = video_path.split("/")[-1].split(".")[0]
            if self.n_motion_frames > 0:
                # motions = video_frames[start_idx - self.n_motion_frames : start_idx]

                if motion_mask:
                    actual_indices = list(start_idx + self.motion_indices_offset)
                    motion_list = []
                    for ind in actual_indices:
                        out_path = test_out_dir + f"{video_name}_{ind}.png"
                        if ind < 0: # we use mask for this case
                            motion_list.append(zero_img)
                            # zero_img.save(out_path)
                            # print("saved motion frames to " + out_path)
                        else:
                            motion = video_frames[ind].asnumpy()
                            motion = Image.fromarray(motion)
                            # if ind < 10:
                            #     motion.save(out_path)
                                # print("saved motion frames to " + out_path)
                            motion_list.append(motion)

                else:
                    # actual_indices = list(start_idx + self.motion_indices_offset)

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

            pixel_values_ref_img = self.augmentation(ref_img, self.pixel_transform, state)
            pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)
            # print(pixel_values_ref)
            # if self.n_motion_frames > 0:
            #     pixel_values_motion = self.augmentation(
            #         motion_list, self.pixel_transform, state
            #     )
            #     pixel_values_ref_img = torch.cat(
            #         [pixel_values_ref_img, pixel_values_motion], dim=0
            #     )

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

            print("clip_image shape", clip_image.shape)


            sample = {
                "video_dir": video_path,
                "pixel_values_vid": pixel_values_vid,
                "pixel_values_mask": pixel_values_mask,
                "pixel_values_face_mask": pixel_values_face_mask,
                "pixel_values_lip_mask": pixel_values_lip_mask,
                "pixel_values_full_mask": pixel_values_full_mask,
                "audio_tensor": audio_tensor,
                "pixel_values_ref_img": pixel_values_ref_img,
                "face_emb": face_emb,
                "clip_image_ref": clip_image,
            }

            # # 若有 clip_image，额外塞进 sample
            # if clip_image is not None:
            #     sample["clip_image_ref"] = clip_image

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
