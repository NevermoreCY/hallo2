from whisper_local.audio2feature import load_audio_model

import whisper
import os
import numpy as np
import librosa
import torch

audio_model_path = "/yuch_ws/DH/EchoMimic/pretrained_weights/audio_processor/whisper_tiny.pt"
audio_path = "/yuch_ws/DH_Data/hallo_liutao/audios/chdtf_num_7251d00f1f45252a75acb1b62918f961_025.wav"

#  4676 video length: 241
#
#  4676 Audio embedding shape: torch.Size([242, 12, 768])
#
#  4676 center_indices are tensor([[77, 78, 79, 80, 81],
#         [78, 79, 80, 81, 82],
#         [79, 80, 81, 82, 83],
#         [80, 81, 82, 83, 84],
#         [81, 82, 83, 84, 85],
#         [82, 83, 84, 85, 86],
#         [83, 84, 85, 86, 87],
#         [84, 85, 86, 87, 88],
#         [85, 86, 87, 88, 89],
#         [86, 87, 88, 89, 90],
#         [87, 88, 89, 90, 91],
#         [88, 89, 90, 91, 92],
#         [89, 90, 91, 92, 93],
#         [90, 91, 92, 93, 94]])
#
#  4676 Audio librosa shape is : (154624,)
#
#  4676 After pad or trim audio shape is : (480000,)
#
#  4676 mels shape is : torch.Size([80, 3000])

#  encider shape0 (1, 5, 1500, 384)
# encider shape1 (1, 1500, 5, 384)
# encider shape2 (1500, 5, 384)
# start idx 0 end idx 966 emb end idx 483
# concatenated array (483, 5, 384)
# whisper feature shape : (483, 5, 384)
# video in 25 FPS, audio idx in 50FPS
# whisper_chunks: (243, 50, 384)
# audio_fea_final: torch.Size([1, 243, 50, 384])
#
#   Audio librosa shape is : (154624,)
#
#   After pad or trim audio shape is : (480000,)
#
#  mels shape is : torch.Size([80, 3000])


fps = 25
# echo mimic :
audio_guider = load_audio_model(model_path=audio_model_path,device='cuda')

whisper_feature = audio_guider.audio2feat(audio_path)
print("whisper feature shape :", whisper_feature.shape)
whisper_chunks = audio_guider.feature2chunks(feature_array=whisper_feature, fps=fps)
print("whisper_chunks:", whisper_chunks.shape)
audio_frame_num = whisper_chunks.shape[0]
audio_fea_final = torch.Tensor(whisper_chunks)
audio_fea_final = audio_fea_final.unsqueeze(0)
print("audio_fea_final:", audio_fea_final.shape)


# liu


if not os.path.exists(audio_path):
    # print(f"Audio path does not exist: {audio_path}")
    mels = np.zeros((80, 3000))
    audio_start_index = -1
    audio_end_index = -1
else:
    audio, _ = librosa.load(audio_path, sr=16000)
    print(f"\n  Audio librosa shape is :", audio.shape)
    # audio_start_index = int(pick_start / 25 * 50)
    # audio_end_index = audio_start_index + int(random_pick_size / 25 * 50)

    # do not cut here due to long-term audio context
    # print('audio_start_index=', audio_start_index)
    # print('audio_end_index=', audio_end_index)
    # audio = audio[audio_start_index:audio_end_index]
    # print('audio.shape=', audio.shape)

    audio = whisper.pad_or_trim(
        audio.flatten())  # as least 30s. you can slide to your specific duration at the usage.

    print(f"\n  After pad or trim audio shape is :", audio.shape)
    mels = whisper.log_mel_spectrogram(audio)
    print(f"\n mels shape is :", mels.shape)



whisper_encoder = whisper.load_model(name='tiny',  device='cpu').encoder
whisper_encoder = whisper_encoder.to(device="cuda")

audio_feats = torch.Tensor(mels)
# whisper feature
with torch.autocast("cuda"):
    _, audio_embs = whisper_encoder(audio_feats)

    print("0 audio_embds shape is ", audio_embs.shape)

    audio_embs = [item.unsqueeze(1) for item in audio_embs]  # (B, 6, T, 512)

    # print(" 1 audio_embds shape is ", audio_embs.)

    audio_embs = torch.cat(audio_embs, dim=1)

    print(" 1 audio_embds shape is ", audio_embs.shape)

    batch_size = audio_embs.shape[0]

    # print('[init]audio_embs.shape', audio_embs.shape)
    # 因为这里是一个序列，所以就不用像 stage1 那样用上下文了，这里上下文已经有了
    sliced_audio_feats = []

    # for i in range(batch_size):
    #     start_idx = int(audio_start_index[i])
    #     end_idx = int(audio_end_index[i])
    #     if start_idx == -1 or end_idx == -1:
    #         sliced = torch.zeros(1, 6, int(12 / 25 * 50), 512).to(device=audio_feats.device, dtype=audio_feats.dtype)
    #     else:
    #         sliced = audio_embs[i:i + 1, :, start_idx:end_idx]
    #     sliced_audio_feats.append(sliced)
    #
    # audio_embs = torch.cat(sliced_audio_feats, dim=0)

    # print('[sliced]audio_embs.shape', audio_embs.shape)