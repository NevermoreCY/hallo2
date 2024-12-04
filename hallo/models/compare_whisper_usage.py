from whisper_local.audio2feature import load_audio_model

import whisper
import os
import numpy as np
import librosa
import torch

audio_model_path = "/yuch_ws/DH/EchoMimic/pretrained_weights/audio_processor/whisper_tiny.pt"
audio_path = "/yuch_ws/DH_Data/hallo_liutao/audios/chdtf_num_7251d00f1f45252a75acb1b62918f961_025.wav"

fps = 25
# echo mimic :
audio_guider = load_audio_model(model_path=audio_model_path)

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