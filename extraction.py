import librosa
import numpy as np
import os

# Step1: input all files of the speakers
root_path = r"D:\JerryHuang\作业\大三下半学期\Iot Security\lab1_ voice identity\pythonProject2\dev-clean"

audio_path = []
audio_label = []

#go through the all speaker path
for speaker_dir in os.listdir(root_path):
    speaker_dir_path = os.path.join(root_path, speaker_dir)

    #insure the path is correct
    if os.path.isdir(speaker_dir_path):
        print(f"Proccessing speaker: {speaker_dir_path}")

        #Go through all book path in the speaker path
        for book_path in os.listdir(speaker_dir_path):
            book_path_dir = os.path.join(speaker_dir_path, book_path)

            if os.path.isdir(book_path_dir):
                print(f"Processing book: {book_path_dir}")

                #go through all files in the book path of each speakers
                for voice_file in os.listdir(book_path_dir):
                    if voice_file.endswith(".flac"):
                        voice_file_path = os.path.join(book_path_dir, voice_file)

                        audio_path.append(voice_file_path)
                        audio_label.append(speaker_dir)

print(f"Total audio files: {len(audio_path)}")
print(f"Labels: {set(audio_label)}")

import random

# 随机抽取5个样本
sample_indices = random.sample(range(len(audio_path)), 5)
for i in sample_indices:
    print(f"Audio Path: {audio_path[i]}, Label: {audio_label[i]}")

print(" ")
# Step 2: Perform feature extraction on the extracted audio files
features = []
labels = []

for path, label in zip(audio_path, audio_label):
    # load the audio files
    y, sr = librosa.load(path, sr=None)

    # extract the MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    print("MFCC extracted")

    # extract the Pitch feature
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    mean_pitches = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    print("Pitches extracted")

    # extract other features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    print("Chroma extracted")
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    print("Spectral Contrast extracted")

    feature_vector = np.hstack([
        np.mean(mfccs, axis=1),
        mean_pitches,
        np.mean(chroma, axis=1),
        np.mean(spectral_contrast, axis=1)
    ])
    print("Feature vector extracted")

    features.append(feature_vector)
    labels.append(label)

    print("Feature load successfully")

features = np.array(features)
labels = np.array(labels)


print(f"Extracted features shape: {features.shape}")
print(f"Label shape: {labels.shape}")

# 保存特征和标签到本地
np.save('features.npy', features)  # 保存特征
np.save('labels.npy', labels)      # 保存标签
print("Features and labels saved as 'features.npy' and 'labels.npy'.")