import librosa
import numpy as np
import os
import pandas as pd


def extractSingleFeature(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    print("MFCC extracted")

    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    mean_pitches = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    print("Pitches extracted")

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
    return feature_vector


def extraction(audio_path, audio_label = None) :
    """
       从音频文件中提取特征。

       参数:
           audio_path (str or list): 单个音频文件路径或音频文件路径列表。
           audio_label (int or list): 单个标签或标签列表。如果为 None，则返回的特征不包含标签。

       返回:
           features (np.ndarray): 提取的特征数组。
           labels (np.ndarray): 对应的标签数组（如果 audio_label 不为 None）。
       """

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

    return features, labels

def file_loader(root_path):

    audio_path = []
    audio_label = []

    # go through the all speaker path
    for speaker_dir in os.listdir(root_path):
        speaker_dir_path = os.path.join(root_path, speaker_dir)

        # insure the path is correct
        if os.path.isdir(speaker_dir_path):
            print(f"Proccessing speaker: {speaker_dir_path}")

            # Go through all book path in the speaker path
            for book_path in os.listdir(speaker_dir_path):
                book_path_dir = os.path.join(speaker_dir_path, book_path)

                if os.path.isdir(book_path_dir):
                    print(f"Processing book: {book_path_dir}")

                    # go through all files in the book path of each speakers
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

    return audio_path, audio_label


def check(audio_path, audio_label):
    import random

    # 随机抽取5个样本
    sample_indices = random.sample(range(len(audio_path)), 20)
    for i in sample_indices:
        print(f"Audio Path: {audio_path[i]}, Label: {audio_label[i]}")

def extraction_main(root_path):
    audio_path, audio_label = file_loader(root_path)

    check(audio_path, audio_label)
    features, labels = extraction(audio_path, audio_label)

    print(f"Extracted features shape: {features.shape}")
    print(f"Label shape: {labels.shape}")

    # Assuming features and labels are numpy arrays
    features = np.array(features)
    labels = np.array(labels)

    # Convert to DataFrame
    df = pd.DataFrame(features)
    df['label'] = labels

    file_name = 'features_labels.csv'
    # Save to CSV
    df.to_csv(file_name, index=False)
    print(f"Features and labels saved as '{file_name}'.")

    return file_name