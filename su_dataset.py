from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import glob
import librosa
import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

class SuDataset(Dataset):
    """
    convert suyiming's affective videos dataset to PyTorch version
    """
    def __init__(self, audio_root_dir, labels_file):
        self.emotion = pd.read_csv(labels_file, sep='\t', header=0, names=['arousal', 'excitement', 'pleasure', 'contentment', 'sleepiness', 'depression', 'misery', 'distress'])
        min_max_scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
        self.emotion[self.emotion.columns] = min_max_scaler.fit_transform(self.emotion[self.emotion.columns])
       
        audios = glob.glob(audio_root_dir + "/*.wav")
        audio_max_length = 0
        json_data = []

        for audio in audios:
            result = {}

            file_name = os.path.splitext(audio)[0].split(os.sep)[-1]
            result['id'] = int(file_name)
            y, sr = librosa.load(audio)
            mfcc = librosa.feature.mfcc(y, sr=sr)

            result['mfcc'] = mfcc

            audio_max_length = max(audio_max_length, mfcc.shape[1]) 

            json_data.append(result)

        self.data = json_data
        self.audio_max_length =audio_max_length

       

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        labels = torch.from_numpy(np.array(self.emotion.iloc[idx])).float()
        return {'id': self.data[idx]['id'], 'labels': labels, 'mfcc': torch.from_numpy(np.concatenate((self.data[idx]['mfcc'], np.zeros((20, self.audio_max_length - self.data[idx]['mfcc'].shape[1]))), axis=1)).float()} 


if __name__ == '__main__':
    suyiming = SuDataset('../audios', 'emotion.txt')
