from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
import itertools
import torch
import numpy as np
import os
import librosa
import pickle

class LirisDataset(Dataset):
    """Load liris database from json output by video-action classification
    """
    
    def __init__(self, json_file, root_dir, audio_root_dir, ranking_file, sets_file, train=True, sep=',', transform=None):
        """
        Args:
            json_file (string): Path to the json file outputed from pre-trained model.
            root_dir (string): Directory with all the videos.
            ranking_file (string): Path to the csv file with annotations.
            transfrom (bool): Whether or not to uniform data.
        """
        data = None
        with open(json_file, 'r') as myfile:
            data = myfile.read()
        self.data_json = json.loads(data)
        mi = 1000
        ma = 0
        for d in self.data_json:
            l = len(d['clips'])
            if mi > l:
                mi = l
            if ma < l:
                ma = l
        self.mi = mi
        self.ma = ma

        self.transform = transform
        self.root_dir = root_dir
        self.audio_root_dir = audio_root_dir
        
        self.scores = pd.read_csv(ranking_file, sep=sep)
        
        audio_max_length = 0
        # get audio information
        for video in self.data_json:
            file_name = os.path.splitext(video['video'])[0]
            file_path = os.path.join(self.audio_root_dir, file_name + '.wav')
            y, sr = librosa.load(file_path)
            mel = librosa.feature.melspectrogram(y)

            video['mel'] = mel

            if audio_max_length < mel.shape[1]:
                audio_max_length = mel.shape[1]
        
        self.audio_max_length = audio_max_length

        # train or test
        self.dataset = self.get_train_or_test(train, sets_file)
        
    def get_input_dim(self):
        return len(self.__getitem__(0)['input'])
        
    def get_train_or_test(self, train, sets_file):
        self.sets = pd.read_csv(sets_file, sep='\t')
        dataset = None
        if not train:
            dataset = [x for x in self.data_json if self.sets[self.sets['name']==x['video']].iloc[0]['set']==0]
        else:
            dataset = [x for x in self.data_json if self.sets[self.sets['name']==x['video']].iloc[0]['set']!=0]
                
        return dataset

    def __len__(self):
        return len(self.dataset)

    def getitem_without_labels(self, idx):
        sample = self.dataset[idx]
        sample['input'] = []

        for s in sample['clips']:
            sample['input'] += s['features']

        # uniform the input length
        # align all the features to the left and pad to the right with zeros
        if self.transform:
            for i in range(len(sample['clips']), self.ma):
                sample['input'] += [0] * len(sample['clips'][0]['features'])

        return sample
        
    def __getitem__(self, idx):
        sample = self.getitem_without_labels(idx)

        sample['valenceValue'] = self.scores[self.scores['name']==sample['video']]['valenceValue'].iloc[0]
        sample['arousalValue'] = self.scores[self.scores['name']==sample['video']]['arousalValue'].iloc[0]

        return {'video': sample['video'], 'input': torch.from_numpy(np.array(sample['input'])).float(), 'labels': torch.from_numpy(np.array([sample['valenceValue'], sample['arousalValue']])).float(), 'mel': torch.from_numpy(np.concatenate((sample['mel'], np.zeros((128, self.audio_max_length - sample['mel'].shape[1]))), axis=1)).float()}

def getLirisDataset(path, train=True):
    dataset = None
    if os.path.exists(path):
        with open(path, 'rb') as my_file:
            dataset = pickle.load(my_file)
    else:
        dataset = LirisDataset(json_file='output-resnet-101-kinetics.json', root_dir='/home/data_common/data_yangsen/data', audio_root_dir='/home/data_common/data_yangsen/audio', train=train, transform=True, ranking_file='ACCEDEranking.txt', sets_file='ACCEDEsets.txt', sep='\t')
        with open(path, 'wb') as out:
            pickle.dump(dataset, out, pickle.HIGHEST_PROTOCOL) 
    assert dataset

    return dataset

if __name__ == '__main__':
    liris = LirisDataset(json_file='output-liris-resnet-34-kinetics.json', root_dir='/home/data_common/data_yangsen/data', audio_root_dir='/home/data_common/data_yangsen/audio', transform=True, ranking_file='ACCEDEranking.txt', sets_file='ACCEDEsets.txt', sep='\t')

    for i in range(5):
        print(liris[i])
