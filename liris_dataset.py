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
from sklearn.preprocessing import MinMaxScaler
import movies
from functools import reduce
import torchaudio


class LirisDataset(Dataset):
    """Load liris database from json output by video-action classification
    """

    def __init__(self, json_file, root_dir, ranking_file, sets_file, window_size=1, train=True, validate=False, sep=',', transform=None):
        """
        Args:
            json_file (string): Path to the json file outputed from pre-trained model.
            root_dir (string): Directory with all the videos.
            ranking_file (string): Path to the csv file with annotations.
            window_size (int): How many clips does a item contain? default: 1
            transfrom (bool): Whether or not to uniform data.
        """
        data = None
        with open(json_file, 'r') as myfile:
            data = myfile.read()
        self.data_json = json.loads(data)
        mi = 1000
        ma = 0
        self.audio = {}
        self.audio_min_length = 1000000
        self.audio_max_length = 0
        for d in self.data_json:
            l = len(d['clips'])
            if mi > l:
                mi = l
            if ma < l:
                ma = l

            # audio preprocessing
            # https://github.com/pytorch/tutorials/blob/master/beginner_source/audio_classifier_tutorial.py
            video_name = d['video']
            file_name, _ = os.path.splitext(video_name)
            audio_name = file_name + ".wav"
            sound, _ = torchaudio.load(os.path.join('/data/LIRIS-ACCEDE/LIRIS-ACCEDE-data/data/audio', audio_name), normalization=True)
            length = sound.shape[1]
            self.audio_max_length = max(self.audio_max_length, length)
            self.audio_min_length = min(self.audio_min_length, length)
            self.audio[video_name] = sound[0]
        self.mi = mi
        self.ma = ma

        self.transform = transform
        self.root_dir = root_dir

        self.scores = pd.read_csv(ranking_file, sep=sep)

        min_max_scaler = MinMaxScaler(feature_range=(-1, 1), copy=False)

        valenceRank = self.scores[['valenceRank']].values.astype(float)
        min_max_scaler.fit_transform(valenceRank)
        self.scores['valenceRankRescaled'] = pd.DataFrame(valenceRank)

        arousalRank = self.scores[['arousalRank']].values.astype(float)
        min_max_scaler.fit_transform(arousalRank)
        self.scores['arousalRankRescaled'] = pd.DataFrame(arousalRank)

        # train or test
        self.dataset = self.get_train_or_test(train, validate, sets_file)

        self.window_size = window_size
        self.remain_clips = list(
            reduce(
                lambda x, y: x + y,
                reduce(
                    lambda x, y: x + y,
                    map(
                        lambda x: x[1].get_continuous_gourp(self.window_size),
                        movies.movie_map.items()
                    ),
                    []
                ),
                []
            )
        )
        # print(self.remain_clips)
        self.clip_feature_map = {}
        for x in self.data_json:
            self.clip_feature_map[x['video']] = x

    def get_input_dim(self):
        return len(self.__getitem__(0)['input'])

    def get_train_or_test(self, train, validate, sets_file):
        self.sets = pd.read_csv(sets_file, sep='\t')
        dataset = None
        if not train:
            dataset = [x for x in self.data_json if self.sets[self.sets['name']
                                                              == x['video']].iloc[0]['set'] == 0]
        else:
            if validate:
                dataset = [x for x in self.data_json if self.sets[self.sets['name']
                                                                  == x['video']].iloc[0]['set'] == 2]
            else:
                dataset = [x for x in self.data_json if self.sets[self.sets['name']
                                                                  == x['video']].iloc[0]['set'] == 2]

        return dataset

    def __len__(self):
        return len(self.remain_clips)

    def getitem_without_labels(self, idx):
        clip = self.remain_clips[idx]
        name = clip.name
        sample = self.clip_feature_map[name]
        sample['input'] = []
        sample['audio'] = self.audio[name][-self.audio_min_length:]

        # align feature
        for i in range(min(len(sample['clips']), self.mi)):
            s = sample['clips'][i]
            sample['input'] += s['features']

        # uniform the input length
        # align all the features to the left and pad to the right with zeros
        # if self.transform:
        #     for _ in range(len(sample['clips']), self.ma):
        #         sample['input'] += [0] * len(sample['clips'][0]['features'])

        return sample

    def __getitem__(self, idx):
        sample = self.getitem_without_labels(idx)

        sample['valenceScore'] = self.scores[self.scores['name']
                                             == sample['video']]['valenceRankRescaled'].iloc[0]
        sample['arousalScore'] = self.scores[self.scores['name']
                                             == sample['video']]['arousalRankRescaled'].iloc[0]

        return {'video': sample['video'], 'audio': sample['audio'], 'input': torch.from_numpy(np.array(sample['input'])).float(), 'labels': torch.from_numpy(np.array([sample['valenceScore'], sample['arousalScore']])).float()}


def getLirisDataset(path, train=True, validate=False):
    dataset = None
    if os.path.exists(path):
        with open(path, 'rb') as my_file:
            dataset = pickle.load(my_file)
    else:
        dataset = LirisDataset(json_file='output-liris-resnet-34-kinetics.json', root_dir=movies.data_path,
                               transform=True, ranking_file=movies.ranking_file, sets_file=movies.sets_file, sep='\t')
        with open(path, 'wb') as out:
            pickle.dump(dataset, out, pickle.HIGHEST_PROTOCOL)
    assert dataset

    return dataset


def getDataLoader():
    batch_size = 128
    # Load dataset
    trainset = getLirisDataset(
        '/home/data_common/data_yangsen/pkl/liris-accede-train-dataset-mfcc-resnext101.pkl', train=True)
    validateset = getLirisDataset(
        '/home/data_common/data_yangsen/pkl/liris-accede-validate-dataset-mfcc-resnext101.pkl', train=True, validate=True)
    train_validateset = torch.utils.data.ConcatDataset([trainset, validateset])
    trainloader = torch.utils.data.DataLoader(
        train_validateset, batch_size=batch_size, shuffle=True)

    testset = getLirisDataset(
        '/home/data_common/data_yangsen/pkl/liris-accede-test-dataset-mfcc-resnext101.pkl', train=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader


if __name__ == '__main__':
    liris = LirisDataset(json_file='output-liris-resnet-34-kinetics.json', root_dir=movies.data_path,
                         transform=True, ranking_file=movies.ranking_file, sets_file=movies.sets_file, sep='\t')

    for i in range(5):
        print(liris[i])
