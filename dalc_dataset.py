from __future__ import print_function, division
import os
import torch
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
from liris_dataset import LirisDataset
import numpy as np

class DalcDataset(LirisDataset):
    """Load DaLC affective analysis database(https://github.com/sparkingarthur/DaLC)
    """
    def __getitem__(self, idx):
        sample = self.getitem_without_labels(idx)

        name_without_extension = os.path.splitext(sample['video'])[0]
        sample['arousal'] = self.scores[self.scores['id']==name_without_extension]['arousal'].iloc[0]
        sample['excitement'] = self.scores[self.scores['id']==name_without_extension]['excitement'].iloc[0]
        sample['pleasure'] = self.scores[self.scores['id']==name_without_extension]['pleasure'].iloc[0]
        sample['contentment'] = self.scores[self.scores['id']==name_without_extension]['contentment'].iloc[0]
        sample['sleepiness'] = self.scores[self.scores['id']==name_without_extension]['sleepiness'].iloc[0]
        sample['depression'] = self.scores[self.scores['id']==name_without_extension]['depression'].iloc[0]
        sample['misery'] = self.scores[self.scores['id']==name_without_extension]['misery'].iloc[0]
        sample['distress'] = self.scores[self.scores['id']==name_without_extension]['distress'].iloc[0]

        return {'input': torch.from_numpy(np.array(sample['input'])).float(), 'labels': torch.from_numpy(np.array([sample['arousal'], sample['excitement'], sample['pleasure'], sample['contentment'], sample['sleepiness'], sample['depression'], sample['misery'], sample['distress']], dtype=float)).float()}

    def get_train_or_test(self, train, sets_file):
        self.sets = pd.read_csv(sets_file)
        test_id = 8
        self.scores['cv_id'] = self.sets['cv_id_10']

        dataset = None
        if not train:
            dataset = [x for x in self.data_json if self.scores[self.scores['id']==os.path.splitext(x['video'])[0]].iloc[0]['cv_id'] == test_id and 'clips' in x.keys()]
        else:
            dataset = [x for x in self.data_json if self.scores[self.scores['id']==os.path.splitext(x['video'])[0]].iloc[0]['cv_id'] != test_id and 'clips' in x.keys()]

        return dataset

if __name__ == '__main__':
    dalc = DalcDataset(json_file='output-resnet-34-kinetics.json', root_dir='/home/data_common/data_yangsen/videos', transform=True, ranking_file='filled-labels_features.csv', sets_file='cv_id_10.txt', sep=',')

    for i in range(5):
        print(dalc[i])
