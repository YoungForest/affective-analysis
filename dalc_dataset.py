from __future__ import print_function, division
import os
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
from liris_dataset import LirisDataset

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

        return sample

if __name__ == '__main__':
    dalc = DalcDataset(json_file='output-resnet-34-kinetics.json', root_dir='/home/data_common/data_yangsen/videos', transform=True, ranking_file='filled-labels_features.csv', sep=',')

    for i in range(5):
        print(dalc[i])
