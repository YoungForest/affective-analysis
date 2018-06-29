from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
import itertools

class LirisDataset(Dataset):
    """Load liris database from json output by video-action classification
    """
    
    def __init__(self, json_file, root_dir, ranking_file, sets_file, train=True, sep=',', transform=None):
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
        for d in self.data_json:
            l = len(d['clips'])
            if mi > l:
                mi = l
        self.mi = mi

        self.transform = transform
        self.root_dir = root_dir
        
        self.scores = pd.read_csv(ranking_file, sep=sep)

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
            dataset = [x for x in self.data_json if self.sets[self.sets['name']!=x['video']].iloc[0]['set']!=0]
                
        return dataset

    def __len__(self):
        return len(self.dataset)

    def getitem_without_labels(self, idx):
        sample = self.dataset[idx]
        sample['input'] = []

        if self.transform:
            for s in sample['clips'][0:self.mi]:
                sample['input'] += s['features']
        else:
            for s in sample['clips']:
                sample['input'] += s['features']

        return sample
        
    def __getitem__(self, idx):
        sample = self.getitem_without_labels(idx)

        sample['valenceValue'] = self.scores[self.scores['name']==sample['video']]['valenceValue'].iloc[0]
        sample['arousalValue'] = self.scores[self.scores['name']==sample['video']]['arousalValue'].iloc[0]

        return {'input': torch.from_numpy(np.array(sample['input'])).float(), 'labels': torch.from_numpy(np.array([sample['valenceValue'], sample['arousalValue']])).float}

if __name__ == '__main__':
    liris = LirisDataset(json_file='output-liris-resnet-34-kinetics.json', root_dir='/home/data_common/data_yangsen/data', transform=True, ranking_file='ACCEDEranking.txt', sets_file='ACCEDEsets.txt', sep='\t')

    for i in range(5):
        print(liris[i])
