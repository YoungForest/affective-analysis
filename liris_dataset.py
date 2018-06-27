from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd

class LirisDataset(Dataset):
    """Load liris database from json output by video-action classification
    """
    
    def __init__(self, json_file, root_dir, ranking_file, transform=None):
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
        
        self.scores = pd.read_csv(ranking_file, sep='\t')
        
    def __len__(self):
        return len(self.data_json)

    def __getitem__(self, idx):
        sample = self.data_json[idx]

        if self.transform:
            sample['clips'] = sample['clips'][0:self.mi]
        
        sample['valenceValue'] = self.scores[self.scores['name']==sample['video']]['valenceValue'].iloc[0]
        sample['arousalValue'] = self.scores[self.scores['name']==sample['video']]['arousalValue'].iloc[0]

        return sample

liris = LirisDataset(json_file='output-liris-resnet-34-kinetics.json', root_dir='/home/data_common/data_yangsen/data', transform=True, ranking_file='ACCEDEranking.txt')

for i in range(5):
    print(liris[i])
