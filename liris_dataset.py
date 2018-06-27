from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd

class LirisDataset(Dataset):
    """Load liris database from json output by video-action classification
    """
    
    def __init__(self, json_file, root_dir, ranking_file, sep=',', transform=None):
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
        
    def __len__(self):
        return len(self.data_json)

    def getitem_without_labels(self, idx):
        sample = self.data_json[idx]

        if self.transform:
            sample['clips'] = sample['clips'][0:self.mi]

        return sample
        
    def __getitem__(self, idx):
        sample = self.getitem_without_labels(idx)

        sample['valenceValue'] = self.scores[self.scores['name']==sample['video']]['valenceValue'].iloc[0]
        sample['arousalValue'] = self.scores[self.scores['name']==sample['video']]['arousalValue'].iloc[0]

        return sample

if __name__ == '__main__':
    liris = LirisDataset(json_file='output-liris-resnet-34-kinetics.json', root_dir='/home/data_common/data_yangsen/data', transform=True, ranking_file='ACCEDEranking.txt', sep='\t')

    for i in range(5):
        print(liris[i])
