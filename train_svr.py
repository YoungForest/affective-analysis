import json
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

data = None
with open('output-resnet-34-kinetics.json', 'r') as myfile:
    data = myfile.read()
data_json = json.loads(data)

mi = 1000
for d in data_json:                                       
    if mi > len(d['clips']):
        mi = len(d['clips'])

labels = ['arousal', 'excitement', 'pleasure', 'contentment',
          'sleepiness', 'depression', 'misery', 'distress']
df = pd.read_csv('./filled-labels_features.csv')
cv_id = pd.read_csv('./cv_id_10.txt')
df['cv_id'] = cv_id['cv_id_10']

test_id = 8
train_data = df[df['cv_id'] != test_id]  # train data
valid_data = df[df['cv_id'] == test_id]  # test data

scores = []

for class_name in labels:
    train_X = []
    train_target = []
    test_X = []
    test_target = []
    for d in data_json:
        sub_clips = d['clips'][0:mi]
        content = []
        for c in sub_clips:
            content += c['features']
        video_name = os.path.splitext(d['video'])[0]
        y = df[df['id'] == video_name][class_name].values[0]

        if df[df['id'] == video_name].iloc[0]['cv_id'] != test_id:
            # train data
            train_X.append(content)
            train_target.append(y)
        else:
            # test data
            test_X.append(content)
            test_target.append(y)

    clf = SVR()
    clf.fit(train_X, train_target)
    result = clf.predict(test_X)
    result = result.clip(0, 2)
    clsscore = mean_squared_error(y_true=test_target, y_pred=result)
    print("%s scored : %f" %(class_name, clsscore))
    scores.append(clsscore)

print(sum(scores)/len(scores))
