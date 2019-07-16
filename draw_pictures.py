import movies
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.interpolate import interp1d

directory = 'smooth'

if not os.path.exists(directory):
    os.mkdir(directory)

scores = pd.read_csv(movies.ranking_file, sep='\t')
min_max_scaler = MinMaxScaler(feature_range=(-1, 1), copy=False)
valenceRank = scores[['valenceRank']].values.astype(float)
min_max_scaler.fit_transform(valenceRank)
scores['valenceRankRescaled'] = pd.DataFrame(valenceRank)
arousalRank = scores[['arousalRank']].values.astype(float)
min_max_scaler.fit_transform(arousalRank)
scores['arousalRankRescaled'] = pd.DataFrame(arousalRank)


def getLabelFromClip(name):
    return scores[scores['name'] == name]['valenceRankRescaled'].iloc[0], scores[scores['name'] == name]['arousalRankRescaled'].iloc[0]


predict = pd.read_csv('predict.csv')

def smooth(X, Y):
    x_array = np.array(X)
    y_array = np.array(Y)
    x_new = np.linspace(x_array.max(), x_array.min(), 500)
    f = interp1d(x_array, y_array, kind = 'quadratic')
    y_smooth = f(x_new)

    return x_new, y_smooth

for name, movie in movies.movie_map.items():
    group_id = 0
    for group in movie.get_continuous_gourp(10):
        print(group)
        X = []
        Y = []
        predict_X = []
        predict_Y = []
        for clip in group:
            valence, arousal = getLabelFromClip(clip.name)
            X.append(arousal)
            Y.append(valence)
            Y.append(predict[predict['name'] ==
                                     clip.name]['ground_truth_valence'].iloc[0])
            X.append(predict[predict['name'] ==
                                     clip.name]['ground_truth_arousal'].iloc[0])
            predict_Y.append(predict[predict['name'] ==
                                     clip.name]['valence'].iloc[0])
            predict_X.append(predict[predict['name'] ==
                                     clip.name]['arousal'].iloc[0])
        fig, ax = plt.subplots()
        fig.suptitle(f'arousal-valence-{name}-{group_id}')
        assert(len(predict_X) != 0)
        x_smooth, y_smooth = smooth(X, Y)
        line, = ax.plot(x_smooth, y_smooth, '--', linewidth=2)
        ax.scatter(X, Y, marker='X')
        predict_x_smooth, predict_y_smooth = smooth(predict_X, predict_Y)
        line, = ax.plot(predict_x_smooth, predict_y_smooth, 'r--',
                        linewidth=2)
        ax.scatter(predict_x_smooth, predict_y_smooth, marker='X') 
        ax.scatter(X[0], Y[0], s=300)
        ax.scatter(predict_X[0], predict_Y[0], s=300)
        ax.set_xlabel('arousal')
        ax.set_ylabel('valence')
        fig.savefig(os.path.join(
            directory, f'arousal-valence-{name}-{group_id}.png'))
        plt.close(fig)
        group_id += 1
