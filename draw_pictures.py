import movies
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

directory = 'ground_truth'

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

for name, movie in movies.movie_map.items():
    group_id = 0
    for group in movie.get_continuous_gourp(10):
        print (group)
        X = []
        Y = []
        for clip in group:
            valence, arousal = getLabelFromClip(clip.name)
            X.append(arousal)
            Y.append(valence)
        fig, ax = plt.subplots()
        fig.suptitle(f'arousal-valence-{name}-{group_id}')
        line, = ax.plot(X, Y, '--', linewidth=2, marker='x')
        ax.scatter(X[0], Y[0], s=300)
        ax.set_xlabel('arousal')
        ax.set_ylabel('valence')
        fig.savefig(os.path.join(directory, f'arousal-valence-{name}-{group_id}.png'))
        plt.close(fig)
        group_id += 1