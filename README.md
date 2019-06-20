# 工作流程

[代码地址](https://github.com/YoungForest/affective-analysis)
V100 地址 ssh zy1806711@10.134.166.8
数据集地址L: A:\work\su\LIRIS-ACCEDE

## 数据集
原视频地址 A:\work\su\LIRIS-ACCEDE\LIRIS-ACCEDE-data\data
以.mp4结尾的所有文件。

## 连续片段分析

| 过滤连续clip的数量阈值(>=) | 符合要求的电影数目 | 连续片段组成的长片段的数量(把这个片段当成新的电影) |
| -- | -- | -- |
| 0 | 160 | 3629 |
| 1 | 159 | 2107 |
| 2 | 154 | 1277 |
| 3 | 150 | 855 |
| 4 | 130 | 594 |
| 5 | 117 | 410 |
| 6 | 97 | 288 |
| 7 | 75 | 209 |
| 8 | 60 | 143 |
| 9 | 47 | 111 |
| 10 | 40 | 91 |
| 11 | 16 | 18 |
| 12 | 12 | 14 |
| 13 | 10 | 12 |
| 14 | 8 | 10 |
| 15 | 4 | 6 |
| 16 | 3 | 5 |
| 17 | 1 | 3 |
| 18 | 1 | 3 |
| 19 | 1 | 3 |
| 20 | 1 | 3 |
| 21 | 1 | 3 |
| 22 | 1 | 2 |
| 23 | 1 | 1 |
| 24 | 1 | 1 |
| 25 | 1 | 1 |
| 26 | 1 | 1 |
| 27 | 0 | 0 |

## 特征提取
python main.py --input /data/affective-analysis/input --video_root /data/LIRIS-ACCEDE/LIRIS-ACCEDE-data/data --output ./output.json --model /data/PretrainedModels/resnet-34-kinetics.pth --mode feature