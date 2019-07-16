# 工作流程

[代码地址](https://github.com/YoungForest/affective-analysis)
V100 地址 ssh zy1806711@10.134.166.8
数据集地址L: A:\work\su\LIRIS-ACCEDE

## 数据集
原视频地址 A:\work\su\LIRIS-ACCEDE\LIRIS-ACCEDE-data\data
以.mp4结尾的所有文件。

## 连续片段分析

连续阈值设置为100, 单位为帧。所有的视频1秒24帧。

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

作为对比，严格的连续片段。

| 过滤连续clip的数量阈值(>=) | 符合要求的电影数目 | 连续片段组成的长片段的数量(把这个片段当成新的电影) |
| -- | -- | -- |
| 0 | 160 | 4300 |
| 1 | 158 | 2279 |
| 2 | 143 | 1246 |
| 3 | 123 | 739 |
| 4 | 101 | 471 |
| 5 | 82 | 293 |
| 6 | 64 | 187 |
| 7 | 49 | 122 |
| 8 | 35 | 71 |
| 9 | 24 | 49 |
| 10 | 19 | 36 |
| 11 | 4 | 4 |
| 12 | 1 | 1 |
| 13 | 1 | 1 |
| 14 | 1 | 1 |
| 15 | 0 | 0 |
| 16 | 0 | 0 |
| 17 | 0 | 0 |
| 18 | 0 | 0 |
| 19 | 0 | 0 |
| 20 | 0 | 0 |
| 21 | 0 | 0 |
| 22 | 0 | 0 |
| 23 | 0 | 0 |
| 24 | 0 | 0 |
| 25 | 0 | 0 |
| 26 | 0 | 0 |
| 27 | 0 | 0 |
| 28 | 0 | 0 |
| 29 | 0 | 0 |
| 30 | 0 | 0 |
| 31 | 0 | 0 |
| 32 | 0 | 0 |
| 33 | 0 | 0 |
| 34 | 0 | 0 |
| 35 | 0 | 0 |
| 36 | 0 | 0 |
| 37 | 0 | 0 |
| 38 | 0 | 0 |
| 39 | 0 | 0 |

## 特征提取
```
python main.py --input /data/affective-analysis/input --video_root /data/LIRIS-ACCEDE/LIRIS-ACCEDE-data/data --output ./output.json --model /data/PretrainedModels/resnet-34-kinetics.pth --mode feature
```

## 训练过程
```
CUDA_VISIBLE_DEVICES=0 python liris_net.py
```
使用2块GPU会有如下错误：
> RuntimeError: Input and hidden tensors are not at the same device, found input tensor at cuda:1 and hidden tensor at cuda:0

训练集 : 测试集 = 3 : 1

4个小时50轮。

## tensorboard

```
tensorboard --logdir log --port 6006
```

### tensorboard 标签解释

test_average_6_28: 最开始的测试结果。因为训练了50轮仍然没有收敛，所以我又load了最后一轮的参数，多训练50轮。但是发现load后会有一个小断层，往上跳了一下。
test_average_6_29: 一次训练150轮。因为retain_graph，所以训练到66轮的时候会显存爆掉。
test_average_7_2: 取消掉retain_graph，同时batch调小到16.
test_average_7_3: batch调大到128。
7_13: 
7_14: 加hidden_dim 2048
7_15: 加lstm层数 5


## 网络结构

特征维数：
14336