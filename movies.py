from xml.dom.minidom import parse
import xml.dom.minidom
import os
from functools import reduce
import platform


# independent with platform
# usually, I work on two platform. The PC or V100 server. The former is more convinient while the latter is more powerful.
path_prefix = None

if platform.system() == 'Linux':
    path_prefix = '/data'
else:
    path_prefix = 'A:/work/su/'

movie_xml = os.path.join(
    path_prefix, 'LIRIS-ACCEDE/LIRIS-ACCEDE-data/ACCEDEmovies.xml')
description_xml = os.path.join(
    path_prefix, 'LIRIS-ACCEDE/LIRIS-ACCEDE-data/ACCEDEdescription.xml')
data_path = os.path.join(path_prefix, 'LIRIS-ACCEDE/LIRIS-ACCEDE-data/data')
ranking_file = os.path.join(
    path_prefix, 'LIRIS-ACCEDE/LIRIS-ACCEDE-annotations/annotations/ACCEDEranking.txt')
sets_file = os.path.join(
    path_prefix, 'LIRIS-ACCEDE/LIRIS-ACCEDE-annotations/annotations/ACCEDEsets.txt')


class Movie(object):
    threshold = 1  # 多长的间隔认为连续

    def __init__(self, name, excerpts):
        self.name = name
        self.excerpts = excerpts
        self.clips = []
        self.current_group = []
        self.continuous_group = [self.current_group]

    def add_clip(self, clip):
        if len(self.clips) > 0:
            back = self.clips[-1]
            if clip.start - back.end <= self.threshold:
                self.current_group.append(clip)
            else:
                self.current_group = []
                self.continuous_group.append(self.current_group)
        self.clips.append(clip)

    def __hash__(self):
        return self.name.__hash__()

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return str([len(x) for x in self.continuous_group])

    def get_continuous_gourp(self, thres):
        return list(filter(lambda x: len(x) >= thres, self.continuous_group))


print(movie_xml)
DOMTree = xml.dom.minidom.parse(movie_xml)
content = DOMTree.documentElement
movies = content.getElementsByTagName("media")
movie_map = {}

for movie in movies:
    name = movie.getElementsByTagName('movie')[0].childNodes[0].data
    excerpts = movie.getElementsByTagName('excerpts')[0].childNodes[0].data
    movie_map[name] = Movie(name, excerpts)

description_dom_tree = xml.dom.minidom.parse(description_xml)
description_content = description_dom_tree.documentElement
medias = description_content.getElementsByTagName('media')


class Clip(object):
    def __init__(self, name, movie, start, end):
        self.name = name
        self.start = int(start)
        self.end = int(end)
        self.movie = movie

    def __str__(self):
        return f'{self.name} + {self.movie} + {self.start} + {self.end}'

    def __repr__(self):
        return self.__str__()


for media in medias:
    name = media.getElementsByTagName('name')[0].childNodes[0].data
    movie = media.getElementsByTagName('movie')[0].childNodes[0].data
    start = media.getElementsByTagName('start')[0].childNodes[0].data
    end = media.getElementsByTagName('end')[0].childNodes[0].data

    movie_map[movie].add_clip(Clip(name, movie, start, end))

print('--------------------------')
print(movie_map)
print('----')
print(len(movie_map))


def get_filter(threshold):
    def movie_continuious_length_filter(pair):
        _, movie = pair
        if max([len(x) for x in movie.continuous_group]) >= threshold:
            return True
        return False
    return movie_continuious_length_filter


for i in range(40):
    long_clips = dict(filter(get_filter(i), movie_map.items()))
    print(
        f'| {i} | {len(long_clips)} | {reduce(lambda x, y: x + y, list(map(lambda x: len(x[1].get_continuous_gourp(i)), long_clips.items())), 0)} |')
