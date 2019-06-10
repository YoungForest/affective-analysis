from xml.dom.minidom import parse
import xml.dom.minidom
import os

class Movie(object):
    threshold = 100 # 多长的间隔认为连续
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

movie_xml = 'A:/work/su/LIRIS-ACCEDE/LIRIS-ACCEDE-data/ACCEDEmovies.xml'
print(movie_xml)
DOMTree = xml.dom.minidom.parse(movie_xml)
content = DOMTree.documentElement
movies = content.getElementsByTagName("media")
movie_map = {}

for movie in movies:
    name = movie.getElementsByTagName('movie')[0].childNodes[0].data
    excerpts = movie.getElementsByTagName('excerpts')[0].childNodes[0].data
    movie_map[name] = Movie(name, excerpts)

description_xml = 'A:/work/su/LIRIS-ACCEDE/LIRIS-ACCEDE-data/ACCEDEdescription.xml'
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

print ('--------------------------')
print (movie_map)