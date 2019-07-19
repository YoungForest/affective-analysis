import os
import glob
import subprocess

data_path = '/data/LIRIS-ACCEDE/LIRIS-ACCEDE-data/data'
audio_path = '/data/LIRIS-ACCEDE/LIRIS-ACCEDE-data/data/audio'
videos = glob.glob("/data/LIRIS-ACCEDE/LIRIS-ACCEDE-data/data/*.mp4")

for video in videos:
    v = os.path.split(video)[-1]
    filename, _ = os.path.splitext(v)
    audio = os.path.join(audio_path, filename) + ".wav"
    if os.path.exists(audio):
        continue
    command = "ffmpeg -i %s %s" % (video, audio)
    subprocess.call(command, shell=True)
