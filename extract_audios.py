import os
import glob
import subprocess

videos = glob.glob("../movieclips/*.mp4")

for video in videos:
    filename, _ = os.path.splitext(video)
    audio = filename + ".wav"
    command = "ffmpeg -i %s %s" % (video, audio)
    subprocess.call(command, shell=True)
