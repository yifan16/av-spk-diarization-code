import os
import glob
import numpy as np
import sys
import re
import cv2
import subprocess
import multiprocessing
import random
import argparse

#for tv in tvs:
def helper(mp4):
    print(mp4) 

    mp4_25fps_folder=os.path.join(input_dir,"video_25fps")
    if not os.path.exists(mp4_25fps_folder):
        os.makedirs(mp4_25fps_folder)
    #mp4s=glob.glob(os.path.join(input_dir,"video","*.mp4"))

    #for mp4 in mp4s:
    mp4_25fps_fn=mp4.split('/')[-1]
    mp4_25fps_path=os.path.join(mp4_25fps_folder,mp4_25fps_fn)
    if os.path.exists(mp4_25fps_path):
        print("aleady exists, skip",mp4_25fps_path)
        #continue

    video=cv2.VideoCapture(mp4)
    fps=video.get(cv2.CAP_PROP_FPS)
    video.release()
    print("fps ",fps)
    #sys.exit()
    print(mp4,mp4_25fps_path)
    if abs(fps-25.0) > 1e-6:
        if not os.path.exists(mp4_25fps_path):
        #sys.exit()
            subprocess.call(["ffmpeg", "-i", mp4, "-r","25", "-strict", "-2", mp4_25fps_path])
    else:
        if not os.path.exists(mp4_25fps_path):
            subprocess.call(["cp", mp4,mp4_25fps_path])

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='', help='input dir')
opt = parser.parse_args()

input_dir=opt.input_dir
tvs=glob.glob(os.path.join(input_dir,'video',"*"))  
random.shuffle(tvs)
thread_pool = multiprocessing.Pool(32)
thread_pool.map(helper, tvs)  
