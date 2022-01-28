import numpy as np
import glob 
import sys
import subprocess
import os
import multiprocessing
import argparse


def helper(mp4):
    #out_dir="/home/auszhang/publicdata_slow/Data/Lip/Seperation_withLip_Train"
    fn=mp4.split('/')[-1]
    des_audio=os.path.join(audio_folder,fn[:-4]+".wav")
    print(mp4,des_audio)
    if not (os.path.exists(des_audio)):
        process1 = subprocess.call(['ffmpeg', '-i', mp4, '-acodec', 'pcm_s16le' ,'-ac', '1', '-ar', '16000', '-vn', des_audio]) # extract audio file from video file"""    

#indir="/scratch/lucayongxu/publicdata_slow2/Data/Lip/Seperation_withLip_Train"
#indir=sys.argv[1]
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='', help='input dir')
opt = parser.parse_args()

indir=opt.input_dir
mp4s=glob.glob(os.path.join(indir,"video_25fps","*.mp4"))
audio_folder=os.path.join(indir,"audio")
if not os.path.exists(audio_folder):
    os.makedirs(audio_folder)

    
thread_pool = multiprocessing.Pool(32)
#results = thread_pool.map(try_preprocess_sample, sample_paths)
thread_pool.map(helper, mp4s)
#results= [p for p in results if p is not None]
