from scipy.io import wavfile
import python_speech_features
import os, sys, math, random, glob, cv2,pdb
import numpy as np
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='', help='input dir')
opt = parser.parse_args()

indir=opt.input_dir
dirs = [indir]
for dir in dirs:
    if not os.path.exists(dir+'/audio_feat'):
        os.makedirs(dir+'/audio_feat')
    piks = glob.glob(dir + '/audio/*.wav')

    #pdb.set_trace()
#piks = glob.glob(path + '/*.wav')
    for pik in piks:

        sample_rate, audio = wavfile.read(pik)
        print(pik)
        #pdb.set_trace()
        #audio = audio[:,0]
        # print('audio',audio.shape)
        mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
        mfcc = np.stack([np.array(i) for i in mfcc])
        # cc = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=0)
        cc = np.expand_dims(mfcc, axis=0)  # , axis=0)
        np.save(pik.replace('.wav', '.npy').replace('audio','audio_feat'), cc)