#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, pdb, argparse, subprocess, pickle, os, gzip

from SyncNetInstance import *

# ==================== PARSE ARGUMENT ====================

parser = argparse.ArgumentParser(description = "SyncNet");
parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='');
parser.add_argument('--batch_size', type=int, default='20', help='');
parser.add_argument('--vshift', type=int, default='15', help='');
#parser.add_argument('--data_dir', type=str, default='./ours/chinese_edit_lowrev', help='Output direcotry');
#parser.add_argument('--videofile', type=str, default='./data/chinese_edit.mp4', help='Input video file');
#parser.add_argument('--audiofile',type=str, default='./data/chinese_edit_lowrev.wav', help='Minimum facetrack duration');
parser.add_argument('--data_dir', type=str, default='./ours/recorded_cut_25', help='Output direcotry');
parser.add_argument('--videofile', type=str, default='./data/recorded_cut_25.mp4', help='Input video file');
parser.add_argument('--audiofile',type=str, default='', help='Minimum facetrack duration');

parser.add_argument('--reference', type=str, default='', help='');
opt = parser.parse_args();

setattr(opt,'avi_dir',os.path.join(opt.data_dir,'pyavi'))
setattr(opt,'tmp_dir',os.path.join(opt.data_dir,'pytmp'))
setattr(opt,'work_dir',os.path.join(opt.data_dir,'pywork'))
setattr(opt,'crop_dir',os.path.join(opt.data_dir,'pycrop'))

pdb.set_trace = lambda: None


# ==================== LOAD MODEL ====================

s = SyncNetInstance();

s.loadParameters(opt.initial_model);
print("Model %s loaded."%opt.initial_model);

# ==================== GET OFFSETS ====================

with open(os.path.join(opt.work_dir,opt.reference,'tracks.pckl'), 'r') as fil:
    tracks = pickle.load(fil)

dists = []
dists1 = []
offsets = []
confs = []
for ii, track in enumerate(tracks):
    offset, conf, dist,_ = s.evaluate(opt,videofile=os.path.join(opt.crop_dir,opt.reference,'%05d.avi'%ii))
    offsets.append(offset)
    dists.append(dist)
    confs.append(conf)
    print('dddddddddddddd',dist.shape)
    dists1.append(dist[:,0])
      
# ==================== PRINT RESULTS TO FILE ====================

with open(os.path.join(opt.work_dir,opt.reference,'offsets.txt'), 'w') as fil:
    fil.write('FILENAME\tOFFSET\tCONF\n')
    for ii, track in enumerate(tracks):
      fil.write('%05d.avi\t%d\t%.3f\n'%(ii, offsets[ii], confs[ii]))
      
with open(os.path.join(opt.work_dir,opt.reference,'activesd.pckl'), 'wb') as fil:
    pickle.dump(dists, fil)

dists1 = numpy.asarray(dists1)
#if len(dists) == len(video_dirs):
numpy.save(os.path.join('test_dists.npy'),dists1)
