#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, pdb, argparse, subprocess, pickle, os, gzip
import numpy as np
from SyncNetInstance import *
from ours_dataset import AudioVideoDataset_mp4 as dataset
import pdb
# ==================== PARSE ARGUMENT ====================

parser = argparse.ArgumentParser(description = "SyncNet");
parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='');
#parser.add_argument('--initial_model', type=str, default="../privatedata_slow/yifandata/output/data2_selected_full_face/ckpt_134.pth", help='');
parser.add_argument('--batch_size', type=int, default='20', help='');
parser.add_argument('--vshift', type=int, default='15', help='');
parser.add_argument('--data_dir', type=str, default='./data/test_videos/video', help='Output direcotry');
parser.add_argument('--output_dir', type=str, default='./results/test_videos', help='Output direcotry');
parser.add_argument('--audiofile', type=str, default='', help='');
parser.add_argument('--reference', type=str, default='', help='');
#parser.add_argument('--data_dir', type=str, default='../privatedata_slow/lucayongxu/ca11/wujian/wujian/LRS2/mvlrs_v1/main', help='Output direcotry');
#parser.add_argument('--output_dir', type=str, default='../privatedata_slow/yifandata/output/bbc', help='Output direcotry');
opt = parser.parse_args();
pdb.set_trace = lambda: None
setattr(opt, 'avi_dir', os.path.join(opt.output_dir, 'pyavi'))
setattr(opt, 'tmp_dir', os.path.join(opt.output_dir, 'pytmp'))
setattr(opt, 'work_dir', os.path.join(opt.output_dir, 'pywork'))
setattr(opt, 'crop_dir', os.path.join(opt.output_dir, 'pycrop'))

np.random.seed(0)
# ==================== LOAD MODEL ====================

s = SyncNetInstance();

#s.load(opt.initial_model);
s.loadParameters(opt.initial_model);
print("Model %s loaded."%opt.initial_model);

# ==================== GET OFFSETS ====================

test_dataset = dataset(opt)


#pdb.set_trace()
for iteration, video_dir in enumerate(test_dataset):
    #if iteration<200:
    opt.reference = video_dir[video_dir.rfind('/')+1:-4]
    dists = []
    offsets = []
    confs = []
    minvals = []
    names = []
    #opt.tmp_dir =
    #opt.reference = str(iteration)

    with open(os.path.join(opt.work_dir,opt.reference,'tracks.pckl'), 'r') as fil:
        tracks = pickle.load(fil)


    for ii, track in enumerate(tracks):
        offset, conf, dist, minval = s.evaluate(opt,videofile=os.path.join(opt.crop_dir,opt.reference,'%05d.avi'%ii))
        offsets.append(numpy.absolute(offset))
        dists.append(dist)
        confs.append(conf)
        minvals.append(minval)
        names.append(opt.reference)
    with open(os.path.join(opt.work_dir,opt.reference,'offsets.txt'), 'w') as fil:
        fil.write('FILENAME\tOFFSET\tCONF\n')
        for ii, track in enumerate(tracks):
          fil.write('%05d.avi\t%d\t%.3f\n'%(ii, offsets[ii], confs[ii]))

    with open(os.path.join(opt.work_dir,opt.reference,'activesd.pckl'), 'wb') as fil:
        pickle.dump(dists, fil)
#pdb.set_trace()
with open('results_ori_test_diar.txt','w') as fil:
    fil.write('average\n')
    fil.write('offsets_avg,%.3f\t minval_avg,%.3f\t conf_avg,%.3f\n'
              %(numpy.array(offsets).mean(),numpy.array(minvals).mean(),numpy.array(confs).mean()))
    for ii, name in enumerate(names):
        fil.write('%s\t offset,%.3f\t minval,%.3f\t conf,%.3f\n' % (name, offsets[ii],minvals[ii],confs[ii]))
    # ==================== PRINT RESULTS TO FILE ====================

    #with open(os.path.join(opt.work_dir,opt.reference,'offsets.txt'), 'w') as fil:
    #    fil.write('FILENAME\tOFFSET\tCONF\n')
    #    for ii, track in enumerate(tracks):
    #      fil.write('%05d.avi\t%d\t%.3f\n'%(ii, offsets[ii], confs[ii]))

    #with open(os.path.join(opt.work_dir,opt.reference,'activesd.pckl'), 'wb') as fil:
    #    pickle.dump(dists, fil)

