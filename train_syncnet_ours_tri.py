#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, pdb, argparse, subprocess, pickle, os, gzip

from SyncNetModel import S,save,load
from ours_dataset import AudioVideoDataset_ours_tri as dataset
import pdb
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from utils.visualizer import Visualizer as vis
from utils.tools import *
from torchnet import meter
from utils.losses import ContrastiveLoss,LiftedLoss,TripletLoss
import logging

# ==================== PARSE ARGUMENT ====================

parser = argparse.ArgumentParser();
#parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='');
#parser.add_argument('--initial_model', type=str, default="../privatedata_slow/yifandata/output/data2_selected_full_face/ckpt_99_1.pth", help='');
parser.add_argument('--train_file', type=str, default="data/trainset_all_wavs_with_spk_info_v2.txt", help='');
#parser.add_argument('--save_dir', type=str, default="data/syncnet_v2.model", help='');
parser.add_argument('--batch_size', type=int, default=20, help='');
parser.add_argument('--start_epoch', type=int, default=154, help='');
parser.add_argument('--chunk_size', type=int, default=16, help='');
parser.add_argument('--frame_len', type=int, default=5, help='');
parser.add_argument('--num_epochs', type=int, default=10000, help='');
parser.add_argument('--lr', type=float, default=0.00001, help='');
parser.add_argument('--vshift', type=int, default='15', help='');
parser.add_argument('--settings', type=int, default='0', help='');
#parser.add_argument('--data_dir', type=str, default='../privatedata_slow/lucayongxu/ca15/data1_lucayongxu_chin_video/chin_video/selected_full_face_v2', help='Output direcotry');
parser.add_argument('--data_dir', type=str, default='../publicfast/data2_selected_full_face', help='Output direcotry');
parser.add_argument('--eval_data_dir', type=str, default='../publicfast/test_diar/diarization1', help='Output direcotry');
#parser.add_argument('--data_dir', type=str, default='../privatedata_slow/yifandata/test_data/', help='Output direcotry');
parser.add_argument('--output_dir', type=str, default='../privatedata_slow/yifandata/output/data2_selected_full_face_tri', help='Output direcotry');
#parser.add_argument('--videofile', type=str, default='', help='');
parser.add_argument('--reference', type=str, default='', help='');
#parser.add_argument('--data_dir', type=str, default='../privatedata_slow/yifandata/ava/data', help='Output direcotry');
#parser.add_argument('--output_dir', type=str, default='../privatedata_slow/yifandata/output/ava', help='Output direcotry');
opt = parser.parse_args()

setattr(opt, 'avi_dir', os.path.join(opt.output_dir, 'pyavi'))
setattr(opt, 'tmp_dir', os.path.join(opt.output_dir, 'pytmp'))
setattr(opt, 'work_dir', os.path.join(opt.output_dir, 'pywork'))
setattr(opt, 'crop_dir', os.path.join(opt.output_dir, 'pycrop'))

DEFAULT_PORT = 8097
DEFAULT_HOSTNAME = "http://localhost"
#DEFAULT_HOSTNAME = "10.12.0.1"
viz = vis(env=opt.output_dir)
loss_meter = meter.AverageValueMeter()
loss_meter_eval = meter.AverageValueMeter()

for handler in logging.root.handlers[:]: 
    logging.root.removeHandler(handler)
logging.basicConfig(filename='./logs/'+opt.output_dir.split('/')[-1]+'.log',filemode = 'a',level=logging.INFO)




#/privatedata_slow/lucayongxu/ca15/data1_lucayongxu_chin_video/chin_video/selected_full_face_v2/sep_with_liptrain_trimmed_video_500ms_v2_p3/
# ==================== LOAD MODEL ====================

s = S();
#s.loadParameters(opt.initial_model);
s.cuda()
optimizer = torch.optim.Adam(s.parameters(), lr=opt.lr, betas=(0.9, 0.999))
criterion = TripletLoss(margin=1.)

if opt.start_epoch > 0:
    checkpoint = torch.load(opt.output_dir + '/ckpt_'+str(opt.start_epoch)+'.pth')
    s.load_state_dict (checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    opt.start_epoch = checkpoint['epoch'] + 1

#criterion = torch.nn.BCELoss()
#s.loadParameters(opt.initial_model);
#s.load(opt.initial_model);
#start_epoch = 0
print("Model %s loaded."%opt.start_epoch);

# ==================== GET OFFSETS ====================

train_dataset = dataset(opt,mode='train',type = 'shift')
train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=20)

eval_dataset = dataset(opt,mode='eval',type = 'shift')
#eval_dataloader = DataLoader(eval_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=20)

dists = []
offsets = []
confs = []
minvals = []
names = []
#pdb.set_trace()

def train_epoch(epoch,optimizer,criterion,dataloader):
    s.train()
    losses = []
    #pdb.set_trace()
    start = 0
    end = 0

    for iteration, data in enumerate(dataloader):
        im_in,cc_in,cc_in1,label,label1 = data
        #print('im',im_in.shape)
        #print('cc',cc_in.shape)
        #pdb.set_trace()
        #viz.vis.images(im_in[0,:,0,:,:],
        #    opts=dict(title='input', caption='input'), win=1 )
        #end = time.time()
        #print('time',end-start)
        #print('label',label)
        #pdb.set_trace()
        optimizer.zero_grad()
        im_out = s.forward_lip(im_in.cuda())
        cc_out = s.forward_aud(cc_in.cuda())
        cc_out1 = s.forward_aud(cc_in1.cuda())
        #dist = torch.nn.functional.pairwise_distance(im_out,cc_out)
        #print(dist)
        #if label<label1:
        #    loss = criterion(im_out,cc_out,cc_out1)
        #else:
        loss = criterion(im_out,cc_out1,cc_out)

            #criterion(im_out,cc_out,label.cuda())
        print(loss)
        loss_meter.add(loss.data.cpu())

        loss.backward()
        optimizer.step()
        #torch.cuda.synchronize()
        #losses.append(loss.data.cpu().numpy())
        #start = time.time()
        #print('gpu time',start - end)
        #if iteration % 10 == 1:
        viz.plot_many_stack({'train_loss': loss_meter.value()[0]})
        #if iteration % 1000 == 1:
        if not os.path.exists(opt.output_dir):
            os.makedirs(opt.output_dir)
        #plt.plot(losses)
        #save(s, opt.output_dir + '/ckpt_'+str(epoch)+'.pth')

    torch.save({
        'epoch': epoch,
        'model_state_dict': s.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }, opt.output_dir + '/ckpt_'+str(epoch)+'.pth')


    return losses

def eval_epoch(epoch,optimizer,criterion,dataset):
    s.eval()
    losses = []
    #pdb.set_trace()
    start = 0
    end = 0
    count = 0.0
    for iteration, data in enumerate(dataset):
        imtv,cct = data
        offset, conf, dist, minval  = s.evaluate(opt,imtv,cct)
        #pdb.set_trace()
        loss_meter_eval.add(abs(offset))
        if abs(offset) <=5:
            count+=1
    viz.plot_many_stack({'eval_loss': loss_meter_eval.value()[0]})
    print('epoch',epoch)
    print('eval offset:' ,loss_meter_eval.value()[0])
    print('eval accuracy:' ,count/len(dataset))
     
#eval_epoch(0,optimizer,criterion,eval_dataset)
#input()
for e in range(opt.start_epoch,opt.num_epochs):
    #train_dataset.load_next_chunk()
    #train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    train_epoch(e,optimizer,criterion,train_dataloader)
    eval_epoch(e,optimizer,criterion,eval_dataset)



