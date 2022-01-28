#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, pdb, argparse, subprocess, pickle, os, gzip,glob

from SyncNetModel_lip import S,save,load
from ours_dataset import AudioVideoDataset_ours_lip as dataset
import pdb
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from utils.visualizer import Visualizer as vis
#from utils.tools import *
from torchnet import meter
from utils.losses import ContrastiveLoss,LiftedLoss,TripletLoss
from utils.tools import load_npy_lip, eval_ours_item,eval_ours_all
from utils.read_ours_gt import get_ours_label1
import numpy as np
import numpy
from scipy.signal import medfilt
import logging
import copy


# ==================== PARSE ARGUMENT ====================

parser = argparse.ArgumentParser(description = "SyncNet");
#parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='');
#parser.add_argument('--initial_model', type=str, default="../privatedata_slow/yifandata/output/data2_selected_full_face/ckpt_99_1.pth", help='');
parser.add_argument('--train_file', type=str, default="data/trainset_all_wavs_with_spk_info_v2.txt", help='');
#parser.add_argument('--save_dir', type=str, default="data/syncnet_v2.model", help='');
parser.add_argument('--batch_size', type=int, default=128, help='');
parser.add_argument('--start_epoch', type=int, default=260, help='');
parser.add_argument('--chunk_size', type=int, default=16, help='');
parser.add_argument('--frame_len', type=int, default=5, help='');
parser.add_argument('--norm_img', type=int, default=0, help='');
parser.add_argument('--margin', type=float, default=1., help='');
parser.add_argument('--num_epochs', type=int, default=10000, help='');
parser.add_argument('--lr', type=float, default=0.0001, help='');
parser.add_argument('--vshift', type=int, default='15', help='');
parser.add_argument('--settings', type=int, default='0', help='');
#parser.add_argument('--data_dir', type=str, default='../privatedata_slow/lucayongxu/ca15/data1_lucayongxu_chin_video/chin_video/selected_full_face_v2', help='Output direcotry');
parser.add_argument('--data_dir', type=str, default='../publicfast', help='Output direcotry');
parser.add_argument('--eval_data_dir', type=str, default='../publicfast/test_diar/diarization1', help='Output direcotry');
#parser.add_argument('--data_dir', type=str, default='../privatedata_slow/yifandata/test_data/', help='Output direcotry');
parser.add_argument('--output_dir', type=str, default='../privatedata_slow/yifandata/output/lip_1_contra', help='Output direcotry');
parser.add_argument('--output_dir_resume', type=str, default='../privatedata_slow/yifandata/output/lip_1_contra', help='Output direcotry');
#parser.add_argument('--videofile', type=str, default='', help='');
parser.add_argument('--reference', type=str, default='', help='');
parser.add_argument('--test', type=int, default=1, help='Output direcotry');
parser.add_argument('--audio', type=str, default='13', help='Output direcotry');
parser.add_argument('--video', type=str, default='lip', help='Output direcotry');

#parser.add_argument('--data_dir', type=str, default='../privatedata_slow/yifandata/ava/data', help='Output direcotry');
#parser.add_argument('--output_dir', type=str, default='../privatedata_slow/yifandata/output/ava', help='Output direcotry');
opt = parser.parse_args();

setattr(opt, 'avi_dir', os.path.join(opt.output_dir, 'pyavi'))
setattr(opt, 'tmp_dir', os.path.join(opt.output_dir, 'pytmp'))
setattr(opt, 'work_dir', os.path.join(opt.output_dir, 'pywork'))
setattr(opt, 'crop_dir', os.path.join(opt.output_dir, 'pycrop'))

DEFAULT_PORT = 8097
DEFAULT_HOSTNAME = "http://localhost"
#DEFAULT_HOSTNAME = "10.12.0.1"
viz = vis(env=opt.output_dir.split('/')[-1])
loss_meter = meter.AverageValueMeter()
loss_meter_eval = meter.AverageValueMeter()

for handler in logging.root.handlers[:]: 
    logging.root.removeHandler(handler)
logging.basicConfig(filename='./logs/'+opt.output_dir.split('/')[-1]+'.log',filemode = 'a',level=logging.INFO)

def train_epoch(epoch,optimizer,criterion,dataloader):
    s.train()
    losses = []
    start = 0
    end = 0

    for iteration, data in enumerate(dataloader):
        im_in,cc_in,label = data
        #print('im',im_in.shape)
        #print('cc',cc_in.shape)
        #pdb.set_trace()
        #viz.vis.images(im_in[0,:,0,:,:],
        #    opts=dict(title='input', caption='input'), win=1 )
        #end = time.time()
        #print('time',end-start)
        #print('label',label)
        #zpdb.set_trace()
        optimizer.zero_grad()
        im_out = s.forward_lip(im_in.cuda())
        cc_out = s.forward_aud(cc_in.cuda())
        #dist = torch.nn.functional.pairwise_distance(im_out,cc_out)
        #print(dist)
        loss = criterion(im_out,cc_out,label.cuda())
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

def eval_epoch(epoch,dataset):
    s.eval()
    losses = []
    #pdb.set_trace()
    start = 0
    end = 0
    count = 0.0
    for iteration, data in enumerate(dataset):
        imtv,cct,cct_neg = data
        offset, conf, dist, minval ,offset_neg = s.evaluate_allcase(opt,imtv,cct,cct_neg)
        #pdb.set_trace()
        loss_meter_eval.add(abs(offset))
        if abs(offset) <=5:
            count+=1
        offset_neg = 15 - abs(offset_neg)
        loss_meter_eval.add(offset_neg)
        if abs(offset_neg) <=10:
            count+=1


    viz.plot_many_stack({'eval_loss': loss_meter_eval.value()[0]})
    print('epoch',epoch)
    print('eval offset:' ,loss_meter_eval.value()[0])
    print('eval accuracy:' ,count/(len(dataset)*2))

    with open('./results/'+opt.output_dir.split('/')[-1]+'.txt','a') as f:
        f.write(str(epoch)+':'+str(count/(len(dataset)*2))+'\n')

def eval_ours(epoch,max_acc,max_acc1,max_acc2,max_acc_peak,max_acc_peak1,max_acc_peak2):
    s.eval()
    losses = []
    labels,labels_vad= get_ours_label1('./data/test_real_25/gt.txt','./data/test_real_25/vad.txt')
    audio_dir = './data/test_real_25/audio.npy'
    video_dirs = sorted(glob.glob('./data/test_real_25/lip*'))
    dists = []
    fdists = []
    

    cc = np.load(audio_dir)
    for iter,video_dir in enumerate(video_dirs):
    #if len(video_dir) > 0:
        #video_dir = opt.data_dir+ video_dir#.replace()
        #time1 = times[iter]
        #time2 = times[iter + 1]
        #pdb.set_trace()
        #opt.reference =  video_dir[video_dir.rfind('/')+1:-4] + '_' + str(iter)
        im = load_npy_lip(video_dir)      
        offset, conf, dist,fdist, minval = s.evaluate_diar_npy(opt,im,cc)
        #pdb.set_trace()
        #dist = dist[:,0]
        dist = medfilt(dist, kernel_size=19)
        fdist = medfilt(fdist, kernel_size=19)


            #offsets.append(numpy.absolute(offset))
        dists.append(dist)
        fdists.append(fdist)
            #confs.append(conf)
                #minvals.append(minval)
                #names.append(opt.reference)
    #pdb.set_trace()
    dists = numpy.asarray(dists)
    fdists = numpy.asarray(fdists)
    #pdb.set_trace()
    print('len dist',len(dists))
    print('num of video',len(video_dirs))

    #print('median',dists.median())
    print('mean',dists.mean())
    print('max',dists.max())
    print('min',dists.min())

    min_dists = np.min(dists,0)
    argmin_dists = np.argmin(dists,0) + 1


    #argmin_dists[min_dists>dists.mean()] = 0

    num_silence = len(labels) - np.count_nonzero(labels)


    #argmin_dists[min_dists>dists.mean()] = 0

    argmin_dists1 = copy.deepcopy(argmin_dists)
    argmin_dists2 = copy.deepcopy(argmin_dists)
    argmin_dists1[labels==0] = 0
    argmin_dists2[argmin_dists>dists.mean()] = 0

    max_fdists = np.max(fdists,0)
    argmax_fdists = np.argmax(fdists,0) + 1
    argmax_fdists1 = copy.deepcopy(argmax_fdists)
    argmax_fdists2 = copy.deepcopy(argmax_fdists)

    argmax_fdists2[max_fdists<fdists.mean()] = 0
    argmax_fdists1[labels==0] = 0


    
    #pdb.set_trace()
    correct = np.sum(labels == argmin_dists)#[:len(label)])
    #argmin_dists[min_dists>dists.mean()] = 0
    correct1 = np.sum(labels == argmin_dists1)#[:len(label)])
    correct2 = np.sum(labels == argmin_dists2)#[:len(label)])
    #correct3 = np.sum(labels == argmin_dists2)#[:len(label)])
    fcorrect = np.sum(labels == argmax_fdists)#[:len(label)])
    #argmin_dists[min_dists>dists.mean()] = 0
    fcorrect1 = np.sum(labels == argmax_fdists1)#[:len(label)])
    fcorrect2 = np.sum(labels == argmax_fdists2)#[:len(label)])

    if len(dists) == len(video_dirs):
        numpy.save(os.path.join('./results/'+opt.output_dir.split('/')[-1]+'_dists.npy'),dists)
        numpy.save(os.path.join('./results/'+opt.output_dir.split('/')[-1]+'_fdists.npy'),fdists)
    else:
        print('mis match!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            #
        #loss_meter_eval.add(probs)
    #viz.plot_many_stack({'eval_loss': loss_meter_eval.value()[0]})
    print('epoch ours',epoch)
    #pdb.set_trace()
    acc = correct/float(len(labels))
    acc2 = correct2/float(len(labels))
    acc1 = (correct1-num_silence)/float((len(labels)-num_silence))
    #acc1 = correct1/float(len(labels))
    #acc3 = correct2/float(len(labels))
    facc = fcorrect/float(len(labels))
    facc2 = fcorrect2/float(len(labels))
    facc1 = (fcorrect1-num_silence)/float((len(labels)-num_silence))
    #pdb.set_trace()
    #(np.count_nonzero(idxs) + len(idxs_neg) - np.count_nonzero(idxs_neg))/float(len(idxs)+len(idxs_neg))
    #print('eval offset:' ,loss_meter_eval.value()[0])
    print('eval ours accuracy:' ,acc)
    print('eval ours accuracy1:' ,acc1)
    print('eval ours accuracy2:' ,acc2)
    print('eval ours accuracy peak:' ,facc)
    print('eval ours accuracy peak1:' ,facc1)
    print('eval ours accuracy peak1:' ,facc2)
    #pdb.set_trace()
    #(np.count_nonzero(idxs) + len(idxs_neg) - np.count_nonzero(idxs_neg))/float(len(idxs)+len(idxs_neg))
    #print('eval offset:' ,loss_meter_eval.value()[0])

    if max_acc < acc:
        max_acc = acc
    if max_acc1 < acc1:
        max_acc1 = acc1
    if max_acc2 < acc2:
        max_acc2 = acc2
    if max_acc_peak < facc:
        max_acc_peak = facc
    if max_acc_peak1 < facc1:
        max_acc_peak1 = facc1
    if max_acc_peak2 < facc2:
        max_acc_peak2 = facc2
    print('max acc:' ,max_acc)
    print('max acc1:' ,max_acc1)
    print('max acc2:' ,max_acc2)
    print('max acc peak:' ,max_acc_peak)
    print('max acc peak1:' ,max_acc_peak1)
    print('max acc peak2:' ,max_acc_peak2)
    #if acc > 0.6:
    #    pdb.set_trace()
    #pdb.set_trace()
    with open('./results/'+opt.output_dir.split('/')[-1]+'.txt','a') as f:
        f.write(str(epoch)+'\t'+str(acc)+'\t'+str(acc1)+'\t'+str(acc2)+'\t'+str(facc)+'\t'+str(facc1)+'\t'+str(facc2)+'\n')
    return max_acc,max_acc1,max_acc2,max_acc_peak,max_acc_peak1,max_acc_peak2

# ==================== LOAD MODEL ====================

s = S();
#s.loadParameters(opt.initial_model);
s.cuda()
optimizer = torch.optim.Adam(s.parameters(), lr=opt.lr, betas=(0.9, 0.999))
criterion = ContrastiveLoss(margin=opt.margin)
  
dists = []
offsets = []
confs = []
minvals = []
names = []
max_accs = ''

if opt.test:
    for epoch in range(1,opt.start_epoch+1):
        checkpoint = torch.load(opt.output_dir + '/ckpt_'+str(epoch)+'.pth')
        s.load_state_dict (checkpoint['model_state_dict']) 
        
        max_accs = eval_ours_all(epoch,s,opt,'',max_accs)
        print(opt.output_dir)

else:
    if opt.start_epoch > 0:
        checkpoint = torch.load(opt.output_dir_resume + '/ckpt_'+str(opt.start_epoch)+'.pth')
        s.load_state_dict (checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        opt.start_epoch = checkpoint['epoch'] + 1
        print("Model %s loaded."%opt.start_epoch);
    else:
        print("Model initialized.",opt.start_epoch);
        opt.start_epoch += 1

    train_dataset = dataset(opt,mode='train',type = 'shift')
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=20)

    for e in range(opt.start_epoch,opt.num_epochs):
        #max_accs = eval_ours_all(e,s,opt,'',max_accs)
        train_epoch(e,optimizer,criterion,train_dataloader)
        print(opt.output_dir)


