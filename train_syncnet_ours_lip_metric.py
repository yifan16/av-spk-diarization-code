#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, pdb, argparse, subprocess, pickle, os, gzip,glob

from SyncNetModel_lip import S,save,load
from ours_dataset import AudioVideoDataset_ours_chunk as dataset
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
from utils.tools import load_npy_lip,eval_ours_all,get_dists
from utils.read_ours_gt import get_ours_label1
import numpy as np
import numpy
from scipy.signal import medfilt
import logging

# ==================== PARSE ARGUMENT ====================

parser = argparse.ArgumentParser(description = "SyncNet");
#parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='');
parser.add_argument('--initial_model', type=str, default="../privatedata_slow/yifandata/output/lip_1_contra/ckpt_262.pth", help='');
parser.add_argument('--train_file', type=str, default="data/trainset_all_wavs_with_spk_info_v2.txt", help='');
#parser.add_argument('--save_dir', type=str, default="data/syncnet_v2.model", help='');
parser.add_argument('--batch_size', type=int, default=16, help='');
parser.add_argument('--start_epoch', type=int, default=4600, help='');
parser.add_argument('--frame_len', type=int, default=35, help='');
parser.add_argument('--norm_img', type=int, default=0, help='');
parser.add_argument('--chunk_size', type=int, default=256, help='');
parser.add_argument('--margin', type=float, default=5., help='');
parser.add_argument('--num_epochs', type=int, default=10000, help='');
parser.add_argument('--lr', type=float, default=0.0001, help='');
parser.add_argument('--vshift', type=int, default='15', help='');
parser.add_argument('--type', type=str, default='shift', help='');
#parser.add_argument('--data_dir', type=str, default='../privatedata_slow/lucayongxu/ca15/data1_lucayongxu_chin_video/chin_video/selected_full_face_v2', help='Output direcotry');
parser.add_argument('--data_dir', type=str, default='../publicfast', help='Output direcotry');
parser.add_argument('--eval_data_dir', type=str, default='../publicfast/test_diar/diarization1', help='Output direcotry');
#parser.add_argument('--data_dir', type=str, default='../privatedata_slow/yifandata/test_data/', help='Output direcotry');
parser.add_argument('--output_dir', type=str, default='../privatedata_slow/yifandata/output/lip_metric_s_m5_l3_v15', help='Output direcotry');
#parser.add_argument('--output_dir_resume', type=str, default='../privatedata_slow/yifandata/output/lip_10_contra_shift', help='Output direcotry');
#parser.add_argument('--videofile', type=str, default='', help='');
parser.add_argument('--reference', type=str, default='', help='');
parser.add_argument('--test', type=int, default=0, help='Output direcotry');
parser.add_argument('--audio', type=str, default='13', help='Output direcotry');
parser.add_argument('--video', type=str, default='lip', help='Output direcotry');
parser.add_argument('--interval', type=int, default=50, help='Output direcotry');
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



#/privatedata_slow/lucayongxu/ca15/data1_lucayongxu_chin_video/chin_video/selected_full_face_v2/sep_with_liptrain_trimmed_video_500ms_v2_p3/
# ==================== LOAD MODEL ====================

s = S();
#s.loadParameters(opt.initial_model);
s.cuda()
optimizer = torch.optim.Adam(s.parameters(), lr=opt.lr, betas=(0.9, 0.999))
criterion = TripletLoss(margin=opt.margin)

if not opt.test:

    if opt.initial_model:
        checkpoint = torch.load(opt.initial_model)
        s.load_state_dict (checkpoint['model_state_dict'])

    if opt.start_epoch > 0:
        checkpoint = torch.load(opt.output_dir + '/ckpt_'+str(opt.start_epoch)+'.pth')
        s.load_state_dict (checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        opt.start_epoch = checkpoint['epoch'] + 1
        logging.info("Model %s loaded."%opt.start_epoch);
    else:
        logging.info("Model %s initialized."%opt.start_epoch);
        opt.start_epoch += 1

    #criterion = torch.nn.BCELoss()
    #s.loadParameters(opt.initial_model);
    #s.load(opt.initial_model);
    #start_epoch = 0
    #logging.info("Model %s loaded."%opt.start_epoch);

    # ==================== GET OFFSETS ====================

    train_dataset = dataset(opt,mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=20)

#eval_dataset = dataset(opt,mode='eval_ours',type = 'shift')

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

    if epoch % opt.interval == 0:
        train_dataset.get_chunk()
        train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=20)

    for iteration, data in enumerate(dataloader):
        im_in,cc_in = data

        idx_smallest_arange,idx_2nd_smallest_arange = get_dists(im_in,cc_in,s,opt)
        #logging.info('im',im_in.shape)
        #logging.info('cc',cc_in.shape)
        #pdb.set_trace()
        #viz.vis.images(im_in[0,:,0,:,:],
        #    opts=dict(title='input', caption='input'), win=1 )
        #end = time.time()
        #logging.info('time',end-start)
        #logging.info('label',label)
        optimizer.zero_grad()

        #pdb.set_trace()
        #losses = []
        im_in_shift = im_in[:,:,opt.vshift:opt.frame_len- opt.vshift,:,:]

        #idx_smallest_arange = torch.stack([torch.arange(i*4,(i+5)*4) for i in idx_smallest])
        cc_in_shift1 =cc_in.gather(3,idx_smallest_arange.view(-1,1,1,20).expand(cc_in.size(0),cc_in.size(1),cc_in.size(2),20))

        #idx_2nd_smallest_arange = torch.stack([torch.arange(i*4,(i+5)*4) for i in idx_2nd_smallest])
        cc_in_shift2 =cc_in.gather(3,idx_2nd_smallest_arange.view(-1,1,1,20).expand(cc_in.size(0),cc_in.size(1),cc_in.size(2),20))
        
        im_out = s.forward_lip(im_in_shift.cuda())
        cc_out = s.forward_aud(cc_in_shift1.cuda())
        cc_out1 = s.forward_aud(cc_in_shift2.cuda())
        #dist = torch.nn.functional.pairwise_distance(im_out,cc_out)
        #logging.info(dist)
        loss = criterion(im_out,cc_out,cc_out1)
        #loss = min(losses)
        loss_cpu = loss.data.cpu()
        logging.info('{0}'.format(loss_cpu))
        print(loss_cpu)
        loss_meter.add(loss_cpu)

        loss.backward()
        optimizer.step()
        #torch.cuda.synchronize()
        #losses.append(loss.data.cpu().numpy())
        #start = time.time()
        #logging.info('gpu time',start - end)
        #if iteration % 10 == 1:
        viz.plot_many_stack({'train_loss': loss_meter.value()[0]})
        #if iteration % 1000 == 1:
        if not os.path.exists(opt.output_dir):
            os.makedirs(opt.output_dir)
        #plt.plot(losses)
        #save(s, opt.output_dir + '/ckpt_'+str(epoch)+'.pth')
    if epoch % opt.interval == 0:
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
    logging.info('epoch',epoch)
    logging.info('eval offset:' ,loss_meter_eval.value()[0])
    logging.info('eval accuracy:' ,count/(len(dataset)*2))

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
    logging.info('len dist',len(dists))
    logging.info('num of video',len(video_dirs))

    #logging.info('median',dists.median())
    logging.info('mean',dists.mean())
    logging.info('max',dists.max())
    logging.info('min',dists.min())

    min_dists = np.min(dists,0)
    argmin_dists = np.argmin(dists,0) + 1


    #argmin_dists[min_dists>dists.mean()] = 0

    num_silence = len(labels) - np.count_nonzero(labels)


    #argmin_dists[min_dists>dists.mean()] = 0

    import copy
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
        logging.info('mis match!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            #
        #loss_meter_eval.add(probs)
    #viz.plot_many_stack({'eval_loss': loss_meter_eval.value()[0]})
    logging.info('epoch ours',epoch)
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
    #logging.info('eval offset:' ,loss_meter_eval.value()[0])
    logging.info('eval ours accuracy:' ,acc)
    logging.info('eval ours accuracy1:' ,acc1)
    logging.info('eval ours accuracy2:' ,acc2)
    logging.info('eval ours accuracy peak:' ,facc)
    logging.info('eval ours accuracy peak1:' ,facc1)
    logging.info('eval ours accuracy peak1:' ,facc2)
    #pdb.set_trace()
    #(np.count_nonzero(idxs) + len(idxs_neg) - np.count_nonzero(idxs_neg))/float(len(idxs)+len(idxs_neg))
    #logging.info('eval offset:' ,loss_meter_eval.value()[0])

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
    logging.info('max acc:' ,max_acc)
    logging.info('max acc1:' ,max_acc1)
    logging.info('max acc2:' ,max_acc2)
    logging.info('max acc peak:' ,max_acc_peak)
    logging.info('max acc peak1:' ,max_acc_peak1)
    logging.info('max acc peak2:' ,max_acc_peak2)
    #if acc > 0.6:
    #    pdb.set_trace()
    #pdb.set_trace()
    with open('./results/'+opt.output_dir.split('/')[-1]+'.txt','a') as f:
        f.write(str(epoch)+'\t'+str(acc)+'\t'+str(acc1)+'\t'+str(acc2)+'\t'+str(facc)+'\t'+str(facc1)+'\t'+str(facc2)+'\n')
    return max_acc,max_acc1,max_acc2,max_acc_peak,max_acc_peak1,max_acc_peak2


max_accs = ''
#eval_epoch(0,eval_dataset)

if opt.test:
    for epoch in range(1,opt.start_epoch+1):
        if epoch % opt.interval == 0:
        #if epoch%10 ==0:
            checkpoint = torch.load(opt.output_dir + '/ckpt_'+str(epoch)+'.pth')
            s.load_state_dict (checkpoint['model_state_dict']) 
            
            max_accs = eval_ours_all(epoch,s,opt,'',max_accs)
            logging.info(opt.output_dir)

     
#input()
for epoch in range(opt.start_epoch,opt.num_epochs):
    #train_dataset.load_next_chunk()
    #train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    train_epoch(epoch,optimizer,criterion,train_dataloader)
    if epoch % opt.interval == 0:
        max_accs = eval_ours_all(epoch,s,opt,'',max_accs)
    logging.info(opt.output_dir)





