import torch
import numpy as np
import time, pdb, argparse, subprocess, os
import cv2
#import python_speech_features

from scipy import signal
from scipy.io import wavfile
#from SyncNetModel import *
import pickle
import random
import math
import numpy
from scipy.signal import medfilt
import itertools
from .read_ours_gt import get_ours_label_all
import logging



label_list = [1,2,3,4,5]
perm_list = []
for i in range(1,len(label_list)+1):
	iter = itertools.permutations(label_list,i)
	perm_list.extend(list(iter))

def calc_pdist(feat1, feat2, vshift=10):
    
    win_size = vshift*2+1

    #feat2 = feat2.unsqueeze(0)
    #pdb.set_trace()
    #feat2p = torch.nn.functional.pad(feat2,(0,0,0,0,vshift,vshift),mode = 'replicate')
    #logging.info('feat shape',feat2.shape)
    feat2p = torch.nn.functional.pad(feat2,(0,0,vshift,vshift))

    dists = []

    for i in range(0,len(feat1)):

        dists.append(torch.nn.functional.pairwise_distance(feat1[[i],:].repeat(win_size, 1), feat2p[i:i+win_size,:]))

    return dists

def calc_pdist_npy(feat1, feat2, vshift=10):
    
    win_size = vshift*2+1
    #pdb.set_trace()
    #feat2p = torch.nn.functional.pad(feat2,(0,0,vshift,vshift),mode = 'replicate')
    feat2p = numpy.pad(feat2,((vshift,vshift),(0,0)),'edge')
    #feat2p = torch.nn.functional.pad(feat2,(0,0,vshift,vshift))
    feat2p = torch.from_numpy(feat2p)#.cuda()

    dists = []

    for i in range(0,len(feat1)):

        dists.append(torch.nn.functional.pairwise_distance(feat1[[i],:].repeat(win_size, 1), feat2p[i:i+win_size,:]))

    return dists


def load_npy(path_file):
    obj =  np.load(path_file,'r')
    #obj = obj / 255.
    #pdb.set_trace()
    #logging.info(obj.shape)
    obj = obj[:,::-1,:,:]
    return obj

def load_npy_lip(path_file):
    obj =  np.load(path_file,'r')
    #obj = obj / 255.
    #pdb.set_trace()
    #logging.info(obj.shape)
    #obj = obj[:,::-1,:,:]
    if not (obj.shape[2] ==112):
        obj = np.transpose(obj, (2,0,1))
    obj = np.expand_dims(obj, axis=1)
    return obj



def eval_ours_item(epoch,s,opt,labels,cc,ims,test_name):
    s.eval()
    losses = []
    
    dists = []
    fdists = []
    dists_mf = []
    fdists_mf = []
    

    #cc = np.load(audio_dir)
    for iter,im in enumerate(ims):
    #if len(video_dir) > 0:
        #video_dir = opt.data_dir+ video_dir#.replace()
        #time1 = times[iter]
        #time2 = times[iter + 1]
        
        #opt.reference =  video_dir[video_dir.rfind('/')+1:-4] + '_' + str(iter)
        #im = load_npy_lip(video_dir)  
        #if im.shape[0]*4 > cc.shape[-1]:
        #    print('longggggggggggggggggg image',iter)
        #    im = im[:int(cc.shape[-1]/4),:,:,:]     
        offset, conf, dist,fdist, minval = s.evaluate_diar_npy(opt,im,cc)
        #pdb.set_trace()
        #dist = dist[:,0]
        dists.append(dist)
        fdists.append(fdist)
        
        dist_mf = medfilt(dist, kernel_size=19)
        fdist_mf = medfilt(fdist, kernel_size=19)
        dists_mf.append(dist_mf)
        fdists_mf.append(fdist_mf)

        

            #offsets.append(numpy.absolute(offset))
        
            #confs.append(conf)
                #minvals.append(minval)
                #names.append(opt.reference)
    #
    dists = numpy.asarray(dists)
    fdists = numpy.asarray(fdists)
    dists_mf = numpy.asarray(dists_mf)
    fdists_mf = numpy.asarray(fdists_mf)
    #pdb.set_trace()
    #logging.info('len dist',len(dists))
    #logging.info('num of video',len(video_dirs))

    #logging.info('median',dists.median())
    #logging.info('mean',dists.mean())
    #logging.info('max',dists.max())
    #logging.info('min',dists.min())

    min_dists = np.min(dists,0)
    min_dists_mf = np.min(dists_mf,0)
    argmin_dists = np.argmin(dists,0) + 1
    argmin_dists_mf = np.argmin(dists_mf,0) + 1


    #argmin_dists[min_dists>dists.mean()] = 0

    num_silence = len(labels) - np.count_nonzero(labels)


    #argmin_dists[min_dists>dists.mean()] = 0

    import copy
    argmin_dists1 = copy.deepcopy(argmin_dists)
    argmin_dists1_mf = copy.deepcopy(argmin_dists_mf)
    argmin_dists2 = copy.deepcopy(argmin_dists)
    argmin_dists1[labels==0] = 0
    argmin_dists1_mf[labels==0] = 0
    argmin_dists2[argmin_dists>dists.mean()] = 0

    max_fdists = np.max(fdists,0)
    argmax_fdists = np.argmax(fdists,0) + 1
    argmax_fdists1 = copy.deepcopy(argmax_fdists)
    argmax_fdists2 = copy.deepcopy(argmax_fdists)

    argmax_fdists2[max_fdists<fdists.mean()] = 0
    argmax_fdists1[labels==0] = 0


    
    #pdb.set_trace()
    correct = np.sum(labels == argmin_dists)#[:len(label)])
    correct_mf = np.sum(labels == argmin_dists_mf)#[:len(label)])
    #argmin_dists[min_dists>dists.mean()] = 0
    correct1 = np.sum(labels == argmin_dists1)#[:len(label)])
    correct1_mf = np.sum(labels == argmin_dists1_mf)#[:len(label)])
    correct2 = np.sum(labels == argmin_dists2)#[:len(label)])
    #correct3 = np.sum(labels == argmin_dists2)#[:len(label)])
    fcorrect = np.sum(labels == argmax_fdists)#[:len(label)])
    #argmin_dists[min_dists>dists.mean()] = 0
    fcorrect1 = np.sum(labels == argmax_fdists1)#[:len(label)])
    fcorrect2 = np.sum(labels == argmax_fdists2)#[:len(label)])

    #pdb.set_trace()
    if len(dists) == len(ims):
        results_dir = os.path.join('./results/',opt.output_dir.split('/')[-1])
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        numpy.save(os.path.join(results_dir,test_name+'_dists.npy'),dists)
        numpy.save(os.path.join(results_dir,test_name+'_fdists.npy'),fdists)
    else:
        logging.info('mis match!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

            #
        #loss_meter_eval.add(probs)
    #viz.plot_many_stack({'eval_loss': loss_meter_eval.value()[0]})
    #logging.info('epoch ours',epoch)
    #pdb.set_trace()
    acc = correct/float(len(labels))
    acc_mf = correct_mf/float(len(labels))
    acc2 = correct2/float(len(labels))
    acc1 = (correct1-num_silence)/float((len(labels)-num_silence))
    acc1_mf = (correct1_mf-num_silence)/float((len(labels)-num_silence))
    #acc1 = correct1/float(len(labels))
    #acc3 = correct2/float(len(labels))
    facc = fcorrect/float(len(labels))
    facc2 = fcorrect2/float(len(labels))
    facc1 = (fcorrect1-num_silence)/float((len(labels)-num_silence))
    #pdb.set_trace()
    #(np.count_nonzero(idxs) + len(idxs_neg) - np.count_nonzero(idxs_neg))/float(len(idxs)+len(idxs_neg))
    #logging.info('eval offset:' ,loss_meter_eval.value()[0])
    logging.info('acc item:{0}'.format(acc))
    logging.info('acc item mf:{0}'.format(acc_mf))
    logging.info('acc item1:{0}'.format(acc1))
    logging.info('acc item1_mf:{0}'.format(acc1_mf))
    logging.info('acc item2:{0}'.format(acc2))
    logging.info('acc item peak:{0}'.format(facc))
    logging.info('acc item peak1:{0}'.format(facc1))
    logging.info('acc item peak1:{0}'.format(facc2))
    
    #pdb.set_trace()
    #(np.count_nonzero(idxs) + len(idxs_neg) - np.count_nonzero(idxs_neg))/float(len(idxs)+len(idxs_neg))
    #logging.info('eval offset:' ,loss_meter_eval.value()[0])
    '''
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
    '''
    #if acc > 0.6:
    #    pdb.set_trace()
    #pdb.set_trace()
    with open('./results/'+opt.output_dir.split('/')[-1]+'.txt','a') as f:
        f.write(str(epoch)+'\t'+str(acc)+'\t'+str(acc1)+'\t'+str(acc2)+'\t'+str(facc)+'\t'+str(facc1)+'\t'+str(facc2)+'\n')
    return acc,acc1,acc2,facc,facc1,facc2,acc_mf,acc1_mf


def eval_ours_all(epoch,s,opt,val_dataset,max_accs=''):
    logging.info('epoch {0}'.format(epoch))


    if not max_accs == '':
        max_acc,max_acc1,max_acc2,max_acc_peak,max_acc_peak1,max_acc_peak2,max_acc_mf,max_acc1_mf,max_epoch = max_accs
    else:
        max_acc,max_acc1,max_acc2,max_acc_peak,max_acc_peak1,max_acc_peak2,max_acc_mf,max_acc1_mf,max_epoch = (0,0,0,0,0,0,0,0,0)

    if val_dataset == "":
        from ours_dataset import AudioVideoDataset_eval_ours as dataset
        val_dataset = dataset(opt,audio =opt.audio,video = opt.video)
        logging.info('loaded val dataset AudioVideoDataset_ours')

    s.eval()
    losses = []
    #pdb.set_trace()
    accs = 0.
    accs_mf = 0.
    accs1 = 0.
    accs1_mf = 0.
    accs2 = 0.
    faccs = 0.
    faccs1 = 0.
    faccs2 = 0.
    for iteration, data in enumerate(val_dataset):
        logging.info('iteration{0}'.format(iteration))
        ims,cc,labels,test_name = data
        if opt.video == 'lip_new':
            best_acc = 0.
            best_acc1 = 0.
            best_acc2 = 0.
            best_facc = 0.
            best_facc1 = 0.
            best_facc2 = 0.
            best_acc_mf = 0.
            best_acc1_mf = 0.
            best_acc_all = 0.
            
            for perm in perm_list:
                if len(perm) == len(ims):
                    label_dict ={}
                    for j in range(len(perm)):
                        label_dict['person'+str(j)] = perm[j]
                    labels = get_ours_label_all(os.path.join(opt.eval_data_dir,'labels',test_name +'.txt'),len(labels),label_dict)
                    acc,acc1,acc2,facc,facc1,facc2,acc_mf,acc1_mf = eval_ours_item(epoch,s,opt,labels,cc,ims,test_name)
                    if acc + acc1 + acc_mf + acc1_mf > best_acc_all:
                        best_acc = acc
                        best_acc1 = acc1
                        best_acc2 = acc2
                        best_facc = facc
                        best_facc1 = facc1
                        best_facc2 = facc2
                        best_acc_mf = acc_mf
                        best_acc1_mf = acc1_mf
                        best_acc_all = acc + acc1
            accs += best_acc
            accs_mf += best_acc_mf
            accs1 += best_acc1
            accs1_mf += best_acc1_mf
            accs2 += best_acc2
            faccs += best_facc
            faccs1 += best_facc1
            faccs2 += best_facc2
        else:
            acc,acc1,acc2,facc,facc1,facc2,acc_mf,acc1_mf = eval_ours_item(epoch,s,opt,labels,cc,ims,test_name)
            accs += acc
            accs_mf += acc_mf
            accs1 += acc1
            accs1_mf += acc1_mf
            accs2 += acc2
            faccs += facc
            faccs1 += facc1
            faccs2 += facc2
        

    accs = accs / len(val_dataset)
    accs_mf = accs_mf / len(val_dataset)
    accs1 = accs1 / len(val_dataset)
    accs1_mf = accs1_mf / len(val_dataset)
    accs2 = accs2 / len(val_dataset)
    faccs = faccs / len(val_dataset)
    faccs1 = faccs1 / len(val_dataset)
    faccs2 = faccs2 / len(val_dataset)
    logging.info('eval ours accuracy:{0}'.format(accs))
    logging.info('eval ours accuracy1:{0}'.format(accs1))
    logging.info('eval ours accuracy2:{0}'.format(accs2))
    logging.info('eval ours accuracy peak:{0}'.format(faccs))
    logging.info('eval ours accuracy peak1:{0}'.format(faccs1))
    logging.info('eval ours accuracy peak1:{0}'.format(faccs2))

    if max_acc < accs:
        max_acc = accs
    if max_acc1 < accs1:
        max_acc1 = accs1
    if max_acc_mf < accs_mf:
        max_acc_mf = accs_mf
    if max_acc1_mf < accs1_mf:
        max_acc1_mf = accs1_mf
        max_epoch = epoch
    if max_acc2 < accs2:
        max_acc2 = accs2
    if max_acc_peak < faccs:
        max_acc_peak = faccs
    if max_acc_peak1 < faccs1:
        max_acc_peak1 = faccs1
    if max_acc_peak2 < faccs2:
        max_acc_peak2 = faccs2
    logging.info('current best accuracy:{0}'.format(max_acc))
    logging.info('current best accuracy mf:{0}'.format(max_acc_mf))
    logging.info('current best accuracy1:{0}'.format(max_acc1))
    logging.info('current best accuracy1 mf:{0}'.format(max_acc1_mf))
    logging.info('current best accuracy2:{0}'.format(max_acc2))
    logging.info('current best accuracy peak:{0}'.format(max_acc_peak))
    logging.info('current best accuracy peak1:{0}'.format(max_acc_peak1))
    logging.info('current best accuracy peak1:{0}'.format(max_acc_peak2))
    logging.info('current best epoch:{0}'.format(max_epoch))

    print('current best accuracy:{0}'.format(max_acc))
    print('current best accuracy mf:{0}'.format(max_acc_mf))
    print('current best accuracy1:{0}'.format(max_acc1))
    print('current best accuracy1 mf:{0}'.format(max_acc1_mf))

    return (max_acc,max_acc1,max_acc2,max_acc_peak,max_acc_peak1,max_acc_peak2,max_acc_mf,max_acc1_mf,max_epoch)

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
    #logging.info('eval offset:{0}'.format(loss_meter_eval.value()[0])
    logging.info('eval ours accuracy:{0}'.format(acc))
    logging.info('eval ours accuracy1:{0}'.format(acc1))
    logging.info('eval ours accuracy2:{0}'.format(acc2))
    logging.info('eval ours accuracy peak:{0}'.format(facc))
    logging.info('eval ours accuracy peak1:{0}'.format(facc1))
    logging.info('eval ours accuracy peak1:{0}'.format(facc2))
    #pdb.set_trace()
    #(np.count_nonzero(idxs) + len(idxs_neg) - np.count_nonzero(idxs_neg))/float(len(idxs)+len(idxs_neg))
    #logging.info('eval offset:{0}'.format(loss_meter_eval.value()[0])

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
    logging.info('max acc:{0}'.format(max_acc))
    logging.info('max acc1:{0}'.format(max_acc1))
    logging.info('max acc2:{0}'.format(max_acc2))
    logging.info('max acc peak:{0}'.format(max_acc_peak))
    logging.info('max acc peak1:{0}'.format(max_acc_peak1))
    logging.info('max acc peak2:{0}'.format(max_acc_peak2))
    #if acc > 0.6:
    #    pdb.set_trace()
    pdb.set_trace()
    with open('./results/'+opt.output_dir.split('/')[-1]+'.txt','a') as f:
        f.write(str(epoch)+'\t'+str(acc)+'\t'+str(acc1)+'\t'+str(acc2)+'\t'+str(facc)+'\t'+str(facc1)+'\t'+str(facc2)+'\n')
    return max_acc,max_acc1,max_acc2,max_acc_peak,max_acc_peak1,max_acc_peak2

def read_silence(silence_dir):
    #import pdb
    #pdb.set_trace()
    with open(silence_dir,'r') as f:
        silence = f.read()
    start,end = silence.split('#')
    start1,end1 = start.split('-')
    start2,end2 = end.split('-')
    end2 = end2.replace('\n','')
    return [int(round(float(start1)*25)),int(round(float(end1)*25)),int(round(float(start2)*25)),int(round(float(end2)*25))]

def check_silence(silence,start_frame ,frame_len):
    if not silence:
        return False
    end_frame = start_frame+ frame_len

    if (start_frame>= silence[0] and end_frame <= silence[1]) or (start_frame>= silence[2] and end_frame <= silence[3]):
        return True
    else:
        return False

def get_dists(im_in,cc_in,s,opt):
    #pdb.set_trace()
    dists = []
    im_in_shift = im_in[:,:,opt.vshift:opt.frame_len- opt.vshift,:,:]
    for i in range(opt.frame_len- 5):
        cc_in_shift = cc_in[:,:,:,i*4:(i+5)*4]
        im_out_shift = s.forward_lip(im_in_shift.cuda())
        cc_out_shift = s.forward_aud(cc_in_shift.cuda())
        dists.append(torch.nn.functional.pairwise_distance(im_out_shift,cc_out_shift).detach().cpu().numpy())
    dists = np.array(dists)
    idx_smallest = np.argmin(dists[opt.frame_len/2-2:opt.frame_len/2+3],axis = 0)
    idx_smallest_1 = np.argmin(dists,axis = 0)
    idx_2nd_smallest = np.argpartition(dists,2,axis = 0)
    idx_2nd_smallest = idx_2nd_smallest[1,:]
    mask = (idx_smallest_1==idx_smallest)
    idx_smallest_1[mask] = idx_2nd_smallest[mask]
    #return idx_smallest,idx_2nd_smallest
    return torch.stack([torch.arange(i*4,(i+5)*4) for i in idx_smallest]),torch.stack([torch.arange(i*4,(i+5)*4) for i in idx_smallest_1])



def get_dists_shift(im_in,cc_in,s,opt):
    #pdb.set_trace()
    dists = []
    im_in_shift = im_in[:,:,3:8,:,:]
    im_out_shift = s.forward_lip(im_in_shift.cuda())
    for i in range(opt.frame_len- 5):
        cc_in_shift = cc_in[:,:,:,i*4:(i+5)*4]
        #im_out_shift = s.forward_lip(im_in_shift.cuda())
        cc_out_shift = s.forward_aud(cc_in_shift.cuda())
        dists.append(torch.nn.functional.pairwise_distance(im_out_shift,cc_out_shift).detach().cpu().numpy())
    dists = np.array(dists)
    idx_smallest = np.argmin(dists,axis = 0)
    #idx_smallest_1 = np.argmin(dists,axis = 0)
    #idx_2nd_smallest = np.argpartition(dists,2,axis = 0)
    #idx_2nd_smallest = idx_2nd_smallest[1,:]
    #mask = (idx_smallest_1==idx_smallest)
    #idx_smallest_1[mask] = idx_2nd_smallest[mask]
    #return idx_smallest,idx_2nd_smallest
    return torch.stack([torch.arange(i*4,(i+5)*4) for i in idx_smallest])#,torch.stack([torch.arange(i*4,(i+5)*4) for i in idx_smallest_1])



def get_dists_shift_replaced(im_in,cc_in,cc_in_replaced,s,opt):
    #pdb.set_trace()
    dists = []
    im_in_shift = im_in[:,:,opt.vshift:opt.frame_len- opt.vshift,:,:]
    im_out_shift = s.forward_lip(im_in_shift.cuda())
    for i in range(opt.frame_len- 5):
        cc_in_shift = cc_in[:,:,:,i*4:(i+5)*4]
        #im_out_shift = s.forward_lip(im_in_shift.cuda())
        cc_out_shift = s.forward_aud(cc_in_shift.cuda())
        dists.append(torch.nn.functional.pairwise_distance(im_out_shift,cc_out_shift).detach().cpu().numpy())
    for i in range(opt.frame_len- 5):
        cc_in_shift = cc_in_replaced[:,:,:,i*4:(i+5)*4]
        #im_out_shift = s.forward_lip(im_in_shift.cuda())
        cc_out_shift = s.forward_aud(cc_in_shift.cuda())
        dists.append(torch.nn.functional.pairwise_distance(im_out_shift,cc_out_shift).detach().cpu().numpy())

    dists = np.array(dists)
    idx_smallest = np.argmin(dists[opt.frame_len/2-2:opt.frame_len/2+3],axis = 0)
    idx_smallest_1 = np.argmin(dists,axis = 0)
    idx_2nd_smallest = np.argpartition(dists,2,axis = 0)
    idx_2nd_smallest = idx_2nd_smallest[1,:]
    mask = (idx_smallest_1==idx_smallest)
    idx_smallest_1[mask] = idx_2nd_smallest[mask]
    #ccs = torch.cat((cc_in,cc_in_replaced),0)
    return torch.stack([torch.arange(i*4,(i+5)*4) for i in idx_smallest]),torch.stack([torch.arange(i*4,(i+5)*4) for i in idx_smallest_1])

def get_dists_shift_replaced_pre(im_in,cc_in,cc_in_replaced,s,opt):
    #pdb.set_trace()
    dists = []
    im_in_shift = im_in[:,:,opt.vshift:opt.frame_len- opt.vshift,:,:]
    im_out_shift = s.forward_lip(im_in_shift.cuda())
    for i in range(opt.frame_len- 5):
        cc_in_shift = cc_in[:,:,:,i*4:(i+5)*4]
        
        cc_out_shift = s.forward_aud(cc_in_shift.cuda())
        dists.append(torch.nn.functional.pairwise_distance(im_out_shift,cc_out_shift).detach().cpu().numpy())
    for i in range(opt.frame_len- 5):
        cc_in_shift = cc_in_replaced[:,:,:,i*4:(i+5)*4]
        #im_out_shift = s.forward_lip(im_in_shift.cuda())
        cc_out_shift = s.forward_aud(cc_in_shift.cuda())
        dists.append(torch.nn.functional.pairwise_distance(im_out_shift,cc_out_shift).detach().cpu().numpy())

    dists = np.array(dists)
    idx_smallest = np.argmin(dists[opt.frame_len/2:opt.frame_len/2+1],axis = 0)
    idx_smallest_1 = np.argmin(dists,axis = 0)
    idx_2nd_smallest = np.argpartition(dists,2,axis = 0)
    idx_2nd_smallest = idx_2nd_smallest[1,:]
    mask = (idx_smallest_1==idx_smallest)
    idx_smallest_1[mask] = idx_2nd_smallest[mask]
    #ccs = torch.cat((cc_in,cc_in_replaced),0)
    return torch.stack([torch.arange(i*4,(i+5)*4) for i in idx_smallest]),torch.stack([torch.arange(i*4,(i+5)*4) for i in idx_smallest_1])

    #return idx_smallest,idx_2nd_smallest