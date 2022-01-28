#!/usr/bin/python
#-*- coding: utf-8 -*-

import torch
import numpy
import time, pdb, argparse, subprocess, pickle, os
import cv2
import pdb
from scipy import signal
import numpy as np

# ==================== PARSE ARGUMENT ====================

parser = argparse.ArgumentParser(description = "SyncNet");
parser.add_argument('--initial_model', type=str, default="data/syncnet.model", help='');
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

#pdb.set_trace = lambda: None

# ==================== LOAD FILES ====================

with open(os.path.join(opt.work_dir,opt.reference,'tracks.pckl'), 'r') as fil:
    tracks = pickle.load(fil)

with open(os.path.join(opt.work_dir,opt.reference,'activesd.pckl'), 'r') as fil:
    dists = pickle.load(fil)

# ==================== SMOOTH FACES ====================

faces = [ [] for i in range(1000000) ]

frame_peak = []
frame_peak_mf=[]

#pdb.set_trace()

for ii, track in enumerate(tracks):

	mean_dists =  numpy.mean(numpy.stack(dists[ii],1),1)
	minidx = numpy.argmin(mean_dists,0)
	minval = mean_dists[minidx] 

	fdist   	= numpy.stack([dist[minidx] for dist in dists[ii]])
	frame_peak.append(fdist)
	fdist   	= numpy.pad(fdist, (2,4), 'constant', constant_values=10)
	fdist_mf	= signal.medfilt(fdist,kernel_size=19)

	for ij, frame in enumerate(track[0][0].tolist()) :
		faces[frame].append([ii, fdist_mf[ij], track[1][0][ij], track[1][1][ij], track[1][2][ij]])

	if len(fdist_mf)<503:
		fdist_mf   	= numpy.pad(fdist_mf, (503-len(fdist_mf),0), 'constant', constant_values=10)
	frame_peak_mf.append(fdist_mf)
	


frame_peak = numpy.asarray(frame_peak)
frame_peak_mf = numpy.asarray(frame_peak_mf)
frame_peak_mf   	= numpy.pad(frame_peak_mf, ((0,0),(0,1)), 'constant', constant_values=10)


from utils.read_ours_gt import get_ours_label
label = get_ours_label('./data/test_real_25/gt.txt',{'austin':3,'raymond':2,'yong':1})
num_silence = len(label) - np.count_nonzero(label)
min_dists = np.min(frame_peak_mf,0)
argmin_dists = np.argmin(frame_peak_mf,0) + 1
import copy
argmin_dists1 = copy.deepcopy(argmin_dists)
#argmin_dists[min_dists>frame_peak_mf.mean()] = 0
argmin_dists1[label == 0] = 0
correct = np.sum(label == argmin_dists)#[:len(label)])
correct1 = np.sum(label == argmin_dists1)#[:len(label)])
acc = correct/float(len(label))
acc1 = (correct1-num_silence)/float((len(label)-num_silence))
print('eval ours accuracy:' ,acc)
print('eval ours accuracy:' ,acc1)

numpy.save(opt.data_dir+'/dist.npy',frame_peak)
numpy.save(opt.data_dir+'/dist_mf.npy',frame_peak_mf)
pdb.set_trace()

# ==================== ADD DETECTIONS TO VIDEO ====================

cap = cv2.VideoCapture(os.path.join(opt.avi_dir,opt.reference,'video.avi'))
fw = int(cap.get(3))
fh = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
vOut = cv2.VideoWriter(os.path.join(opt.avi_dir,opt.reference,'video_only.avi'), fourcc, cap.get(5), (fw,fh))

frame_num=0

min_dists = []
for frame in faces:
	if len(frame)>0:
		dists = numpy.array(frame)
		min_dist = numpy.min(dists,0)
		min_dists.append(min_dist[1])
	else:
		min_dists.append(0)


while True:
	ret, image = cap.read()
	if ret == 0:
		break

	#pdb.set_trace()

	for i,face in enumerate(faces[frame_num]):

		'''
		if i == 0 :
			cv2.rectangle(image,(int(face[3]-face[2]),int(face[4]-face[2])),(int(face[3]+face[2]),int(face[4]+face[2])),(0,255,0),3)
			cv2.putText(image,'Track %d, L2 Dist %.3f'%(face[0],face[1]), (int(face[3]-face[2]),int(face[4]-face[2])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
		elif i == 1 :
			cv2.rectangle(image,(int(face[3]-face[2]),int(face[4]-face[2])),(int(face[3]+face[2]),int(face[4]+face[2])),(0,0,255),3)
			cv2.putText(image,'Track %d, L2 Dist %.3f'%(face[0],face[1]), (int(face[3]-face[2]),int(face[4]-face[2])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
		else:
			cv2.rectangle(image,(int(face[3]-face[2]),int(face[4]-face[2])),(int(face[3]+face[2]),int(face[4]+face[2])),(255,0,0),3)
			cv2.putText(image,'Track %d, L2 Dist %.3f'%(face[0],face[1]), (int(face[3]-face[2]),int(face[4]-face[2])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)

		'''

		if min_dists[frame_num] == face[1]:
			#pdb.set_trace()

			cv2.rectangle(image,(int(face[3]-face[2]),int(face[4]-face[2])),(int(face[3]+face[2]),int(face[4]+face[2])),(0,255,0),3)
			cv2.putText(image,'Track %d, L2 Dist %.3f'%(face[0],face[1]), (int(face[3]-face[2]),int(face[4]-face[2])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
		
		else:
			cv2.rectangle(image, (int(face[3] - face[2]), int(face[4] - face[2])),
						  (int(face[3] + face[2]), int(face[4] + face[2])), (255, 204, 229), 3)
			cv2.putText(image, 'Track %d, L2 Dist %.3f' % (face[0], face[1]),
						(int(face[3] - face[2]), int(face[4] - face[2])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
						(255, 255, 255), 2)
		
	vOut.write(image)

	print('Frame %d'%frame_num)

	frame_num+=1

cap.release()
vOut.release()

# ========== CROP AUDIO FILE ==========

command = ("ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (os.path.join(opt.avi_dir,opt.reference,'video.avi'),os.path.join(opt.avi_dir,opt.reference,'audio_only.avi'))) 
output = subprocess.call(command, shell=True, stdout=None)

# ========== COMBINE AUDIO AND VIDEO FILES ==========

command = ("ffmpeg -y -i %s -i %s -c:v copy -c:a copy %s" % (os.path.join(opt.avi_dir,opt.reference,'video_only.avi'),os.path.join(opt.avi_dir,opt.reference,'audio_only.avi'),os.path.join(opt.avi_dir,opt.reference,'video_out.avi'))) #-async 1 
output = subprocess.call(command, shell=True, stdout=None)


