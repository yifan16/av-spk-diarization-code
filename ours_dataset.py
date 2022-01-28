# ## python lib
import os, sys, math, random, glob, cv2
import numpy as np
import pdb
# ## torch lib
import torch
import torch.utils.data as data
import python_speech_features
from scipy import signal
from scipy.io import wavfile
# ## custom lib
import utils
import scipy
from os import walk
import ipdb
from scipy.io import wavfile
import pickle
import random
# import numpy
import torch.nn.functional as F
from utils.read_ours_gt import get_ours_label,get_ours_label_all
from utils.tools import read_silence,check_silence


def get_random_video(ex_list, max_idx, audio_video_list, frame_len):
    random_idx = random.randint(0, max_idx)
    while True:
        while random_idx in ex_list:
            random_idx = random.randint(0, max_idx)
        random_video = audio_video_list[random_idx]
        random_im = load_npy(random_video)
        #try:
        if len(random_im) - frame_len > 0:
            start_frame = random.randint(0, len(random_im) - frame_len)
            break
    #except:
    #pdb.set_trace()
    im_batch = random_im[start_frame:start_frame + frame_len, :, :, :]
    #print('im batch', im_batch.shape)
        
    im_in = np.transpose(im_batch, (1, 0, 2, 3))
    #im_in = random_im[:, :, start_frame :(start_frame + self.opt.frame_len)]
    return im_in, random_idx

def get_random_video_lip(ex_list, max_idx, audio_video_list, frame_len):
    random_idx = random.randint(0, max_idx)
    while True:
        while random_idx in ex_list:
            random_idx = random.randint(0, max_idx)
        random_video = audio_video_list[random_idx]
        random_im = load_npy_lip(random_video)
        #try:
        if len(random_im) - frame_len > 0:
            start_frame = random.randint(0, len(random_im) - frame_len)
            break
    #except:
    #pdb.set_trace()
    im_batch = random_im[start_frame:start_frame + frame_len, :, :, :]
    #print('im batch', im_batch.shape)
        
    im_in = np.transpose(im_batch, (1, 0, 2, 3))
    #im_in = random_im[:, :, start_frame :(start_frame + self.opt.frame_len)]
    return im_in, random_idx


def get_random_audio(ex_list, max_idx, audio_video_list, frame_len,audio = 'audio_feat',video = 'face_crop'):
    random_idx = random.randint(0, max_idx)
    while True:
        while random_idx in ex_list :
            random_idx = random.randint(0, max_idx)
        random_audio = audio_video_list[random_idx].replace(video, audio)
        #print('random_audio', random_audio)
        #print(random_idx)
        random_cc = np.load(random_audio)
        #try:
        if random_cc.shape[2] - frame_len*4 > 0:
            start_frame = random.randint(0, random_cc.shape[2] - frame_len*4-1)
            break
    #except:
    #    pdb.set_trace()
    cc_in = random_cc[:, :, start_frame:start_frame + frame_len * 4]
    return cc_in, random_idx

def get_random_audio_lip(ex_list, max_idx, audio_video_list, frame_len,audio = 'audio_feat',video = 'center_crop_112'):
    #pdb.set_trace()
    random_idx = random.randint(0, max_idx)
    while True:
        while random_idx in ex_list :
            random_idx = random.randint(0, max_idx)
        random_audio = audio_video_list[random_idx].replace(video, audio)
        #print('random_audio', random_audio)
        print(random_idx)
        random_cc = np.load(random_audio)
        #try:
        if random_cc.shape[2] - frame_len*4 > 0:
            start_frame = random.randint(0, random_cc.shape[2] - frame_len*4-1)
            break
    #except:
    #    pdb.set_trace()
    cc_in = random_cc[:, :, start_frame:start_frame + frame_len * 4]
    return cc_in, random_idx

def get_random_audio_chunk(ex_list, audios, opt,audio = 'audio_feat',video = 'center_crop_112'):
    #pdb.set_trace()
    max_idx = len(audios)-1
    random_idx = random.randint(0, max_idx)
    while True:
        while random_idx in ex_list :
            random_idx = random.randint(0, max_idx)
        random_cc = audios[random_idx]
        #print('random_audio', random_audio)
        #print(random_idx)
        #random_cc = np.load(random_audio)
        #try:
        if random_cc.shape[-1] - opt.frame_len*4 >= 0:
            start_frame = random.randint(0, random_cc.shape[-1] - opt.frame_len*4)
            break
    #except:
    #    pdb.set_trace()
    cc_in = random_cc[:, :, start_frame:start_frame + opt.frame_len * 4]
    return cc_in, random_idx

def get_random_video_chunk(ex_list, videos, opt,audio = 'audio_feat',video = 'center_crop_112'):
    #pdb.set_trace()
    max_idx = len(videos)-1
    random_idx = random.randint(0, max_idx)
    while True:
        while random_idx in ex_list :
            random_idx = random.randint(0, max_idx)
        random_im = videos[random_idx]
        #print('random_audio', random_audio)
        #print(random_idx)
        #random_cc = np.load(random_audio)
        #try:
        #print(random_im.shape)
        if random_im.shape[0] - opt.frame_len >= 0:
            start_frame = random.randint(0, random_im.shape[0] - opt.frame_len)
            break
    #except:
    #    pdb.set_trace()
    im_in = random_im[start_frame:start_frame + opt.frame_len,:,:,:]
    im_in = np.transpose(im_in, (1, 0, 2, 3))

    return im_in, random_idx

def load_pickle(path_file):
    with open(path_file, "rb") as f:
        obj = pickle.load(f)
        obj = obj/255.
        obj = np.transpose(obj, (3, 2, 0, 1))
        obj = np.reshape(obj, (-1, 3, 224, 224))
        return obj

def load_npy(path_file):
    obj =  np.load(path_file, 'r')
    #obj = obj / 255.
    #pdb.set_trace()
    #print(obj.shape)
    obj = obj[:, ::-1, :, :]
    return obj

def load_npy_lip(path_file):
    obj =  np.load(path_file, 'r')
    #obj = obj / 255.
    #pdb.set_trace()
    #print(obj.shape)
    #obj = obj[:, ::-1, :, :]
    if not (obj.shape[2] == 112):
        obj = np.transpose(obj, (2, 0, 1))
    obj = np.expand_dims(obj, axis = 1)
    return obj


def load_npy_lip_norm(path_file):
    obj =  np.load(path_file, 'r')
    obj = obj / 255. - 0.5
    #pdb.set_trace()
    #print(obj.shape)
    #obj = obj[:, ::-1, :, :]
    if not (obj.shape[2] == 112):
        obj = np.transpose(obj, (2, 0, 1))
    obj = np.expand_dims(obj, axis = 1)
    return obj

def load_pickle1(path_file):
    with open(path_file, "rb") as f:
        obj = pickle.load(f)
        obj = obj/255.
        return obj

class RandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.ch, self.cw = crop_size
        ih, iw = image_size

        self.h1 = random.randint(0, ih - self.ch)
        self.w1 = random.randint(0, iw - self.cw)

        self.h2 = self.h1 + self.ch
        self.w2 = self.w1 + self.cw

    def __call__(self, img):
        if len(img.shape) == 3:
            return img[self.h1: self.h2, self.w1: self.w2, :]
        else:
            return img[self.h1: self.h2, self.w1: self.w2]




class AudioVideoDataset_eval_ours(data.Dataset):
    def __init__(self, opt, audio = '39',video = 'lip',seed = np.random.seed(0)):
        super(AudioVideoDataset_eval_ours, self).__init__()


        #self.chunk_index = 0
        #self.mode = mode
        #self.type =  type
        self.opt = opt
        if audio == '39':
            self.audio_folder = 'audio_feat_39'
        else:
            self.audio_folder = 'audio_feat'

        if video == 'lip':
            self.video_folder = 'crop_face_112'
        elif video =='lip_new':
            self.video_folder = 'crop_face_112_new'
        else:
            self.video_folder = 'face_crop_224'


        #pdb.set_trace()
            
        #if self.mode == 'eval_ours':
        video_list = []
        audio_list = []
        label_list = []
        video_list.extend(glob.glob(opt.eval_data_dir + '/'+self.video_folder+'/*.npy'))
        audio_list.extend(glob.glob(opt.eval_data_dir + '/'+self.audio_folder +'/*.npy'))
        label_list.extend(glob.glob(opt.eval_data_dir + '/labels/*.txt'))
        self.audio_video_list_ = [i for i in video_list if ((i.replace(self.video_folder, self.audio_folder)[:-12]+'.npy' in audio_list) and (i.replace(self.video_folder, 'labels')[:-12]+'.txt' in label_list))]
        videos = set([i[:-12] for i in self.audio_video_list_])
        self.audio_video_list = []
        for video in videos:
            self.audio_video_list.append([i for i in self.audio_video_list_ if i[:-12]==video])
        self.num = len(self.audio_video_list)
        self.count_chunk = -1
        
        #pdb.set_trace()
        #for i in range(self.num):
        #    self.__getitem__(i)
        print('# Pairs found:', self.num)

    def __len__(self):
        return self.num
        
  
    def __getitem__(self, index):   
       
        videofiles = sorted(self.audio_video_list[index])
        audiofile = self.audio_video_list[index][0].replace(self.video_folder, self.audio_folder)[:-12]+'.npy'#.replace('.npy', '.wav')
        labelfile = self.audio_video_list[index][0].replace(self.video_folder, 'labels')[:-12]+'.txt'#.replace('.npy', '.wav')
        ims = []
        for videofile in videofiles:
            print(videofile)
            imtv = load_npy_lip(videofile)
            #imtv = np.expand_dims(imtv, axis = 0)
            #imtv = np.transpose(imtv, (0, 2, 1, 3, 4))
            #imtv = torch.autograd.Variable(torch.from_numpy(imtv.astype(float)).float())
            ims.append(imtv)
        cct = np.load(audiofile)            
        #cct = np.expand_dims(cct, axis = 0)
        #cct = torch.autograd.Variable(torch.from_numpy(cct.astype(float)).float())
        label = get_ours_label_all(labelfile,ims[0].shape[0])
        #label = np.ones((ims[0].shape[0]))
        #print('synced label',ims[0].shape)

        return ims, cct, label, audiofile.split("/")[-1][:-4]

class AudioVideoDataset_ours_tri(data.Dataset):

    def __init__(self, opt, mode = 'train', audio = '39',video = 'lip', seed = np.random.seed(0)):
        super(AudioVideoDataset_ours_tri, self).__init__()


        self.chunk_index = 0
        self.mode = mode
        self.opt = opt
        if opt.audio == '39':
            self.audio_folder = 'audio_feat_39'
        elif opt.audio == '13':
            self.audio_folder = 'audio_feat'
        else:
            self.audio_folder = 'audio_feat_far'


        if opt.video == 'lip':
            self.video_folder = 'center_crop_112'
        else:
            self.video_folder = 'face_crop_224'

        #pdb.set_trace()
            
      
        folders_all = glob.glob(opt.data_dir + '/*')
        folder_list_train = ['data1_selected_full_face_v2', 'data2_selected_full_face', 'local_143_selected_full_face_v2', 'data1_selected_full_face']
        #print('shooooooooooooooooooooooort listtttttttttttttttttttttttttttt')
        folder_list_val = ['devset']
    # folder_list_val_ours = ['test_diar']

        #pdb.set_trace()
        if self.mode == 'train':
            folders = [i for i in folders_all if i.split('/')[-1] in folder_list_train]
        else:
            folders = [i for i in folders_all if i.split('/')[-1] in folder_list_val]

        video_list = []
        audio_list = []
        
        for folder in folders:
            dirs = glob.glob(folder + '/*')
            for dir in dirs:
                video_list.extend(glob.glob(dir + '/'+self.video_folder+'/*.npy'))
                audio_list.extend(glob.glob(dir + '/'+self.audio_folder+'/*.npy'))
                 

        
        self.audio_video_list = [i for i in video_list if i.replace(self.video_folder, self.audio_folder)in audio_list]
            

        #if mode == 'train':
        #    self.audio_video_list = [i for i in audio_video_list if i[i.rfind('/')+1:i.find('.npy')] in train_names]
        #else:
        #    self.audio_video_list = audio_video_list
        #    self.audio_video_list = [i for i in audio_video_list if i[i.rfind('/') + 1:i.find('.npy')] not in train_names]

        #self.num = len(self.audio_video_list_all)
        self.num = len(self.audio_video_list)
        self.count_chunk = -1
        
        #pdb.set_trace()
        #for i in range(self.num):
        #    self.__getitem__(i)
        print('# Pairs found:', self.num)

    def len_chunk(self):
        return self.opt.chunk_size

    def __len__(self):
        return self.num
        
    def load_chunk(self):
        self.audio_video_list = random.sample(self.audio_video_list_all, 100)

    def __getitem__(self, index):   
        '''    
        if self.mode == 'train':
            self.count_chunk += 1
            if self.count_chunk%1000 == 0:
                self.load_chunk()
                self.count_chunk = 0
            index = self.count_chunk
        else:
            self.audio_video_list = self.audio_video_list_all
        '''
      
        videofile = self.audio_video_list[index]
        audiofile = self.audio_video_list[index].replace(self.video_folder, self.audio_folder)#.replace('.npy', '.wav')
        #pdb.set_trace()
        #im = load_pickle(videofile)
        # print('im_shape', im.shape)
        if self.mode == 'eval':
            imtv = load_npy_lip(videofile)
            cct = np.load(audiofile)
            imtv = np.expand_dims(imtv, axis = 0)
            cct = np.expand_dims(cct, axis = 0)
            #imtv = np.transpose(imtv, (1, 0, 2, 3))
            print('shape', imtv.shape)

            imtv = np.transpose(imtv, (0, 2, 1, 3, 4))
            imtv = torch.autograd.Variable(torch.from_numpy(imtv.astype(float)).float())

            cct = torch.autograd.Variable(torch.from_numpy(cct.astype(float)).float())

            cct_neg = []

            lastframe = imtv.shape[2]-self.opt.frame_len-1
            for i in range (0, imtv.shape[2], self.opt.frame_len):
                #imtv = imtv_all[:, :, i:i+self.opt.frame_len, :, :] 
                cc_in, new_idx = get_random_audio_lip([index], len(self.audio_video_list)-1, self.audio_video_list, self.opt.frame_len)
                cct_neg.extend(cc_in)

            cct_neg = np.array(cct_neg)
            cct_neg = np.expand_dims(np.transpose(cct_neg, (1, 0, 2)).reshape((cct_neg.shape[1], -1)), axis = 0)
            cct_neg = np.expand_dims(cct_neg, axis = 0)
            cct_neg = torch.autograd.Variable(torch.from_numpy(cct_neg.astype(float)).float())
            



            if self.opt.norm_img:
                imtv = imtv/255. - 0.5
            return imtv, cct, cct_neg

        

        else:
            im = load_npy_lip(videofile)

            cc = np.load(audiofile)
            
            try:
                start_frame = random.randint(0, int(min(cc.shape[2]/4 - self.opt.frame_len, len(im) - self.opt.frame_len)))
            except:
                #print('cc.shape[2]', cc.shape[2])
                #print('len(im)', len(im))
                start_frame = 0
            # print('start from', start_frame)

            im_batch = im[start_frame:start_frame + self.opt.frame_len, :, :, :]
            # print(im_batch.shape)

            # pdb.set_trace()
            im_in = np.transpose(im_batch, (1, 0, 2, 3))

            #print('im_in', im_in.shape)
            '''
            if self.type == 'replace':
                random_audio = random.randint(0, self.num)
                while random_audio == index:
                    random_audio = random.randint(0, self.num)

                audiofile = self.audio_video_list[random_audio].replace('center_crop_112', 'audio_feat')#.replace('.npy', '.wav')
            '''
            
            #shift_len1 = 0
            #while -6 <= shift_len <= 6:
            if self.opt.settings == 0:
                shift_len1 = 0
                shift_len = random.randint(int(max(-start_frame, -self.opt.vshift)), int(min(self.opt.vshift, max(0, min((cc.shape[2]/4 - self.opt.frame_len - start_frame), (len(im) - start_frame - self.opt.frame_len))))))
                cc_in = cc[:, :, (start_frame + shift_len) * 4:(start_frame + shift_len + self.opt.frame_len) * 4]

                comp_type = random.randint(0, 3)

                if comp_type == 2:
                    random_audio = random.randint(0, len(self.audio_video_list)-1)
                    while abs(random_audio) == abs(index):
                        random_audio = random.randint(0, len(self.audio_video_list))

                    random_audio = self.audio_video_list[random_audio].replace(self.video_folder, self.audio_folder)#.replace('.npy', '.wav')
                    random_cc = np.load(random_audio)
                    try:
                        random_start_frame = random.randint(0, random_cc.shape[2] - self.opt.frame_len*4)
                    except:
                        random_start_frame = 0
                    cc_in1 = random_cc[:, :, random_start_frame:random_start_frame + self.opt.frame_len * 4]
                    shift_len1 = 10000
                    
                elif (comp_type == 0 or comp_type == 1 ) and not shift_len == 0:    
                    cc_in1 = cc[:, :, start_frame * 4:(start_frame + self.opt.frame_len) * 4]
                    shift_len1 = 0

                else:
                    while shift_len1 == shift_len:
                        #print('shift_len1',shift_len1) 
                        start = int(max(-start_frame, -self.opt.vshift))
                        end = int(min(self.opt.vshift, max(0, min((cc.shape[2]/4 - self.opt.frame_len - start_frame), (len(im) - start_frame - self.opt.frame_len)))))
                        if start < end and not shift_len == start:
                            shift_len1 = random.randint(start, end)
                        else:
                            print('shift_len1',shift_len1)
                            shift_len1 = 0 
                            break
                    cc_in1 = cc[:, :, (start_frame + shift_len1) * 4:(start_frame + shift_len1 + self.opt.frame_len) * 4]
            
            elif self.opt.settings == 1: 
                cc_in = cc[:, :, start_frame * 4:(start_frame + self.opt.frame_len) * 4]
                shift_len = 0
                random_audio = 0
                #if comp_type == 2:
                while abs(random_audio) == abs(index):
                    random_audio = random.randint(0, len(self.audio_video_list)-1)
                
                    #random_audio = random.randint(0, self.num)

                random_audio = self.audio_video_list[random_audio].replace(self.video_folder, self.audio_folder)#.replace('.npy', '.wav')
                random_cc = np.load(random_audio)
                random_start_frame = random.randint(0, random_cc.shape[2] - self.opt.frame_len*4)
                cc_in1 = random_cc[:, :, random_start_frame:random_start_frame + self.opt.frame_len * 4]
                shift_len1 = 10000
                
        


            #else:

            # cc_in = torch.cat(cc_batch, 0)

            if not cc_in.shape[2] == self.opt.frame_len*4:
                print('wrong shape', cc_in.shape[2])
                pdb.set_trace()
                _cc_in = np.zeros((1, 13, self.opt.frame_len*4))
                _cc_in[:, :, :cc_in.shape[2]] = cc_in
                cc_in = _cc_in
            if not cc_in1.shape[2] == self.opt.frame_len*4:
                print('wrong shape cc1', cc_in1.shape[2])
                #pdb.set_trace()
                _cc_in = np.zeros((1, 13, self.opt.frame_len*4))
                _cc_in[:, :, :cc_in1.shape[2]] = cc_in1
                cc_in1 = _cc_in
            imtv = torch.autograd.Variable(torch.from_numpy(im_in.astype(float)).float())
            cct = torch.autograd.Variable(torch.from_numpy(cc_in.astype(float)).float())
            cct1 = torch.autograd.Variable(torch.from_numpy(cc_in1.astype(float)).float())
            # label = torch.autograd.Variable(torch.tensor(1-(np.absolute(float(shift_len))/self.opt.frame_len)))
            label = torch.autograd.Variable(torch.tensor(float(shift_len)))
            label1 = torch.autograd.Variable(torch.tensor(float(shift_len1)))
            # print('ok')
            #if self.type == 'shift':
            if self.opt.norm_img:
                imtv = imtv/255. - 0.5
            if abs(label) < abs(label1):
                return imtv, cct, cct1, label, label1
            else:
                return imtv, cct1, cct, label1, label
            #else:  # self.mode == 'rep'
            #    return imtv, cct


class AudioVideoDataset_ours_tri_silence(data.Dataset):

    def __init__(self, opt, mode = 'train',seed = np.random.seed(0)):
        super(AudioVideoDataset_ours_tri_silence, self).__init__()


        self.chunk_index = 0
        self.mode = mode
        self.opt = opt

        if opt.audio == '39':
            self.audio_folder = 'audio_feat_39'
        elif opt.audio == '13':
            self.audio_folder = 'audio_feat'
        else:
            self.audio_folder = 'audio_feat_far'


        if opt.video == 'lip':
            self.video_folder = 'center_crop_112'
        else:
            self.video_folder = 'face_crop_224'
            
        if self.mode == 'train':
            folders_all = glob.glob(opt.train_dir + '/*')
        else:
            folders_all = glob.glob(opt.test_dir + '/*')
        
        #pdb.set_trace()

        folder_list_train = ['local_selected_full_face_slow2_v2_extendVAD','local_selected_full_face_slow2_v1_extendVAD','local_selected_full_face_slow1_v5_extendVAD']
        #print('shoooooooooooooooooooooooooooooooort listttttttttttttttttttt')
        folder_list_val = ['devset']

        #
        if self.mode == 'train':
            folders = [i for i in folders_all if i.split('/')[-1] in folder_list_train]
        else:
            folders = [i for i in folders_all if i.split('/')[-1] in folder_list_val]

        video_list = []
        audio_list = []
        label_list = []
        silence_list = []
        for folder in folders:
            dirs = glob.glob(folder + '/*')
            for dir in dirs:
                audio_list.extend(glob.glob(dir + '/'+self.audio_folder+'/*.npy'))
                if self.mode == 'train':
                    silence_list.extend(glob.glob(dir + '/sil_info_in_ori_video/*.txt'))
                    video_list.extend(glob.glob(dir + '/'+self.video_folder+'/*.npy'))
                else:
                    video_list.extend(glob.glob(dir + '/'+self.video_folder+'/*.npy'))

        if self.mode == 'train':
            self.audio_video_list = [i for i in video_list if ((i.replace(self.video_folder, self.audio_folder)in audio_list) and (i.replace(self.video_folder, 'sil_info_in_ori_video')[:-4] +'.txt' in silence_list))]
            #self.audio_video_list = [i for i in video_list if ]
        else:
            self.audio_video_list = [i for i in video_list if i.replace(self.video_folder, self.audio_folder)in audio_list]

        '''
        self.chunk_index = 0
        self.mode = mode
        self.type =  type
        self.opt = opt
            
        folders_all = glob.glob(opt.data_dir + '/*')
        folder_list_train = ['data1_selected_full_face_v2', 'data2_selected_full_face', 'local_143_selected_full_face_v2', 'data1_selected_full_face']
        folder_list_val = ['devset']
        folder_list_val_ours = ['test_diar']
        folder_list_silence = ['local_selected_full_face_slow2_v2_extendVAD','local_selected_full_face_slow2_v1_extendVAD','local_selected_full_face_slow1_v5_extendVAD']

        #pdb.set_trace()
        if self.mode == 'train':
            folders = [i for i in folders_all if i.split('/')[-1] in folder_list_train]
            folders_all_silence = glob.glob('/publicdata_slow2/Data/Lip/new_added_extendVAD_dataset' + '/*')
            folders_silence = [i for i in folders_all_silence if i.split('/')[-1] in folder_list_silence]

        elif self.mode == 'eval_ours':
            folders = [i for i in folders_all if i.split('/')[-1] in folder_list_val_ours]
        else:
            folders = [i for i in folders_all if i.split('/')[-1] in folder_list_val]

        video_list = []
        audio_list = []
        label_list = []
        for folder in folders:
            dirs = glob.glob(folder + '/*')
            for dir in dirs:
                video_list.extend(glob.glob(dir + '/center_crop_112/*.npy'))
                audio_list.extend(glob.glob(dir + '/audio_feat/*.npy'))
                if self.mode == 'eval_ours':
                    label_list.extend(glob.glob(dir + '/labels/*.txt'))
        silence_list = []        
        if self.mode == 'train':
            for folder in folders_silence:
                dirs = glob.glob(folder + '/*')
                for dir in dirs:
                    silence_list.extend(glob.glob(dir + '/sil_info_in_ori_video/*.txt'))
            self.silence_dict = {}
            for silence in silence_list:
                self.silence_dict[silence.split('/')[-1]] = silence

        if self.mode == 'eval_ours':
            #
            self.audio_video_list_ = [i for i in video_list if i.replace('center_crop_112', 'audio_feat')[:-12]+'.npy' in audio_list]
            #pdb.set_trace()
            self.audio_video_list_ = [i for i in video_list if i.replace('center_crop_112', 'labels')[:-12]+'.txt' in label_list]
            videos = set([i[:-12] for i in self.audio_video_list_])
            self.audio_video_list = []
            for video in videos:
                self.audio_video_list.append([i for i in self.audio_video_list_ if i[:-12]==video])
       
        else:
            self.audio_video_list = [i for i in video_list if i.replace('center_crop_112', 'audio_feat')in audio_list]
            if self.mode == 'train':
                self.audio_video_list = [i for i in video_list if i.split('/')[-1] in self.silence_dict.keys()]
        '''
        #if mode == 'train':
        #    self.audio_video_list = [i for i in audio_video_list if i[i.rfind('/')+1:i.find('.npy')] in train_names]
        #else:
        #    self.audio_video_list = audio_video_list
        #    self.audio_video_list = [i for i in audio_video_list if i[i.rfind('/') + 1:i.find('.npy')] not in train_names]

        #self.num = len(self.audio_video_list_all)
        self.num = len(self.audio_video_list)
        self.count_chunk = -1
        
        #pdb.set_trace()
        #for i in range(self.num):
        #    self.__getitem__(i)
        print('# Pairs found:', self.num)

    def len_chunk(self):
        return self.opt.chunk_size

    def __len__(self):
        return self.num
        
    def load_chunk(self):
        self.audio_video_list = random.sample(self.audio_video_list_all, 100)

    def __getitem__(self, index):   
        '''    
        if self.mode == 'train':
            self.count_chunk += 1
            if self.count_chunk%1000 == 0:
                self.load_chunk()
                self.count_chunk = 0
            index = self.count_chunk
        else:
            self.audio_video_list = self.audio_video_list_all
        '''
      
        if self.mode == 'eval':
            videofile = self.audio_video_list[index]
            audiofile = self.audio_video_list[index].replace(self.video_folder, self.audio_folder)#.replace('.npy', '.wav')
            imtv = load_npy_lip(videofile)
            cct = np.load(audiofile)
            imtv = np.expand_dims(imtv, axis = 0)
            cct = np.expand_dims(cct, axis = 0)
            #imtv = np.transpose(imtv, (1, 0, 2, 3))
            print('shape', imtv.shape)

            imtv = np.transpose(imtv, (0, 2, 1, 3, 4))
            imtv = torch.autograd.Variable(torch.from_numpy(imtv.astype(float)).float())

            cct = torch.autograd.Variable(torch.from_numpy(cct.astype(float)).float())

            cct_neg = []

            lastframe = imtv.shape[2]-self.opt.frame_len-1
            for i in range (0, imtv.shape[2], self.opt.frame_len):
                #imtv = imtv_all[:, :, i:i+self.opt.frame_len, :, :] 
                cc_in, new_idx = get_random_audio_lip([index], len(self.audio_video_list)-1, self.audio_video_list, self.opt.frame_len)
                cct_neg.extend(cc_in)

            cct_neg = np.array(cct_neg)
            cct_neg = np.expand_dims(np.transpose(cct_neg, (1, 0, 2)).reshape((cct_neg.shape[1], -1)), axis = 0)
            cct_neg = np.expand_dims(cct_neg, axis = 0)
            cct_neg = torch.autograd.Variable(torch.from_numpy(cct_neg.astype(float)).float())
            



            if self.opt.norm_img:
                imtv = imtv/255. - 0.5
            return imtv, cct, cct_neg

        

        else:
            videofile = self.audio_video_list[index]
            audiofile = self.audio_video_list[index].replace(self.video_folder, self.audio_folder)#.replace('.npy', '.wav')
            im = load_npy_lip(videofile)

            cc = np.load(audiofile)

            #silencefile = self.silence_dict[videofile.split('/')[-1]]
            silencefile = self.audio_video_list[index].replace(self.video_folder, 'sil_info_in_ori_video')[:-4]+'.txt'#.replace('.npy', '.wav')
            silence = read_silence(silencefile)
            
            try:
                start_frame = random.randint(0 + silence[0], int(min(cc.shape[2]/4 - self.opt.frame_len, len(im) - self.opt.frame_len-silence[-1])))
            except:
                #print('cc.shape[2]', cc.shape[2])
                #print('len(im)', len(im))
                start_frame = 0
            # print('start from', start_frame)

            im_batch = im[start_frame:start_frame + self.opt.frame_len, :, :, :]
            # print(im_batch.shape)

            # pdb.set_trace()
            im_in = np.transpose(im_batch, (1, 0, 2, 3))
                
            



            #print('im_in', im_in.shape)
            '''
            if self.type == 'replace':
                random_audio = random.randint(0, self.num)
                while random_audio == index:
                    random_audio = random.randint(0, self.num)

                audiofile = self.audio_video_list[random_audio].replace('center_crop_112', 'audio_feat')#.replace('.npy', '.wav')
            '''
            
            #shift_len1 = 0
            #while -6 <= shift_len <= 6:
            if self.opt.type == 'shift_replaced':
                shift_len1 = 0
                shift_len = random.randint(int(max(-start_frame, -self.opt.vshift)), int(min(self.opt.vshift, max(0, min((cc.shape[2]/4 - self.opt.frame_len - start_frame), (len(im) - start_frame - self.opt.frame_len))))))
                cc_in = cc[:, :, (start_frame + shift_len) * 4:(start_frame + shift_len + self.opt.frame_len) * 4]
                if check_silence(silence,start_frame + shift_len ,self.opt.frame_len) and check_silence(silence,start_frame,self.opt.frame_len):
                    #cc_in = cc[:, :, (start_frame ) * 4:(start_frame  + self.opt.frame_len) * 4]
                    shift_len = 0    
                comp_type = random.randint(0, 3)

                if comp_type == 2:
                    random_audio = random.randint(0, len(self.audio_video_list)-1)
                    while abs(random_audio) == abs(index):
                        print('new random audio')

                        random_audio = random.randint(0, len(self.audio_video_list))

                    random_audio = self.audio_video_list[random_audio].replace(self.video_folder,self.audio_folder)#.replace('.npy', '.wav')
                    random_cc = np.load(random_audio)
                    try:
                        random_start_frame = random.randint(0, random_cc.shape[2] - self.opt.frame_len*4)
                    except:
                        random_start_frame = 0
                    cc_in1 = random_cc[:, :, random_start_frame:random_start_frame + self.opt.frame_len * 4]
                    shift_len1 = 10000
                    
                elif (comp_type == 0 or comp_type == 1 ) and not shift_len == 0:    
                    cc_in1 = cc[:, :, start_frame * 4:(start_frame + self.opt.frame_len) * 4]
                    shift_len1 = 0

                else:
                    while shift_len1 == shift_len:# or ((check_silence(silence,start_frame + shift_len1 ,self.opt.frame_len) and check_silence(silence,start_frame,self.opt.frame_len)) and shift_len == 0):
                        #print()
                        #print('new shift_len1',shift_len1)
                        #print('silence',silence)
                        #print('start_frame',start_frame)
                        start = int(max(silence[1],(max(-start_frame, -self.opt.vshift))))
                        end = int(min(silence[3] - self.opt.frame_len,min(self.opt.vshift, max(0, min((cc.shape[2]/4 - self.opt.frame_len - start_frame), (len(im) - start_frame - self.opt.frame_len))))))
                        if start < end and not shift_len == start:
                            shift_len1 = random.randint(start, end)
                        else:
                            print('shift_len1',shift_len1)
                            shift_len1 = 0 
                            break
                    cc_in1 = cc[:, :, (start_frame + shift_len1) * 4:(start_frame + shift_len1 + self.opt.frame_len) * 4]
                    if check_silence(silence,start_frame + shift_len1 ,self.opt.frame_len) and check_silence(silence,start_frame,self.opt.frame_len):
                    #cc_in = cc[:, :, (start_frame ) * 4:(start_frame  + self.opt.frame_len) * 4]
                        shift_len1 = 0 
            elif self.opt.settings == 'replaced': 
                cc_in = cc[:, :, start_frame * 4:(start_frame + self.opt.frame_len) * 4]
                shift_len = 0
                random_audio = 0
                #if comp_type == 2:
                while abs(random_audio) == abs(index):
                    random_audio = random.randint(0, len(self.audio_video_list)-1)
                
                    #random_audio = random.randint(0, self.num)

                random_audio = self.audio_video_list[random_audio].replace(self.video_folder, self.audio_folder)#.replace('.npy', '.wav')
                random_cc = np.load(random_audio)
                random_start_frame = random.randint(0, random_cc.shape[2] - self.opt.frame_len*4)
                cc_in1 = random_cc[:, :, random_start_frame:random_start_frame + self.opt.frame_len * 4]
                shift_len1 = 10000
                
        
            

            #else:

            # cc_in = torch.cat(cc_batch, 0)

            if not cc_in.shape[2] == self.opt.frame_len*4:
                print('wrong shape', cc_in.shape[2])
                pdb.set_trace()
                _cc_in = np.zeros((1, 13, self.opt.frame_len*4))
                _cc_in[:, :, :cc_in.shape[2]] = cc_in
                cc_in = _cc_in
            if not cc_in1.shape[2] == self.opt.frame_len*4:
                print('wrong shape cc1', cc_in1.shape[2])
                #pdb.set_trace()
                _cc_in = np.zeros((1, 13, self.opt.frame_len*4))
                _cc_in[:, :, :cc_in1.shape[2]] = cc_in1
                cc_in1 = _cc_in
            imtv = torch.autograd.Variable(torch.from_numpy(im_in.astype(float)).float())
            cct = torch.autograd.Variable(torch.from_numpy(cc_in.astype(float)).float())
            cct1 = torch.autograd.Variable(torch.from_numpy(cc_in1.astype(float)).float())
            # label = torch.autograd.Variable(torch.tensor(1-(np.absolute(float(shift_len))/self.opt.frame_len)))
            label = torch.autograd.Variable(torch.tensor(float(shift_len)))
            label1 = torch.autograd.Variable(torch.tensor(float(shift_len1)))
            # print('ok')
            #if self.type == 'shift':
            if self.opt.norm_img:
                imtv = imtv/255. - 0.5
            if abs(label) < abs(label1):
                return imtv, cct, cct1, label, label1
            else:
                return imtv, cct1, cct, label1, label


class AudioVideoDataset_ours_tri_lip_chunk(data.Dataset):

    def __init__(self, opt, mode = 'train', type = 'shift', seed = np.random.seed(0)):
        super(AudioVideoDataset_ours_tri_lip_chunk, self).__init__()


        self.chunk_index = 0
        self.mode = mode
        self.type =  type
        self.opt = opt
            
        folders_all = glob.glob(opt.data_dir + '/*')
        folder_list_train = ['data1_selected_full_face_v2', 'data2_selected_full_face', 'local_143_selected_full_face_v2', 'data1_selected_full_face']
        folder_list_val = ['devset']
        folder_list_val_ours = ['test_diar']

        #pdb.set_trace()
        if self.mode == 'train':
            folders = [i for i in folders_all if i.split('/')[-1] in folder_list_train]
        elif self.mode == 'eval_ours':
            folders = [i for i in folders_all if i.split('/')[-1] in folder_list_val_ours]
        else:
            folders = [i for i in folders_all if i.split('/')[-1] in folder_list_val]

        video_list = []
        audio_list = []
        label_list = []
        for folder in folders:
            dirs = glob.glob(folder + '/*')
            for dir in dirs:
                video_list.extend(glob.glob(dir + '/center_crop_112/*.npy'))
                audio_list.extend(glob.glob(dir + '/audio_feat/*.npy'))
                if self.mode == 'eval_ours':
                    label_list.extend(glob.glob(dir + '/labels/*.txt'))

        if self.mode == 'eval_ours':
            #
            self.audio_video_list_ = [i for i in video_list if i.replace('center_crop_112', 'audio_feat')[:-12]+'.npy' in audio_list]
            #pdb.set_trace()
            self.audio_video_list_ = [i for i in video_list if i.replace('center_crop_112', 'labels')[:-12]+'.txt' in label_list]
            videos = set([i[:-12] for i in self.audio_video_list_])
            self.audio_video_list = []
            for video in videos:
                self.audio_video_list.append([i for i in self.audio_video_list_ if i[:-12]==video])
       
        else:
            self.audio_video_list = [i for i in video_list if i.replace('center_crop_112', 'audio_feat')in audio_list]
        

        #if mode == 'train':
        #    self.audio_video_list = [i for i in audio_video_list if i[i.rfind('/')+1:i.find('.npy')] in train_names]
        #else:
        #    self.audio_video_list = audio_video_list
        #    self.audio_video_list = [i for i in audio_video_list if i[i.rfind('/') + 1:i.find('.npy')] not in train_names]

        #self.num = len(self.audio_video_list_all)
        self.num = len(self.audio_video_list)
        if self.mode == 'train':          
            self.num_chunk = self.opt.chunk_size
            self.get_chunk()
        else:
            self.num_chunk = self.num
        
        #pdb.set_trace()
        #for i in range(self.num):
        #    self.__getitem__(i)
        print('# Pairs found:', self.num)

    def len_chunk(self):
        return self.num_chunk

    def __len__(self):
        return self.num_chunk
        
    def load_chunk(self):
        self.audio_video_list = random.sample(self.audio_video_list_all, 100)
    def get_chunk(self):
        idxs = random.sample(np.arange(self.num),self.num_chunk)
        self.videos = []
        self.audios = []
        for index in idxs:
            videofile = self.audio_video_list[index]
            audiofile = self.audio_video_list[index].replace('center_crop_112', 'audio_feat')#.replace('.npy', '.wav')
            self.videos.append(load_npy_lip(videofile))
            self.audios.append(np.load(audiofile))

    def __getitem__(self, index):   
        '''    
        if self.mode == 'train':
            self.count_chunk += 1
            if self.count_chunk%1000 == 0:
                self.load_chunk()
                self.count_chunk = 0
            index = self.count_chunk
        else:
            self.audio_video_list = self.audio_video_list_all
        '''
        if self.mode == 'eval_ours':
            videofiles = sorted(self.audio_video_list[index])
            audiofile = self.audio_video_list[index][0].replace('center_crop_112', 'audio_feat')[:-12]+'.npy'#.replace('.npy', '.wav')
            labelfile = self.audio_video_list[index][0].replace('center_crop_112', 'labels')[:-12]+'.txt'#.replace('.npy', '.wav')
            ims = []
            for videofile in videofiles:
                print(videofile)
                imtv = load_npy_lip(videofile)
                #imtv = np.expand_dims(imtv, axis = 0)
                #imtv = np.transpose(imtv, (0, 2, 1, 3, 4))
                #imtv = torch.autograd.Variable(torch.from_numpy(imtv.astype(float)).float())
                ims.append(imtv)
            cct = np.load(audiofile)            
            #cct = np.expand_dims(cct, axis = 0)
            #cct = torch.autograd.Variable(torch.from_numpy(cct.astype(float)).float())
            label = get_ours_label_all(labelfile,ims[0].shape[0])
            #label = np.ones((ims[0].shape[0]))
            #print('synced label',ims[0].shape)

            return ims, cct, label, audiofile.split("/")[-1][:-4]
        #print(index)
        videofile = self.audio_video_list[index]
        audiofile = self.audio_video_list[index].replace('center_crop_112', 'audio_feat')#.replace('.npy', '.wav')
        #pdb.set_trace()
        #im = load_pickle(videofile)
        # print('im_shape', im.shape)
        if self.mode == 'eval':
            imtv = load_npy_lip(videofile)
            cct = np.load(audiofile)
            imtv = np.expand_dims(imtv, axis = 0)
            cct = np.expand_dims(cct, axis = 0)
            #imtv = np.transpose(imtv, (1, 0, 2, 3))
            print('shape', imtv.shape)

            imtv = np.transpose(imtv, (0, 2, 1, 3, 4))
            imtv = torch.autograd.Variable(torch.from_numpy(imtv.astype(float)).float())

            cct = torch.autograd.Variable(torch.from_numpy(cct.astype(float)).float())

            cct_neg = []

            lastframe = imtv.shape[2]-self.opt.frame_len-1
            for i in range (0, imtv.shape[2], self.opt.frame_len):
                #imtv = imtv_all[:, :, i:i+self.opt.frame_len, :, :] 
                cc_in, new_idx = get_random_audio_lip([index], len(self.audio_video_list)-1, self.audio_video_list, self.opt.frame_len)
                cct_neg.extend(cc_in)

            cct_neg = np.array(cct_neg)
            cct_neg = np.expand_dims(np.transpose(cct_neg, (1, 0, 2)).reshape((cct_neg.shape[1], -1)), axis = 0)
            cct_neg = np.expand_dims(cct_neg, axis = 0)
            cct_neg = torch.autograd.Variable(torch.from_numpy(cct_neg.astype(float)).float())
            



            if self.opt.norm_img:
                imtv = imtv/255. - 0.5
            return imtv, cct, cct_neg

        

        else:


            im = self.videos[index]
            cc = self.audios[index]
            # print('im_shape', im.shape)
            try:
                start_frame = random.randint(0, min(cc.shape[2]/4 - self.opt.frame_len, len(im) - self.opt.frame_len))
            except:
                start_frame = 0
            # print('start from', start_frame)

            im_batch = im[start_frame:start_frame + self.opt.frame_len, :, :, :]
            # print(im_batch.shape)

            # pdb.set_trace()
            im_in = np.transpose(im_batch, (1, 0, 2, 3))

            #print('im_in', im_in.shape)
            '''
            if self.type == 'replace':
                random_audio = random.randint(0, self.num)
                while random_audio == index:
                    random_audio = random.randint(0, self.num)

                audiofile = self.audio_video_list[random_audio].replace('center_crop_112', 'audio_feat')#.replace('.npy', '.wav')
            '''
            
            #shift_len1 = 0
            #while -6 <= shift_len <= 6:
            if self.opt.settings == 0:
                shift_len1 = 0
                shift_len = random.randint(int(max(-start_frame, -self.opt.vshift)), int(min(self.opt.vshift, max(0, min((cc.shape[2]/4 - self.opt.frame_len - start_frame), (len(im) - start_frame - self.opt.frame_len))))))
                cc_in = cc[:, :, (start_frame + shift_len) * 4:(start_frame + shift_len + self.opt.frame_len) * 4]

                comp_type = random.randint(0, 3)

                if comp_type == 2:
                    random_audio = random.randint(0, len(self.audio_video_list)-1)
                    while abs(random_audio) == abs(index):
                        random_audio = random.randint(0, len(self.audio_video_list))

                    random_audio = self.audio_video_list[random_audio].replace('center_crop_112', 'audio_feat')#.replace('.npy', '.wav')
                    random_cc = np.load(random_audio)
                    try:
                        random_start_frame = random.randint(0, random_cc.shape[2] - self.opt.frame_len*4)
                    except:
                        random_start_frame = 0
                    cc_in1 = random_cc[:, :, random_start_frame:random_start_frame + self.opt.frame_len * 4]
                    shift_len1 = 10000
                    
                elif (comp_type == 0 or comp_type == 1 ) and not shift_len == 0:    
                    cc_in1 = cc[:, :, start_frame * 4:(start_frame + self.opt.frame_len) * 4]
                    shift_len1 = 0

                else:
                    while shift_len1 == shift_len:
                        shift_len1 = random.randint(int(max(-start_frame, -self.opt.vshift)), int(min(self.opt.vshift, max(0, min((cc.shape[2]/4 - self.opt.frame_len - start_frame), (len(im) - start_frame - self.opt.frame_len))))))
                    cc_in1 = cc[:, :, (start_frame + shift_len1) * 4:(start_frame + shift_len1 + self.opt.frame_len) * 4]
            
            elif self.opt.settings == 1: 
                cc_in = cc[:, :, start_frame * 4:(start_frame + self.opt.frame_len) * 4]
                shift_len = 0
                random_audio = 0
                #if comp_type == 2:
                while abs(random_audio) == abs(index):
                    random_audio = random.randint(0, len(self.audio_video_list)-1)
                
                    #random_audio = random.randint(0, self.num)

                random_audio = self.audio_video_list[random_audio].replace('center_crop_112', 'audio_feat')#.replace('.npy', '.wav')
                random_cc = np.load(random_audio)
                random_start_frame = random.randint(0, random_cc.shape[2] - self.opt.frame_len*4)
                cc_in1 = random_cc[:, :, random_start_frame:random_start_frame + self.opt.frame_len * 4]
                shift_len1 = 10000
                
        


            #else:

            # cc_in = torch.cat(cc_batch, 0)

            if not cc_in.shape[2] == self.opt.frame_len*4:
                print('wrong shape', cc_in.shape[2])
                pdb.set_trace()
                _cc_in = np.zeros((1, 13, self.opt.frame_len*4))
                _cc_in[:, :, :cc_in.shape[2]] = cc_in
                cc_in = _cc_in
            if not cc_in1.shape[2] == self.opt.frame_len*4:
                print('wrong shape cc1', cc_in1.shape[2])
                #pdb.set_trace()
                _cc_in = np.zeros((1, 13, self.opt.frame_len*4))
                _cc_in[:, :, :cc_in1.shape[2]] = cc_in1
                cc_in1 = _cc_in
            imtv = torch.autograd.Variable(torch.from_numpy(im_in.astype(float)).float())
            cct = torch.autograd.Variable(torch.from_numpy(cc_in.astype(float)).float())
            cct1 = torch.autograd.Variable(torch.from_numpy(cc_in1.astype(float)).float())
            # label = torch.autograd.Variable(torch.tensor(1-(np.absolute(float(shift_len))/self.opt.frame_len)))
            label = torch.autograd.Variable(torch.tensor(float(shift_len)))
            label1 = torch.autograd.Variable(torch.tensor(float(shift_len1)))
            # print('ok')
            #if self.type == 'shift':
            if self.opt.norm_img:
                imtv = imtv/255. - 0.5
            if abs(label) < abs(label1):
                return imtv, cct, cct1, label, label1
            else:
                return imtv, cct1, cct, label1, label


class AudioVideoDataset_ours_tri_duoinput(data.Dataset):

    def __init__(self, opt, mode = 'train', type = 'shift', seed = np.random.seed(0)):
        super(AudioVideoDataset_ours_tri_duoinput, self).__init__()

        self.chunk_index = 0
        

        with open(opt.train_file, 'rb') as fil:
            train_list = fil.readlines()

        train_names = [i[i.find('audio/')+6:i.find('.wav')] for i in train_list]
        #pdb.set_trace()

        #self.batch_size = opt.batch_size

        #import ipdb
        #ipdb.set_trace()
        self.mode = mode
        self.type =  type
        self.opt = opt
        if self.mode == 'train':
            dirs = glob.glob(self.opt.data_dir + '/*')
        else: 
            #pdb.set_trace()
            dirs = glob.glob(self.opt.eval_data_dir + '/*')
        video_list = []
        audio_list = []
        for dir in dirs:
            video_list.extend(glob.glob(dir + '/face_crop/*.npy'))
            audio_list.extend(glob.glob(dir + '/audio_feat/*.npy'))
        #print('npys', len(video_list))
        '''
        for video in video_list:
            obj = load_pickle1(video)
            if not obj.shape[0] == 224:
                #pdb.set_trace()
                video_list.remove(video)
                os.remove(video)
        '''
        #video_list = os.listdir((os.path.join(self.opt.data_dir, self.opt.work_dir, self.opt.video_dir)))
        #audio_list = os.listdir((os.path.join(self.opt.data_dir, self.opt.work_dir, self.opt.audio_dir)))


        audio_video_list = [i for i in video_list if i.replace('face_crop', 'audio_feat')in audio_list]

        if mode == 'train':
            self.audio_video_list = [i for i in audio_video_list if i[i.rfind('/')+1:i.find('.npy')] in train_names]
        else:
            self.audio_video_list = audio_video_list
        #    self.audio_video_list = [i for i in audio_video_list if i[i.rfind('/') + 1:i.find('.npy')] not in train_names]

        self.num = len(self.audio_video_list)

        print('# Pairs found:', self.num)

    def len_chunk(self):
        return self.opt.chunk_size

    def __len__(self):
        return self.num

    

    def __getitem__(self, index):
        videofile = self.audio_video_list[index]
        audiofile = self.audio_video_list[index].replace('face_crop', 'audio_feat')#.replace('.npy', '.wav')

        #im = load_pickle(videofile)
        # print('im_shape', im.shape)
        if self.mode == 'eval':
            imtv = load_npy(videofile)
            cct = np.load(audiofile)
            #imtv = np.expand_dims(imtv, axis = 0)
            #cct = np.expand_dims(cct, axis = 0)

            imtv = np.transpose(imtv, (1, 0, 2, 3))
            imtv = torch.autograd.Variable(torch.from_numpy(imtv.astype(float)).float())

            cct = torch.autograd.Variable(torch.from_numpy(cct.astype(float)).float())

            im_in1, new_idx = get_random_video([index], self.num-1, self.audio_video_list, self.opt.frame_len)
            im_in2, _ = get_random_video([index], self.num-1, self.audio_video_list, self.opt.frame_len)

            imtv1 = torch.autograd.Variable(torch.from_numpy(im_in1.astype(float)).float())
            imtv2 = torch.autograd.Variable(torch.from_numpy(im_in2.astype(float)).float())
            return imtv, cct, imtv1, imtv2
        else:
            im = load_npy(videofile)


            start_frame = random.randint(0, len(im) - self.opt.frame_len)
            # print('start from', start_frame)

            im_batch = im[start_frame:start_frame + self.opt.frame_len, :, :, :]
            # print(im_batch.shape)

            # pdb.set_trace()
            im_in = np.transpose(im_batch, (1, 0, 2, 3))
            cc = np.load(audiofile)
            

      
            #shift_len1 = 0
            #while -6 <= shift_len <= 6:
            label = random.randint(0, 1)
            ex_list = [index]
            if label == 1: 
                cc_in = cc[:, :, start_frame * 4:(start_frame + self.opt.frame_len) * 4]
                #shift_len = 0
            else:
                cc_in, new_idx = get_random_audio(ex_list, self.num-1, self.audio_video_list, self.opt.frame_len)
                ex_list.append(new_idx)
            '''
            im_in1, new_idx = get_random_video(ex_list, self.num-1, self.audio_video_list, self.opt.frame_len)
            ex_list.append(new_idx)
            im_in2, _ = get_random_video(ex_list, self.num-1, self.audio_video_list, self.opt.frame_len)
            '''
            im_in1, new_idx = get_random_video([], self.num-1, self.audio_video_list, self.opt.frame_len)
            im_in2, _ = get_random_video([], self.num-1, self.audio_video_list, self.opt.frame_len)

            #else:

            # cc_in = torch.cat(cc_batch, 0)

            if not cc_in.shape[2] == 20:
                print('wrong shape')
                #pdb.set_trace()
                _cc_in = np.zeros((1, 13, 20))
                _cc_in[:, :, :cc_in.shape[2]] = cc_in
                cc_in = _cc_in
         
            imtv = torch.autograd.Variable(torch.from_numpy(im_in.astype(float)).float())
            imtv1 = torch.autograd.Variable(torch.from_numpy(im_in1.astype(float)).float())
            imtv2 = torch.autograd.Variable(torch.from_numpy(im_in2.astype(float)).float())
            cct = torch.autograd.Variable(torch.from_numpy(cc_in.astype(float)).float())
            #cct1 = torch.autograd.Variable(torch.from_numpy(cc_in1.astype(float)).float())
            # label = torch.autograd.Variable(torch.tensor(1-(np.absolute(float(shift_len))/self.opt.frame_len)))
            label = torch.autograd.Variable(torch.tensor(label)).long()
            #label1 = torch.autograd.Variable(torch.tensor(float(shift_len1)))
            # print('ok')
           
            return imtv, cct, imtv1, imtv2, label
               
class AudioVideoDataset_ours_tri_duoinput_lip(data.Dataset):

    def __init__(self, opt, mode = 'train', type = 'shift', seed = np.random.seed(0)):
        super(AudioVideoDataset_ours_tri_duoinput_lip, self).__init__()

        self.chunk_index = 0
        self.mode = mode
        self.type =  type
        self.opt = opt
            
        folders_all = glob.glob(opt.data_dir + '/*')
        folder_list_train = ['data1_selected_full_face_v2', 'data2_selected_full_face', 'local_143_selected_full_face_v2', 'data1_selected_full_face']
        folder_list_val = ['devset']


        if self.mode == 'train':
            folders = [i for i in folders_all if i.split('/')[-1] in folder_list_train]
        else:
            folders = [i for i in folders_all if i.split('/')[-1] in folder_list_val]

        video_list = []
        audio_list = []
        for folder in folders:
            dirs = glob.glob(folder + '/*')
            for dir in dirs:
                video_list.extend(glob.glob(dir + '/center_crop_112/*.npy'))
                audio_list.extend(glob.glob(dir + '/audio_feat/*.npy'))
       
        self.audio_video_list = [i for i in video_list if i.replace('center_crop_112', 'audio_feat')in audio_list]

        #if mode == 'train':
        #    self.audio_video_list = [i for i in audio_video_list if i[i.rfind('/')+1:i.find('.npy')] in train_names]
        #else:
        #    self.audio_video_list = audio_video_list
        #    self.audio_video_list = [i for i in audio_video_list if i[i.rfind('/') + 1:i.find('.npy')] not in train_names]

        self.num = len(self.audio_video_list)

        print('# Pairs found:', self.num)
        #pdb.set_trace()
    def len_chunk(self):
        return self.opt.chunk_size

    def __len__(self):
        return self.num

    

    def __getitem__(self, index):
        videofile = self.audio_video_list[index]
        audiofile = self.audio_video_list[index].replace('center_crop_112', 'audio_feat')#.replace('.npy', '.wav')

        #im = load_pickle(videofile)
        #print('im_shape', im.shape)
        if self.mode == 'eval':
            imtv = load_npy_lip(videofile)
            #print('imtv', imtv.shape)
            cct = np.load(audiofile)
            #imtv = np.expand_dims(imtv, axis = 0)
            #cct = np.expand_dims(cct, axis = 0)

            imtv = np.transpose(imtv, (1, 0, 2, 3))
            #imtv = torch.autograd.Variable(torch.from_numpy(imtv.astype(float)).float())

            #cct = torch.autograd.Variable(torch.from_numpy(cct.astype(float)).float())

            imtv1, new_idx = get_random_video_lip([index], self.num-1, self.audio_video_list, self.opt.frame_len)
            imtv2, _ = get_random_video_lip([index], self.num-1, self.audio_video_list, self.opt.frame_len)
            imtv1 = np.expand_dims(imtv1, axis = 0)
            imtv1 = np.repeat(imtv1, 20, axis = 0)

            imtv2 = np.expand_dims(imtv2, axis = 0)
            imtv2 = np.repeat(imtv2, 20, axis = 0)

            imtv1 = torch.autograd.Variable(torch.from_numpy(imtv1.astype(float)).float())
            imtv2 = torch.autograd.Variable(torch.from_numpy(imtv2.astype(float)).float())
            return imtv, cct, imtv1, imtv2

        if self.mode == 'eval_short':
            imtv_all = load_npy_lip(videofile)
            #print('imtv', imtv.shape)
            cct_all = np.load(audiofile)
            #imtv = np.expand_dims(imtv, axis = 0)
            #cct = np.expand_dims(cct, axis = 0)

            imtv_all = np.transpose(imtv_all, (1, 0, 2, 3))
            #imtv = torch.autograd.Variable(torch.from_numpy(imtv.astype(float)).float())

            #cct = torch.autograd.Variable(torch.from_numpy(cct.astype(float)).float())
            imtv_neg1 = []
            #labels = []
            imtv_neg2 = []
            cct_neg = []

            lastframe = imtv_all.shape[1]-self.opt.frame_len-1
            for i in range (0, imtv_all.shape[1], self.opt.frame_len):
                #imtv = imtv_all[:, :, i:i+self.opt.frame_len, :, :] 
                cc_in, new_idx = get_random_audio_lip([index], self.num-1, self.audio_video_list, self.opt.frame_len)
                cct_neg.extend(cc_in)


                imtv1, new_idx = get_random_video_lip([index], self.num-1, self.audio_video_list, self.opt.frame_len)
                imtv2, _ = get_random_video_lip([index], self.num-1, self.audio_video_list, self.opt.frame_len)
                imtv_neg1.extend(imtv1)
                imtv_neg2.extend(imtv2)
                #labels.extend([0, 1])
            #pdb.set_trace()
            cct_neg = np.array(cct_neg)
            cct_neg = np.expand_dims(np.transpose(cct_neg, (1, 0, 2)).reshape((cct_neg.shape[1], -1)), axis = 0)
            imtv_neg1 = np.array(imtv_neg1)
            imtv_neg2 = np.array(imtv_neg2)
                        
            imtv_neg1 = np.expand_dims(imtv_neg1.reshape((-1, imtv_neg1.shape[2], imtv_neg1.shape[3])), axis = 0)
            imtv_neg2 = np.expand_dims(imtv_neg2.reshape((-1, imtv_neg2.shape[2], imtv_neg2.shape[3])), axis = 0)


            '''
            imtv1 = np.expand_dims(imtv1, axis = 0)
            imtv1 = np.repeat(imtv1, 20, axis = 0)

            imtv2 = np.expand_dims(imtv2, axis = 0)
            imtv2 = np.repeat(imtv2, 20, axis = 0)
            '''
            #imtv_all = torch.autograd.Variable(torch.from_numpy(imtv_all.astype(float)).float())
            #imtv_neg1 = torch.autograd.Variable(torch.from_numpy(imtv_neg1.astype(float)).float())
            #imtv_neg2 = torch.autograd.Variable(torch.from_numpy(imtv_neg2.astype(float)).float())
            if self.opt.norm_img:
                return imtv_all/255. -0.5 , cct_all, cct_neg, imtv_neg1/255. -0.5, imtv_neg2/255. -0.5
            return imtv_all, cct_all, cct_neg, imtv_neg1, imtv_neg2
        else:
            im = load_npy_lip(videofile)
            cc = np.load(audiofile)


            #try:
            start_frame = random.randint(0, min(cc.shape[2]/4 - self.opt.frame_len, len(im) - self.opt.frame_len))
            #except:
            #    pdb.set_trace()
            # print('start from', start_frame)

            im_batch = im[start_frame:start_frame + self.opt.frame_len, :, :, :]
            # print(im_batch.shape)

            # pdb.set_trace()
            im_in = np.transpose(im_batch, (1, 0, 2, 3))
            

      
            #shift_len1 = 0
            #while -6 <= shift_len <= 6:
            label = random.randint(0, 1)
            ex_list = [index]
            if label == 1: 
                cc_in = cc[:, :, start_frame * 4:(start_frame + self.opt.frame_len) * 4]
                #shift_len = 0
            else:
                cc_in, new_idx = get_random_audio_lip([index], self.num-1, self.audio_video_list, self.opt.frame_len)
                ex_list.append(new_idx)
            '''
            im_in1, new_idx = get_random_video(ex_list, self.num-1, self.audio_video_list, self.opt.frame_len)
            ex_list.append(new_idx)
            im_in2, _ = get_random_video(ex_list, self.num-1, self.audio_video_list, self.opt.frame_len)
            '''
            im_in1, new_idx = get_random_video_lip([index], self.num-1, self.audio_video_list, self.opt.frame_len)
            im_in2, _ = get_random_video_lip([index], self.num-1, self.audio_video_list, self.opt.frame_len)

            #else:

            # cc_in = torch.cat(cc_batch, 0)

            if not cc_in.shape[2] == self.opt.frame_len*4:
                print('wrong shape')
                #pdb.set_trace()
                _cc_in = np.zeros((1, 13, self.opt.frame_len*4))
                _cc_in[:, :, :cc_in.shape[2]] = cc_in
                cc_in = _cc_in
         
            imtv = torch.autograd.Variable(torch.from_numpy(im_in.astype(float)).float())
            imtv1 = torch.autograd.Variable(torch.from_numpy(im_in1.astype(float)).float())
            imtv2 = torch.autograd.Variable(torch.from_numpy(im_in2.astype(float)).float())
            cct = torch.autograd.Variable(torch.from_numpy(cc_in.astype(float)).float())
            #cct1 = torch.autograd.Variable(torch.from_numpy(cc_in1.astype(float)).float())
            # label = torch.autograd.Variable(torch.tensor(1-(np.absolute(float(shift_len))/self.opt.frame_len)))
            label = torch.autograd.Variable(torch.tensor(label)).long()
            #label1 = torch.autograd.Variable(torch.tensor(float(shift_len1)))
            # print('ok')
            if self.opt.norm_img:
                return imtv/255. -0.5 , cct, imtv1/255. -0.5, imtv1/255. -0.5, label
            return imtv, cct, imtv1, imtv2, label

class AudioVideoDataset_ours_tri_duoinput_lip_2losses(data.Dataset):

    def __init__(self, opt, mode = 'train', type = 'shift', seed = np.random.seed(0)):
        super(AudioVideoDataset_ours_tri_duoinput_lip_2losses, self).__init__()

        self.chunk_index = 0
        self.mode = mode
        self.type =  type
        self.opt = opt
            
        folders_all = glob.glob(opt.data_dir + '/*')
        folder_list_train = ['data1_selected_full_face_v2', 'data2_selected_full_face', 'local_143_selected_full_face_v2', 'data1_selected_full_face']
        folder_list_val = ['devset']


        if self.mode == 'train':
            folders = [i for i in folders_all if i.split('/')[-1] in folder_list_train]
        else:
            folders = [i for i in folders_all if i.split('/')[-1] in folder_list_val]

        video_list = []
        audio_list = []
        for folder in folders:
            dirs = glob.glob(folder + '/*')
            for dir in dirs:
                video_list.extend(glob.glob(dir + '/center_crop_112/*.npy'))
                audio_list.extend(glob.glob(dir + '/audio_feat/*.npy'))
       
        self.audio_video_list = [i for i in video_list if i.replace('center_crop_112', 'audio_feat')in audio_list]

        #if mode == 'train':
        #    self.audio_video_list = [i for i in audio_video_list if i[i.rfind('/')+1:i.find('.npy')] in train_names]
        #else:
        #    self.audio_video_list = audio_video_list
        #    self.audio_video_list = [i for i in audio_video_list if i[i.rfind('/') + 1:i.find('.npy')] not in train_names]

        self.num = len(self.audio_video_list)
        self.count_chunk = -1
        print('# Pairs found:', self.num)
        #pdb.set_trace()
    def len_chunk(self):
        return self.opt.chunk_size

    def __len__(self):
        return self.num

    
    def load_chunk(self):
        self.audio_video_list = random.sample(self.audio_video_list_all, 100)

    def __getitem__(self, index):
        '''
        if self.mode == 'train':
            self.count_chunk += 1
            if self.count_chunk%1000 == 0:
                self.load_chunk()
                self.count_chunk = 0
            index = self.count_chunk
        else:
            self.audio_video_list = self.audio_video_list_all
        print(index)
        '''
        videofile = self.audio_video_list[index]
        audiofile = self.audio_video_list[index].replace('center_crop_112', 'audio_feat')#.replace('.npy', '.wav')

        if self.mode == 'eval_short':
            imtv_all = load_npy_lip(videofile)
            #print('imtv', imtv.shape)
            cct_all = np.load(audiofile)
            #imtv = np.expand_dims(imtv, axis = 0)
            #cct = np.expand_dims(cct, axis = 0)

            imtv_all = np.transpose(imtv_all, (1, 0, 2, 3))
            #imtv = torch.autograd.Variable(torch.from_numpy(imtv.astype(float)).float())

            #cct = torch.autograd.Variable(torch.from_numpy(cct.astype(float)).float())
            imtv_neg1 = []
            #labels = []
            imtv_neg2 = []
            cct_neg = []

            lastframe = imtv_all.shape[1]-self.opt.frame_len-1
            for i in range (0, imtv_all.shape[1], self.opt.frame_len):
                #imtv = imtv_all[:, :, i:i+self.opt.frame_len, :, :] 
                cc_in, new_idx = get_random_audio_lip([index], len(self.audio_video_list)-1, self.audio_video_list, self.opt.frame_len)
                cct_neg.extend(cc_in)


                imtv1, new_idx = get_random_video_lip([index], len(self.audio_video_list)-1, self.audio_video_list, self.opt.frame_len)
                imtv2, _ = get_random_video_lip([index], len(self.audio_video_list)-1, self.audio_video_list, self.opt.frame_len)
                imtv_neg1.extend(imtv1)
                imtv_neg2.extend(imtv2)
                #labels.extend([0, 1])
            #pdb.set_trace()
            cct_neg = np.array(cct_neg)
            cct_neg = np.expand_dims(np.transpose(cct_neg, (1, 0, 2)).reshape((cct_neg.shape[1], -1)), axis = 0)
            imtv_neg1 = np.array(imtv_neg1)
            imtv_neg2 = np.array(imtv_neg2)
                        
            imtv_neg1 = np.expand_dims(imtv_neg1.reshape((-1, imtv_neg1.shape[2], imtv_neg1.shape[3])), axis = 0)
            imtv_neg2 = np.expand_dims(imtv_neg2.reshape((-1, imtv_neg2.shape[2], imtv_neg2.shape[3])), axis = 0)


            '''
            imtv1 = np.expand_dims(imtv1, axis = 0)
            imtv1 = np.repeat(imtv1, 20, axis = 0)

            imtv2 = np.expand_dims(imtv2, axis = 0)
            imtv2 = np.repeat(imtv2, 20, axis = 0)
            '''
            #imtv_all = torch.autograd.Variable(torch.from_numpy(imtv_all.astype(float)).float())
            #imtv_neg1 = torch.autograd.Variable(torch.from_numpy(imtv_neg1.astype(float)).float())
            #imtv_neg2 = torch.autograd.Variable(torch.from_numpy(imtv_neg2.astype(float)).float())
            if self.opt.norm_img:
                return imtv_all/255. -0.5 , cct_all, cct_neg, imtv_neg1/255. -0.5, imtv_neg2/255. -0.5
            return imtv_all, cct_all, cct_neg, imtv_neg1, imtv_neg2
        else:
            im = load_npy_lip(videofile)
            cc = np.load(audiofile)


            try:
                start_frame = random.randint(0, min(cc.shape[2]/4 - self.opt.frame_len, len(im) - self.opt.frame_len))
            except:
                start_frame = 0
            #    print('cc shape', cc.shape)
            #    print('im shape', im.shape)
            #    pdb.set_trace()
            # print('start from', start_frame)

            im_batch = im[start_frame:start_frame + self.opt.frame_len, :, :, :]
            # print(im_batch.shape)

            # pdb.set_trace()
            im_in = np.transpose(im_batch, (1, 0, 2, 3))
            

      
            #shift_len1 = 0
            #while -6 <= shift_len <= 6:
            #label = random.randint(0, 1)
            ex_list = [index]

            if self.opt.settings == 0:
           
                shift_len1 = 0
                shift_len = 0
                while shift_len == 0:
                    shift_len = random.randint(int(max(-start_frame, -self.opt.vshift)), int(min(self.opt.vshift, max(0, min((cc.shape[2]/4 - self.opt.frame_len - start_frame), (len(im) - start_frame - self.opt.frame_len))))))
                cc_in = cc[:, :, (start_frame + shift_len) * 4:(start_frame + shift_len + self.opt.frame_len) * 4]

                comp_type = random.randint(0, 3)

                if comp_type == 2:
                    random_audio = random.randint(0, len(self.audio_video_list)-1)
                    while abs(random_audio) == abs(index):
                        random_audio = random.randint(0, len(self.audio_video_list)-1)

                    random_audio = self.audio_video_list[random_audio].replace('center_crop_112', 'audio_feat')#.replace('.npy', '.wav')
                    random_cc = np.load(random_audio)
                    random_start_frame = random.randint(0, random_cc.shape[2] - self.opt.frame_len*4)
                    cc_in1 = random_cc[:, :, random_start_frame:random_start_frame + self.opt.frame_len * 4]
                    shift_len1 = 10000
                    
                elif (comp_type == 0 or comp_type == 1 ):# and not shift_len == 0:    
                    cc_in1 = cc[:, :, start_frame * 4:(start_frame + self.opt.frame_len) * 4]
                    shift_len1 = 0

                else:
                    while shift_len1 == shift_len:
                        shift_len1 = random.randint(int(max(-start_frame, -self.opt.vshift)), int(min(self.opt.vshift, max(0, min((cc.shape[2]/4 - self.opt.frame_len - start_frame), (len(im) - start_frame - self.opt.frame_len))))))
                    cc_in1 = cc[:, :, (start_frame + shift_len1) * 4:(start_frame + shift_len1 + self.opt.frame_len) * 4]
            
                    
                '''
                im_in1, new_idx = get_random_video(ex_list, self.num-1, self.audio_video_list, self.opt.frame_len)
                ex_list.append(new_idx)
                im_in2, _ = get_random_video(ex_list, self.num-1, self.audio_video_list, self.opt.frame_len)
                '''
                
            
            elif self.opt.settings == 1:
                cc_in = cc[:, :, start_frame * 4:(start_frame + self.opt.frame_len) * 4]
                shift_len = 0
                random_audio = 0
                #if comp_type == 2:
                while abs(random_audio) == abs(index):
                    random_audio = random.randint(0, len(self.audio_video_list)-1)
                
                    #random_audio = random.randint(0, self.num)

                random_audio = self.audio_video_list[random_audio].replace('center_crop_112', 'audio_feat')#.replace('.npy', '.wav')
                random_cc = np.load(random_audio)
                random_start_frame = random.randint(0, random_cc.shape[2] - self.opt.frame_len*4)
                cc_in1 = random_cc[:, :, random_start_frame:random_start_frame + self.opt.frame_len * 4]
                shift_len1 = 10000

            #else:
            im_in1, _ = get_random_video_lip([index], len(self.audio_video_list)-1, self.audio_video_list, self.opt.frame_len)
            im_in2, _ = get_random_video_lip([index], len(self.audio_video_list)-1, self.audio_video_list, self.opt.frame_len)
            # cc_in = torch.cat(cc_batch, 0)

            if not cc_in.shape[2] == self.opt.frame_len*4:
                print('wrong shape')
                #pdb.set_trace()
                _cc_in = np.zeros((1, 13, self.opt.frame_len*4))
                _cc_in[:, :, :cc_in.shape[2]] = cc_in
                cc_in = _cc_in
         
            imtv = torch.autograd.Variable(torch.from_numpy(im_in.astype(float)).float())
            imtv1 = torch.autograd.Variable(torch.from_numpy(im_in1.astype(float)).float())
            imtv2 = torch.autograd.Variable(torch.from_numpy(im_in2.astype(float)).float())
            #imtv = torch.autograd.Variable(torch.from_numpy(im_in.astype(float)).float())
            cct = torch.autograd.Variable(torch.from_numpy(cc_in.astype(float)).float())
            cct1 = torch.autograd.Variable(torch.from_numpy(cc_in1.astype(float)).float())

            if self.opt.settings == 1:
                return imtv, cct, cct1, torch.autograd.Variable(torch.tensor(float(1)).long()), torch.autograd.Variable(torch.tensor(float(0)).long()), imtv1, imtv2
            # label = torch.autograd.Variable(torch.tensor(1-(np.absolute(float(shift_len))/self.opt.frame_len)))
            if shift_len1 == 0:
                label = torch.autograd.Variable(torch.tensor(float(1)).long())
            else:
                label = torch.autograd.Variable(torch.tensor(float(0)).long())

            #label1 = torch.autograd.Variable(torch.tensor(float(shift_len1)))
            #label1 = torch.autograd.Variable(torch.tensor(float(shift_len1)))
            # print('ok')
            # if self.opt.norm_img:
            #     return imtv/255. -0.5 , cct, imtv1/255. -0.5, imtv1/255. -0.5, label
            #return imtv, cct, imtv1, imtv2, label

            if abs(shift_len) < abs(shift_len1):
                return imtv, cct, cct1, label, imtv1, imtv2
            else:
                return imtv, cct1, cct, label, imtv1, imtv2

class AudioVideoDataset_ours(data.Dataset):

    def __init__(self, opt, mode = 'train', type = 'shift', seed = np.random.seed(0)):
        super(AudioVideoDataset_ours, self).__init__()

        self.chunk_index = 0

        with open(opt.train_file, 'rb') as fil:
            train_list = fil.readlines()

        train_names = [i[i.find('audio/')+6:i.find('.wav')] for i in train_list]
        #pdb.set_trace()

        #self.batch_size = opt.batch_size

        #import ipdb
        #ipdb.set_trace()
        self.mode = mode
        self.type =  type
        self.opt = opt
        dirs = glob.glob(self.opt.data_dir + '/*')
        video_list = []
        audio_list = []
        for dir in dirs:
            video_list.extend(glob.glob(dir + '/face_crop/*.npy'))
            audio_list.extend(glob.glob(dir + '/audio_feat/*.npy'))
        #print('npys', len(video_list))
        '''
        for video in video_list:
            obj = load_pickle1(video)
            if not obj.shape[0] == 224:
                #pdb.set_trace()
                video_list.remove(video)
                os.remove(video)
        '''
        #video_list = os.listdir((os.path.join(self.opt.data_dir, self.opt.work_dir, self.opt.video_dir)))
        #audio_list = os.listdir((os.path.join(self.opt.data_dir, self.opt.work_dir, self.opt.audio_dir)))


        audio_video_list = [i for i in video_list if i.replace('face_crop', 'audio_feat')in audio_list]

        if mode == 'train':
            self.audio_video_list = [i for i in audio_video_list if i[i.rfind('/')+1:i.find('.npy')] in train_names]
        else:
            self.audio_video_list = [i for i in audio_video_list if i[i.rfind('/') + 1:i.find('.npy')] not in train_names]

        self.num = len(self.audio_video_list)

        print('# Pairs found:', self.num)

    def len_chunk(self):
        return self.opt.chunk_size

    def __len__(self):
        return self.num

    def load_next_chunk(self):
        self.chunk_video = []
        self.chunk_audio = []

        if self.chunk_index+self.opt.chunk_size <= len(self.audio_video_list):
            end_index = self.chunk_index+self.opt.chunk_size
        else:
            end_index = len(self.audio_video_list)
        #pdb.set_trace()
        for index in range(self.chunk_index, end_index):
            videofile = self.audio_video_list[index]
            audiofile = self.audio_video_list[index].replace('face_crop', 'audio_feat').replace('.npy', '.wav')

            im = load_pickle(videofile)
            im = np.transpose(im, (1, 0, 2, 3))
            self.chunk_video.append(im)
            sample_rate, audio = wavfile.read(audiofile)
            # print(sample_rate)
            # print('audio_feat', audio.shape)
            mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
            mfcc = np.stack([np.array(i) for i in mfcc])
            # cc = np.expand_dims(np.expand_dims(mfcc, axis = 0), axis = 0)
            cc = np.expand_dims(mfcc, axis = 0)  # , axis = 0)
            self.chunk_audio.append(cc)
        if end_index - self.chunk_index < self.opt.chunk_size:
            end_index = self.opt.chunk_size - (end_index - self.chunk_index)
            for index in range(end_index):
                videofile = self.audio_video_list[index]
                audiofile = self.audio_video_list[index].replace('face_crop', 'audio_feat').replace('.npy', '.wav')

                im = load_pickle(videofile)
                im = np.transpose(im, (1, 0, 2, 3))
                self.chunk_video.append(im)
                sample_rate, audio = wavfile.read(audiofile)
                # print(sample_rate)
                # print('audio_feat', audio.shape)
                mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
                mfcc = np.stack([np.array(i) for i in mfcc])
                # cc = np.expand_dims(np.expand_dims(mfcc, axis = 0), axis = 0)
                cc = np.expand_dims(mfcc, axis = 0)  # , axis = 0)
                self.chunk_audio.append(cc)
        self.chunk_index = end_index%len(self.audio_video_list)
        #if self.chunk_index >= len(self.audio_video_list):
        #    self.chunk_index = 0
    '''
    def get_item_chunk(self, index):
        
        videofile = self.audio_video_list[index]
        audiofile = self.audio_video_list[index].replace('face_crop', 'audio_feat').replace('.pkl', '.wav')

        im = load_pickle(videofile)
        #print('im_shape', im.shape)
        start_frame = random.randint(0, len(im)-self.opt.frame_len)
        #print('start from', start_frame)

        im_batch = im[start_frame:start_frame + self.opt.frame_len, :, :, :]
        #print(im_batch.shape)

        # pdb.set_trace()
        im_in = np.transpose(im_batch, (1, 0, 2, 3))

       # print('im_in', im_in.shape)

        if self.type == 'replace':
            random_audio = random.randint(0, self.num)
            while random_audio == index:
                random_audio = random.randint(0, self.num)

            audiofile = self.audio_video_list[random_audio].replace('face_crop', 'audio_feat').replace('.pkl', '.wav')

        sample_rate, audio = wavfile.read(audiofile)
        #print(sample_rate)
        #print('audio_feat', audio.shape)
        mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
        mfcc = np.stack([np.array(i) for i in mfcc])
        #cc = np.expand_dims(np.expand_dims(mfcc, axis = 0), axis = 0)
        cc = np.expand_dims(mfcc, axis = 0)#, axis = 0)
        #print('cc', cc.shape)
        #start_frame = random.randint(opt.frame_len, len(im)-2*opt.frame_len)
        #cc_batch = [cc[:, :, :, start_frame * 4:(start_frame +  opt.frame_len) * 4 ]]
        
        im = self.chunk_video[index]
        print('imshape', im.shape)
        start_frame = random.randint(0, im.shape[1] - self.opt.frame_len)
        print('start from', start_frame)

        im_in = im[:, start_frame:start_frame + self.opt.frame_len, :, :]
        if self.type == 'replace':
            random_audio = random.randint(0, self.num)
            while random_audio == index:
                random_audio = random.randint(0, self.num)
            index = random_audio

        cc = self.chunk_audio[index]
        print('cc', cc.shape)

        if self.type == 'shift':

            label = random.randint(0, 1)
            if label == 0:
                shift_len = 0
                while -6 <= shift_len <= 6:
                #while shift_len == 0:
                    shift_len = random.randint(max(-self.opt.frame_len, -start_frame), min(self.opt.frame_len, im.shape[1] - (start_frame+self.opt.frame_len)))
                cc_in = cc[ :, :, (start_frame+shift_len) * 4:(start_frame+shift_len +  self.opt.frame_len) * 4 ]
            else:
                cc_in = cc[:, :, start_frame * 4:(start_frame + self.opt.frame_len) * 4]


        else:
            cc_in = cc[ :, :, start_frame * 4:(start_frame + self.opt.frame_len) * 4]

        #cc_in = torch.cat(cc_batch, 0)
        print(im_in.shape)
        print(cc_in.shape)
        imtv = torch.autograd.Variable(torch.from_numpy(im_in.astype(float)).float())
        cct = torch.autograd.Variable(torch.from_numpy(cc_in.astype(float)).float())
        #label = torch.autograd.Variable(torch.tensor(1-(np.absolute(float(shift_len))/self.opt.frame_len)))
        label = torch.autograd.Variable(torch.tensor(float(label)))
        #print('ok')
        if self.type == 'shift':
            return imtv, cct, label
        else: #self.mode == 'rep'
            return  imtv, cct
    '''
    def __getitem__(self, index):
        videofile = self.audio_video_list[index]
        audiofile = self.audio_video_list[index].replace('face_crop', 'audio_feat')#.replace('.npy', '.wav')

        #im = load_pickle(videofile)
        im = load_npy(videofile)
        # print('im_shape', im.shape)
        start_frame = random.randint(0, len(im) - self.opt.frame_len)
        # print('start from', start_frame)

        im_batch = im[start_frame:start_frame + self.opt.frame_len, :, :, :]
        # print(im_batch.shape)

        # pdb.set_trace()
        im_in = np.transpose(im_batch, (1, 0, 2, 3))

        #print('im_in', im_in.shape)

        if self.type == 'replace':
            random_audio = random.randint(0, self.num)
            while random_audio == index:
                random_audio = random.randint(0, self.num)

            audiofile = self.audio_video_list[random_audio].replace('face_crop', 'audio_feat')#.replace('.npy', '.wav')
        '''
        sample_rate, audio = wavfile.read(audiofile)
        # print(sample_rate)
        # print('audio_feat', audio.shape)
        mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
        mfcc = np.stack([np.array(i) for i in mfcc])
        # cc = np.expand_dims(np.expand_dims(mfcc, axis = 0), axis = 0)
        cc = np.expand_dims(mfcc, axis = 0)  # , axis = 0)
        #print('cc', cc.shape)
        # start_frame = random.randint(opt.frame_len, len(im)-2*opt.frame_len)
        # cc_batch = [cc[:, :, :, start_frame * 4:(start_frame +  opt.frame_len) * 4 ]]
        '''
        cc = np.load(audiofile)
        if self.type == 'shift':
            #print('shift')
            #print(self.opt.frame_len)
            #print(start_frame)
            #print(shift_len)

            label = random.randint(0, 1)
            if label == 0:
                while True:
                    #
                #while shift_len == 0:
                    #shift_len = random.randint(max(-self.opt.frame_len, -start_frame), max(0, (len(im) - start_frame - self.opt.frame_len)))
                    shift_len = random.randint(max(-start_frame, -self.opt.vshift), min(self.opt.vshift, max(0, (len(im) - start_frame - self.opt.frame_len))))

                    #if -6 < shift_len or shift_len > 6:
                    if not shift_len == 0:
                        break
                    
                #print('ok')
                cc_in = cc[ :, :, (start_frame+shift_len) * 4:(start_frame+shift_len +  self.opt.frame_len) * 4 ]
            else:
                cc_in = cc[:, :, start_frame * 4:(start_frame + self.opt.frame_len) * 4]
            #print('shift_len', shift_len)

        else:
            cc_in = cc[:, :, start_frame * 4:(start_frame + self.opt.frame_len) * 4]

        # cc_in = torch.cat(cc_batch, 0)

        if not cc_in.shape[2] == 20:
            #print('wrong shape')
            #pdb.set_trace()
            cc_in1 = np.zeros((1, 13, 20))
            cc_in1[:, :, :cc_in.shape[2]] = cc_in
            cc_in = cc_in1
        imtv = torch.autograd.Variable(torch.from_numpy(im_in.astype(float)).float())
        cct = torch.autograd.Variable(torch.from_numpy(cc_in.astype(float)).float())
        # label = torch.autograd.Variable(torch.tensor(1-(np.absolute(float(shift_len))/self.opt.frame_len)))
        label = torch.autograd.Variable(torch.tensor(float(label)))
        # print('ok')
        if self.type == 'shift':
            return imtv, cct, label
        else:  # self.mode == 'rep'
            return imtv, cct


class AudioVideoDataset_ours_lip(data.Dataset):

    def __init__(self, opt, mode = 'train', type = 'shift', seed = np.random.seed(0)):
        super(AudioVideoDataset_ours_lip, self).__init__()

        self.chunk_index = 0
        self.mode = mode
        self.type =  type
        self.opt = opt
            
        folders_all = glob.glob(opt.data_dir + '/*')
        folder_list_train = ['data1_selected_full_face_v2', 'data2_selected_full_face', 'local_143_selected_full_face_v2', 'data1_selected_full_face']
        folder_list_val = ['devset']


        if self.mode == 'train':
            folders = [i for i in folders_all if i.split('/')[-1] in folder_list_train]
        else:
            folders = [i for i in folders_all if i.split('/')[-1] in folder_list_val]

        video_list = []
        audio_list = []
        for folder in folders:
            dirs = glob.glob(folder + '/*')
            for dir in dirs:
                video_list.extend(glob.glob(dir + '/center_crop_112/*.npy'))
                audio_list.extend(glob.glob(dir + '/audio_feat/*.npy'))
       
        self.audio_video_list = [i for i in video_list if i.replace('center_crop_112', 'audio_feat')in audio_list]

        #if mode == 'train':
        #    self.audio_video_list = [i for i in audio_video_list if i[i.rfind('/')+1:i.find('.npy')] in train_names]
        #else:
        #    self.audio_video_list = audio_video_list
        #    self.audio_video_list = [i for i in audio_video_list if i[i.rfind('/') + 1:i.find('.npy')] not in train_names]

        self.num = len(self.audio_video_list)

        print('# Pairs found:', self.num)

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        videofile = self.audio_video_list[index]
        audiofile = self.audio_video_list[index].replace('center_crop_112', 'audio_feat')#.replace('.npy', '.wav')


        if self.mode == 'eval':
            imtv = load_npy_lip(videofile)
            cct = np.load(audiofile)
            imtv = np.expand_dims(imtv, axis = 0)
            cct = np.expand_dims(cct, axis = 0)
            #imtv = np.transpose(imtv, (1, 0, 2, 3))
            print('shape', imtv.shape)

            imtv = np.transpose(imtv, (0, 2, 1, 3, 4))
            imtv = torch.autograd.Variable(torch.from_numpy(imtv.astype(float)).float())

            cct = torch.autograd.Variable(torch.from_numpy(cct.astype(float)).float())

            cct_neg = []

            lastframe = imtv.shape[2]-self.opt.frame_len-1
            for i in range (0, imtv.shape[2], self.opt.frame_len):
                #imtv = imtv_all[:, :, i:i+self.opt.frame_len, :, :] 
                cc_in, new_idx = get_random_audio_lip([index], len(self.audio_video_list)-1, self.audio_video_list, self.opt.frame_len)
                cct_neg.extend(cc_in)

            cct_neg = np.array(cct_neg)
            cct_neg = np.expand_dims(np.transpose(cct_neg, (1, 0, 2)).reshape((cct_neg.shape[1], -1)), axis = 0)
            cct_neg = np.expand_dims(cct_neg, axis = 0)
            cct_neg = torch.autograd.Variable(torch.from_numpy(cct_neg.astype(float)).float())
            



            if self.opt.norm_img:
                imtv = imtv/255. - 0.5
            return imtv, cct, cct_neg
        
        #im = load_pickle(videofile)
        im = load_npy_lip(videofile)
        cc = np.load(audiofile)
        # print('im_shape', im.shape)
        try:
            start_frame = random.randint(0, min(cc.shape[2]/4 - self.opt.frame_len, len(im) - self.opt.frame_len))
        except:
            start_frame = 0
        # print('start from', start_frame)

        im_batch = im[start_frame:start_frame + self.opt.frame_len, :, :, :]
        # print(im_batch.shape)

        # pdb.set_trace()
        im_in = np.transpose(im_batch, (1, 0, 2, 3))

        #print('im_in', im_in.shape)
        
        if self.type == 'replace':
            random_audio = random.randint(0, self.num)
            while random_audio == index:
                random_audio = random.randint(0, self.num)

            audiofile = self.audio_video_list[random_audio].replace('center_crop_112', 'audio_feat')#.replace('.npy', '.wav')
        '''
        sample_rate, audio = wavfile.read(audiofile)
        # print(sample_rate)
        # print('audio_feat', audio.shape)
        mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
        mfcc = np.stack([np.array(i) for i in mfcc])
        # cc = np.expand_dims(np.expand_dims(mfcc, axis = 0), axis = 0)
        cc = np.expand_dims(mfcc, axis = 0)  # , axis = 0)
        #print('cc', cc.shape)
        # start_frame = random.randint(opt.frame_len, len(im)-2*opt.frame_len)
        # cc_batch = [cc[:, :, :, start_frame * 4:(start_frame +  opt.frame_len) * 4 ]]
        '''
        
        if self.type == 'shift':
            #print('shift')
            #print(self.opt.frame_len)
            #print(start_frame)
            #print(shift_len)

            label = random.randint(0, 1)
            if label == 0:
                while True:
                    #
                #while shift_len == 0:
                    #shift_len = random.randint(max(-self.opt.frame_len, -start_frame), max(0, (len(im) - start_frame - self.opt.frame_len)))
                    shift_len = random.randint(int(max(-start_frame, -self.opt.vshift)), int(min(self.opt.vshift, max(0, min((cc.shape[2]/4 - self.opt.frame_len - start_frame), (len(im) - start_frame - self.opt.frame_len))))))

                    #if -6 < shift_len or shift_len > 6:
                    if not shift_len == 0:
                        break
                    
                #print('ok')
                cc_in = cc[ :, :, (start_frame+shift_len) * 4:(start_frame+shift_len +  self.opt.frame_len) * 4 ]
            else:
                cc_in = cc[:, :, start_frame * 4:(start_frame + self.opt.frame_len) * 4]
            #print('shift_len', shift_len)

        else:
            cc_in = cc[:, :, start_frame * 4:(start_frame + self.opt.frame_len) * 4]

        # cc_in = torch.cat(cc_batch, 0)

        if not cc_in.shape[2] == self.opt.frame_len*4:
            print('wrong shape')
            #pdb.set_trace()
            cc_in1 = np.zeros((1, 13, self.opt.frame_len*4))
            cc_in1[:, :, :cc_in.shape[2]] = cc_in
            cc_in = cc_in1
        imtv = torch.autograd.Variable(torch.from_numpy(im_in.astype(float)).float())
        cct = torch.autograd.Variable(torch.from_numpy(cc_in.astype(float)).float())
        # label = torch.autograd.Variable(torch.tensor(1-(np.absolute(float(shift_len))/self.opt.frame_len)))
        label = torch.autograd.Variable(torch.tensor(float(label)))
        # print('ok')
        if self.type == 'shift':
            return imtv, cct, label
        else:  # self.mode == 'rep'
            return imtv, cct

class AudioVideoDataset_ours_finetune(data.Dataset):

    def __init__(self, opt, mode = 'train', type = 'shift', seed = np.random.seed(0)):
        super(AudioVideoDataset_ours_finetune, self).__init__()

        self.chunk_index = 0
        self.mode = mode
        self.type =  type
        self.opt = opt
            
        folders_all = glob.glob(opt.data_dir + '/*')
        folder_list_train = ['data1_selected_full_face_v2', 'data2_selected_full_face', 'local_143_selected_full_face_v2', 'data1_selected_full_face']
        folder_list_val = ['devset']


        if self.mode == 'train':
            folders = [i for i in folders_all if i.split('/')[-1] in folder_list_train]
        else:
            folders = [i for i in folders_all if i.split('/')[-1] in folder_list_val]

        video_list = []
        audio_list = []
        for folder in folders:
            dirs = glob.glob(folder + '/*')
            for dir in dirs:
                audio_list.extend(glob.glob(dir + '/audio_feat/*.npy'))
                dir_video = dir.replace('publicfast','all_data')
                video_list.extend(glob.glob(dir + '/face_crop/*.npy'))

       
        self.audio_video_list = [i for i in video_list if i.replace('face_crop', 'audio_feat').replace('all_data','publicfast')in audio_list]

        #if mode == 'train':
        #    self.audio_video_list = [i for i in audio_video_list if i[i.rfind('/')+1:i.find('.npy')] in train_names]
        #else:
        #    self.audio_video_list = audio_video_list
        #    self.audio_video_list = [i for i in audio_video_list if i[i.rfind('/') + 1:i.find('.npy')] not in train_names]

        self.num = len(self.audio_video_list)

        print('# Pairs found:', self.num)

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        videofile = self.audio_video_list[index]
        audiofile = self.audio_video_list[index].replace('face_crop', 'audio_feat')#.replace('.npy', '.wav')


        if self.mode == 'eval':
            imtv = load_npy(videofile)
            cct = np.load(audiofile)
            imtv = np.expand_dims(imtv, axis = 0)
            cct = np.expand_dims(cct, axis = 0)
            #imtv = np.transpose(imtv, (1, 0, 2, 3))
            print('shape', imtv.shape)

            imtv = np.transpose(imtv, (0, 2, 1, 3, 4))
            imtv = torch.autograd.Variable(torch.from_numpy(imtv.astype(float)).float())

            cct = torch.autograd.Variable(torch.from_numpy(cct.astype(float)).float())

            cct_neg = []

            lastframe = imtv.shape[2]-self.opt.frame_len-1
            for i in range (0, imtv.shape[2], self.opt.frame_len):
                #imtv = imtv_all[:, :, i:i+self.opt.frame_len, :, :] 
                cc_in, new_idx = get_random_audio([index], len(self.audio_video_list)-1, self.audio_video_list, self.opt.frame_len)
                cct_neg.extend(cc_in)

            cct_neg = np.array(cct_neg)
            cct_neg = np.expand_dims(np.transpose(cct_neg, (1, 0, 2)).reshape((cct_neg.shape[1], -1)), axis = 0)
            cct_neg = np.expand_dims(cct_neg, axis = 0)
            cct_neg = torch.autograd.Variable(torch.from_numpy(cct_neg.astype(float)).float())
            



            if self.opt.norm_img:
                imtv = imtv/255. - 0.5
            return imtv, cct, cct_neg
        
        #im = load_pickle(videofile)
        im = load_npy(videofile)
        cc = np.load(audiofile)
        # print('im_shape', im.shape)
        try:
            start_frame = random.randint(0, min(cc.shape[2]/4 - self.opt.frame_len, len(im) - self.opt.frame_len))
        except:
            start_frame = 0
        # print('start from', start_frame)

        im_batch = im[start_frame:start_frame + self.opt.frame_len, :, :, :]
        # print(im_batch.shape)

        # pdb.set_trace()
        im_in = np.transpose(im_batch, (1, 0, 2, 3))

        #print('im_in', im_in.shape)
        
        if self.type == 'replace':
            random_audio = random.randint(0, self.num)
            while random_audio == index:
                random_audio = random.randint(0, self.num)

            audiofile = self.audio_video_list[random_audio].replace('face_crop', 'audio_feat')#.replace('.npy', '.wav')
        '''
        sample_rate, audio = wavfile.read(audiofile)
        # print(sample_rate)
        # print('audio_feat', audio.shape)
        mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
        mfcc = np.stack([np.array(i) for i in mfcc])
        # cc = np.expand_dims(np.expand_dims(mfcc, axis = 0), axis = 0)
        cc = np.expand_dims(mfcc, axis = 0)  # , axis = 0)
        #print('cc', cc.shape)
        # start_frame = random.randint(opt.frame_len, len(im)-2*opt.frame_len)
        # cc_batch = [cc[:, :, :, start_frame * 4:(start_frame +  opt.frame_len) * 4 ]]
        '''
        
        if self.type == 'shift':
            #print('shift')
            #print(self.opt.frame_len)
            #print(start_frame)
            #print(shift_len)

            label = random.randint(0, 1)
            if label == 0:
                while True:
                    #
                #while shift_len == 0:
                    #shift_len = random.randint(max(-self.opt.frame_len, -start_frame), max(0, (len(im) - start_frame - self.opt.frame_len)))
                    shift_len = random.randint(int(max(-start_frame, -self.opt.vshift)), int(min(self.opt.vshift, max(0, min((cc.shape[2]/4 - self.opt.frame_len - start_frame), (len(im) - start_frame - self.opt.frame_len))))))

                    #if -6 < shift_len or shift_len > 6:
                    if not shift_len == 0:
                        break
                    
                #print('ok')
                cc_in = cc[ :, :, (start_frame+shift_len) * 4:(start_frame+shift_len +  self.opt.frame_len) * 4 ]
            else:
                cc_in = cc[:, :, start_frame * 4:(start_frame + self.opt.frame_len) * 4]
            #print('shift_len', shift_len)

        else:
            cc_in = cc[:, :, start_frame * 4:(start_frame + self.opt.frame_len) * 4]

        # cc_in = torch.cat(cc_batch, 0)

        if not cc_in.shape[2] == self.opt.frame_len*4:
            print('wrong shape')
            #pdb.set_trace()
            cc_in1 = np.zeros((1, 13, self.opt.frame_len*4))
            cc_in1[:, :, :cc_in.shape[2]] = cc_in
            cc_in = cc_in1
        imtv = torch.autograd.Variable(torch.from_numpy(im_in.astype(float)).float())
        cct = torch.autograd.Variable(torch.from_numpy(cc_in.astype(float)).float())
        # label = torch.autograd.Variable(torch.tensor(1-(np.absolute(float(shift_len))/self.opt.frame_len)))
        label = torch.autograd.Variable(torch.tensor(float(label)))
        # print('ok')
        if self.type == 'shift':
            return imtv, cct, label
        else:  # self.mode == 'rep'
            return imtv, cct



class AudioVideoDataset_ours_finetune_chunk(data.Dataset):

    def __init__(self, opt, mode = 'train', type = 'shift', seed = np.random.seed(0)):
        super(AudioVideoDataset_ours_finetune_chunk, self).__init__()

        self.chunk_index = 0
        self.mode = mode
        self.type =  type
        self.opt = opt
            
        folders_all = glob.glob(opt.data_dir + '/*')
        folder_list_train = ['data1_selected_full_face_v2', 'data2_selected_full_face', 'local_143_selected_full_face_v2', 'data1_selected_full_face']
        folder_list_val = ['devset']


        if self.mode == 'train':
            folders = [i for i in folders_all if i.split('/')[-1] in folder_list_train]
        else:
            folders = [i for i in folders_all if i.split('/')[-1] in folder_list_val]

        video_list = []
        audio_list = []
        for folder in folders:
            dirs = glob.glob(folder + '/*')
            for dir in dirs:
                audio_list.extend(glob.glob(dir + '/audio_feat/*.npy'))
                dir_video = dir.replace('publicfast','all_data')
                video_list.extend(glob.glob(dir + '/face_crop/*.npy'))
       
        self.audio_video_list = [i for i in video_list if i.replace('face_crop', 'audio_feat')in audio_list]

        #if mode == 'train':
        #    self.audio_video_list = [i for i in audio_video_list if i[i.rfind('/')+1:i.find('.npy')] in train_names]
        #else:
        #    self.audio_video_list = audio_video_list
        #    self.audio_video_list = [i for i in audio_video_list if i[i.rfind('/') + 1:i.find('.npy')] not in train_names]

        self.num = len(self.audio_video_list)
        if self.mode == 'train':          
            self.num_chunk = self.opt.chunk_size
            self.get_chunk()
        else:
            self.num_chunk = self.num

        print('# Pairs found:', self.num)

    def __len__(self):
        return self.num_chunk

    def get_chunk(self):
        idxs = random.sample(np.arange(self.num),self.num_chunk)
        self.videos = []
        self.audios = []
        for index in idxs:
            videofile = self.audio_video_list[index]
            audiofile = self.audio_video_list[index].replace('face_crop', 'audio_feat')#.replace('.npy', '.wav')
            self.videos.append(load_npy(videofile))
            self.audios.append(np.load(audiofile))


    def __getitem__(self, index):
        


        if self.mode == 'eval':
            videofile = self.audio_video_list[index]
            audiofile = self.audio_video_list[index].replace('face_crop', 'audio_feat')#.replace('.npy', '.wav')
            imtv = load_npy(videofile)
            cct = np.load(audiofile)
            imtv = np.expand_dims(imtv, axis = 0)
            cct = np.expand_dims(cct, axis = 0)
            #imtv = np.transpose(imtv, (1, 0, 2, 3))
            print('shape', imtv.shape)

            imtv = np.transpose(imtv, (0, 2, 1, 3, 4))
            imtv = torch.autograd.Variable(torch.from_numpy(imtv.astype(float)).float())

            cct = torch.autograd.Variable(torch.from_numpy(cct.astype(float)).float())

            cct_neg = []

            lastframe = imtv.shape[2]-self.opt.frame_len-1
            for i in range (0, imtv.shape[2], self.opt.frame_len):
                #imtv = imtv_all[:, :, i:i+self.opt.frame_len, :, :] 
                cc_in, new_idx = get_random_audio([index], len(self.audio_video_list)-1, self.audio_video_list, self.opt.frame_len)
                cct_neg.extend(cc_in)

            cct_neg = np.array(cct_neg)
            cct_neg = np.expand_dims(np.transpose(cct_neg, (1, 0, 2)).reshape((cct_neg.shape[1], -1)), axis = 0)
            cct_neg = np.expand_dims(cct_neg, axis = 0)
            cct_neg = torch.autograd.Variable(torch.from_numpy(cct_neg.astype(float)).float())
            



            if self.opt.norm_img:
                imtv = imtv/255. - 0.5
            return imtv, cct, cct_neg
        
        #im = load_pickle(videofile)
        im = self.videos[index]
        cc = self.audios[index]
        # print('im_shape', im.shape)
        try:
            start_frame = random.randint(0, min(cc.shape[2]/4 - self.opt.frame_len, len(im) - self.opt.frame_len))
        except:
            start_frame = 0
        # print('start from', start_frame)

        im_batch = im[start_frame:start_frame + self.opt.frame_len, :, :, :]
        # print(im_batch.shape)

        # pdb.set_trace()
        im_in = np.transpose(im_batch, (1, 0, 2, 3))

        #print('im_in', im_in.shape)
        
        '''
        sample_rate, audio = wavfile.read(audiofile)
        # print(sample_rate)
        # print('audio_feat', audio.shape)
        mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
        mfcc = np.stack([np.array(i) for i in mfcc])
        # cc = np.expand_dims(np.expand_dims(mfcc, axis = 0), axis = 0)
        cc = np.expand_dims(mfcc, axis = 0)  # , axis = 0)
        #print('cc', cc.shape)
        # start_frame = random.randint(opt.frame_len, len(im)-2*opt.frame_len)
        # cc_batch = [cc[:, :, :, start_frame * 4:(start_frame +  opt.frame_len) * 4 ]]
        '''
        
        if self.type == 'shift':
            #print('shift')
            #print(self.opt.frame_len)
            #print(start_frame)
            #print(shift_len)

            label = random.randint(0, 1)
            if label == 0:
                while True:
                    #
                #while shift_len == 0:
                    #shift_len = random.randint(max(-self.opt.frame_len, -start_frame), max(0, (len(im) - start_frame - self.opt.frame_len)))
                    shift_len = random.randint(int(max(-start_frame, -self.opt.vshift)), int(min(self.opt.vshift, max(0, min((cc.shape[2]/4 - self.opt.frame_len - start_frame), (len(im) - start_frame - self.opt.frame_len))))))

                    #if -6 < shift_len or shift_len > 6:
                    if not shift_len == 0:
                        break
                    
                #print('ok')
                cc_in = cc[ :, :, (start_frame+shift_len) * 4:(start_frame+shift_len +  self.opt.frame_len) * 4 ]
            else:
                cc_in = cc[:, :, start_frame * 4:(start_frame + self.opt.frame_len) * 4]
            #print('shift_len', shift_len)

        else:
            cc_in = cc[:, :, start_frame * 4:(start_frame + self.opt.frame_len) * 4]

        # cc_in = torch.cat(cc_batch, 0)

        if not cc_in.shape[2] == self.opt.frame_len*4:
            print('wrong shape')
            #pdb.set_trace()
            cc_in1 = np.zeros((1, 13, self.opt.frame_len*4))
            cc_in1[:, :, :cc_in.shape[2]] = cc_in
            cc_in = cc_in1
        imtv = torch.autograd.Variable(torch.from_numpy(im_in.astype(float)).float())
        cct = torch.autograd.Variable(torch.from_numpy(cc_in.astype(float)).float())
        # label = torch.autograd.Variable(torch.tensor(1-(np.absolute(float(shift_len))/self.opt.frame_len)))
        label = torch.autograd.Variable(torch.tensor(float(label)))
        # print('ok')
        if self.type == 'shift':
            return imtv, cct, label
        else:  # self.mode == 'rep'
            return imtv, cct


class AudioVideoDataset_ours_chunk(data.Dataset):

    def __init__(self, opt, mode = 'train', seed = np.random.seed(0)):
        super(AudioVideoDataset_ours_chunk, self).__init__()

        self.chunk_index = 0
        self.mode = mode
        self.type = opt.type
        self.opt = opt

        if opt.audio == '39':
            self.audio_folder = 'audio_feat_39'
        else:
            self.audio_folder = 'audio_feat'

        if  opt.video == 'lip':
            self.video_folder = 'center_crop_112'
        elif  opt.video =='lip_new':
            self.video_folder = 'crop_face_112_new'
        else:
            self.video_folder = 'face_crop_224'

            
        folders_all = glob.glob(opt.data_dir + '/*')
        folder_list_train = ['data1_selected_full_face_v2']#, 'data2_selected_full_face', 'local_143_selected_full_face_v2', 'data1_selected_full_face']
        print('just one folderrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr')
        folder_list_val = ['devset']
        folder_list_val_ours = ['test_diar']

        #pdb.set_trace()
        if self.mode == 'train':
            folders = [i for i in folders_all if i.split('/')[-1] in folder_list_train]
        elif self.mode == 'eval_ours':
            folders = [i for i in folders_all if i.split('/')[-1] in folder_list_val_ours]
        else:
            folders = [i for i in folders_all if i.split('/')[-1] in folder_list_val]

        video_list = []
        audio_list = []
        label_list = []
        for folder in folders:
            dirs = glob.glob(folder + '/*')
            for dir in dirs:
                video_list.extend(glob.glob(dir + '/'+self.video_folder+'/*.npy'))
                audio_list.extend(glob.glob(dir + '/'+self.audio_folder+'/*.npy'))
                if self.mode == 'eval_ours':
                    label_list.extend(glob.glob(dir + '/labels/*.txt'))

        if self.mode == 'eval_ours':
            #
            self.audio_video_list_ = [i for i in video_list if i.replace(self.video_folder, self.audio_folder)[:-12]+'.npy' in audio_list]
            #pdb.set_trace()
            self.audio_video_list_ = [i for i in video_list if i.replace(self.video_folder, 'labels')[:-12]+'.txt' in label_list]
            videos = set([i[:-12] for i in self.audio_video_list_])
            self.audio_video_list = []
            for video in videos:
                self.audio_video_list.append([i for i in self.audio_video_list_ if i[:-12]==video])
       
        else:
            self.audio_video_list = [i for i in video_list if i.replace(self.video_folder, self.audio_folder)in audio_list]
        #pdb.set_trace()

        #self.audio_video_list_ = [i for i in self.audio_video_list if (np.load(i,'r').shape[0] > opt.frame_len and np.load(i.replace('center_crop_112', 'audio_feat'),'r').shape[-1] > opt.frame_len*4)]
        #self.audio_video_list = self.audio_video_list_
        #if mode == 'train':
        #    self.audio_video_list = [i for i in audio_video_list if i[i.rfind('/')+1:i.find('.npy')] in train_names]
        #else:
        #    self.audio_video_list = audio_video_list
        #    self.audio_video_list = [i for i in audio_video_list if i[i.rfind('/') + 1:i.find('.npy')] not in train_names]

        self.num = len(self.audio_video_list)
        if self.mode == 'train':          
            self.num_chunk = self.opt.chunk_size
            self.get_chunk()
        else:
            self.num_chunk = self.num

        print('# Pairs found:', self.num)

    def __len__(self):
        return self.num_chunk

    def get_chunk(self):
        idxs = random.sample(list(np.arange(len(self.audio_video_list))),self.opt.chunk_size)
        self.videos = []
        self.audios = []
        del_index = []
        for index in idxs:
            videofile = self.audio_video_list[index]
            audiofile = self.audio_video_list[index].replace(self.video_folder, self.audio_folder)#.replace('.npy', '.wav')
            video = load_npy_lip(videofile)
            audio = np.load(audiofile) 
            if video.shape[0] > self.opt.frame_len and audio.shape[-1] > self.opt.frame_len*4:
                self.videos.append(video)
                self.audios.append(audio)
            else:
                del_index.append(index)
        for ele in sorted(del_index, reverse = True):  
        #del list1[ele] 
            del self.audio_video_list[ele]
        self.num_chunk = len(self.videos)


    def __getitem__(self, index):
        
        if self.mode == 'eval_ours':
            videofiles = sorted(self.audio_video_list[index])
            audiofile = self.audio_video_list[index][0].replace(self.video_folder, self.audio_folder)[:-12]+'.npy'#.replace('.npy', '.wav')
            labelfile = self.audio_video_list[index][0].replace(self.video_folder, 'labels')[:-12]+'.txt'#.replace('.npy', '.wav')
            ims = []
            for videofile in videofiles:
                print(videofile)
                imtv = load_npy_lip(videofile)
                #imtv = np.expand_dims(imtv, axis = 0)
                #imtv = np.transpose(imtv, (0, 2, 1, 3, 4))
                #imtv = torch.autograd.Variable(torch.from_numpy(imtv.astype(float)).float())
                ims.append(imtv)
            cct = np.load(audiofile)            
            #cct = np.expand_dims(cct, axis = 0)
            #cct = torch.autograd.Variable(torch.from_numpy(cct.astype(float)).float())
            label = get_ours_label_all(labelfile,ims[0].shape[0])
            #label = np.ones((ims[0].shape[0]))
            #print('synced label',ims[0].shape)

            return ims, cct, label, audiofile.split("/")[-1][:-4]

        if self.mode == 'eval':
            videofile = self.audio_video_list[index]
            audiofile = self.audio_video_list[index].replace(self.video_folder, self.audio_folder)#.replace('.npy', '.wav')
            imtv = load_npy_lip(videofile)
            cct = np.load(audiofile)
            imtv = np.expand_dims(imtv, axis = 0)
            cct = np.expand_dims(cct, axis = 0)
            #imtv = np.transpose(imtv, (1, 0, 2, 3))
            print('shape', imtv.shape)

            imtv = np.transpose(imtv, (0, 2, 1, 3, 4))
            imtv = torch.autograd.Variable(torch.from_numpy(imtv.astype(float)).float())

            cct = torch.autograd.Variable(torch.from_numpy(cct.astype(float)).float())

            cct_neg = []

            lastframe = imtv.shape[2]-self.opt.frame_len-1
            for i in range (0, imtv.shape[2], self.opt.frame_len):
                #imtv = imtv_all[:, :, i:i+self.opt.frame_len, :, :] 
                cc_in, new_idx = get_random_audio_lip([index], len(self.audio_video_list)-1, self.audio_video_list, self.opt.frame_len)
                cct_neg.extend(cc_in)

            cct_neg = np.array(cct_neg)
            cct_neg = np.expand_dims(np.transpose(cct_neg, (1, 0, 2)).reshape((cct_neg.shape[1], -1)), axis = 0)
            cct_neg = np.expand_dims(cct_neg, axis = 0)
            cct_neg = torch.autograd.Variable(torch.from_numpy(cct_neg.astype(float)).float())
            



            if self.opt.norm_img:
                imtv = imtv/255. - 0.5
            return imtv, cct, cct_neg
        
        #im = load_pickle(videofile)
        im = self.videos[index]
        cc = self.audios[index]
        # print('im_shape', im.shape)
        try:
            start_frame = random.randint(0, min(cc.shape[2]/4 - self.opt.frame_len, len(im) - self.opt.frame_len))
        except:
            start_frame = 0
        # print('start from', start_frame)

        im_batch = im[start_frame:start_frame + self.opt.frame_len, :, :, :]
        # print(im_batch.shape)

        # pdb.set_trace()
        im_in = np.transpose(im_batch, (1, 0, 2, 3))

        #print('im_in', im_in.shape)
        
        '''
        sample_rate, audio = wavfile.read(audiofile)
        # print(sample_rate)
        # print('audio_feat', audio.shape)
        mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
        mfcc = np.stack([np.array(i) for i in mfcc])
        # cc = np.expand_dims(np.expand_dims(mfcc, axis = 0), axis = 0)
        cc = np.expand_dims(mfcc, axis = 0)  # , axis = 0)
        #print('cc', cc.shape)
        # start_frame = random.randint(opt.frame_len, len(im)-2*opt.frame_len)
        # cc_batch = [cc[:, :, :, start_frame * 4:(start_frame +  opt.frame_len) * 4 ]]
        '''
        
        if self.type == 'shift_replaced':
            #print('shift')
            #print(self.opt.frame_len)
            #print(start_frame)
            #print(shift_len)

            #label = random.randint(0, 1)

            cc_in = cc[:, :, start_frame * 4:(start_frame + self.opt.frame_len) * 4]
            cc_in_replaced, _ = get_random_audio_chunk([index], self.audios, self.opt,self.audio_folder,self.video_folder,)
            #print('shift_len', shift_len)

        elif self.type == 'shift':
            cc_in = cc[:, :, start_frame * 4:(start_frame + self.opt.frame_len) * 4]

        elif self.type == 'shift_contra':
            label = random.randint(0, 1)
            if label == 0:
                while True:
                    shift_len = random.randint(int(max(-start_frame, -self.opt.vshift)), int(min(self.opt.vshift, max(0, min((cc.shape[2]/4 - self.opt.frame_len - start_frame), (len(im) - start_frame - self.opt.frame_len))))))
                    if not shift_len == 0:
                        break
                    
                #print('ok')
                cc_in = cc[ :, :, (start_frame+shift_len) * 4:(start_frame+shift_len +  self.opt.frame_len) * 4 ]
            else:
                cc_in = cc[:, :, start_frame * 4:(start_frame + self.opt.frame_len) * 4]
        # cc_in = torch.cat(cc_batch, 0)
        
        if not im_in.shape[1] == self.opt.frame_len:
            print('wrong shape im',im_in.shape)
            #pdb.set_trace()
            im_in1 = np.zeros((1, self.opt.frame_len,112,112))
            im_in1[:, :im_in.shape[1],:,:] = im_in
            im_in = im_in1

        if not cc_in.shape[2] == self.opt.frame_len*4:
            print('wrong shape',cc_in.shape)
            #pdb.set_trace()
            cc_in1 = np.zeros((1, 13, self.opt.frame_len*4))
            cc_in1[:, :, :cc_in.shape[2]] = cc_in
            cc_in = cc_in1
        imtv = torch.autograd.Variable(torch.from_numpy(im_in.astype(float)).float())
        cct = torch.autograd.Variable(torch.from_numpy(cc_in.astype(float)).float())
        # label = torch.autograd.Variable(torch.tensor(1-(np.absolute(float(shift_len))/self.opt.frame_len)))
        #
        # print('ok')
        if self.type == 'shift':
            return imtv, cct
        elif self.type == 'shift_replaced':
            cct_replaced = torch.autograd.Variable(torch.from_numpy(cc_in_replaced.astype(float)).float())
            if not cc_in_replaced.shape[2] == self.opt.frame_len*4:
                print('wrong shape replaced',cc_in_replaced.shape)
                #pdb.set_trace()
                cc_in1 = np.zeros((1, 13, self.opt.frame_len*4))
                cc_in1[:, :, :cc_in_replaced.shape[2]] = cc_in_replaced
                cc_in_replaced = cc_in1
            return imtv, cct, cct_replaced
        elif self.type == 'shift_contra':
            label = torch.autograd.Variable(torch.tensor(float(label)))
            return imtv, cct, label
            
        else:  # self.mode == 'rep'
            return imtv, cct

class AudioVideoDataset_ours_chunk_2im(data.Dataset):

    def __init__(self, opt, mode = 'train', seed = np.random.seed(0)):
        super(AudioVideoDataset_ours_chunk_2im, self).__init__()

        self.chunk_index = 0
        self.mode = mode
        self.type = opt.type
        self.opt = opt

        if opt.audio == '39':
            self.audio_folder = 'audio_feat_39'
        else:
            self.audio_folder = 'audio_feat'

        if  opt.video == 'lip':
            self.video_folder = 'center_crop_112'
        elif  opt.video =='lip_new':
            self.video_folder = 'crop_face_112_new'
        else:
            self.video_folder = 'face_crop_224'

            
        folders_all = glob.glob(opt.data_dir + '/*')
        folders_all1 = glob.glob('/publicdata_slow2/Data/Lip/new_added_extendVAD_dataset' + '/*')
        folder_list_train = ['data1_selected_full_face_v2', 'data2_selected_full_face', 'local_143_selected_full_face_v2', 'data1_selected_full_face']
        folder_list_train1 = ['local_selected_full_face_slow2_v2_extendVAD','local_selected_full_face_slow2_v1_extendVAD','local_selected_full_face_slow1_v5_extendVAD']
        #print('just one folderrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr')
        folder_list_val = ['devset']
        folder_list_val_ours = ['test_diar']

        #pdb.set_trace()
        if self.mode == 'train':
            folders = [i for i in folders_all if i.split('/')[-1] in folder_list_train]
            folders1 = [i for i in folders_all1 if i.split('/')[-1] in folder_list_train1]

        video_list = []
        audio_list = []
        label_list = []
        for folder in folders:
            dirs = glob.glob(folder + '/*')
            for dir in dirs:
                video_list.extend(glob.glob(dir + '/'+self.video_folder+'/*.npy'))
                audio_list.extend(glob.glob(dir + '/'+self.audio_folder+'/*.npy'))
        
        video_list1 = []
        audio_list1 = []
       
        for folder in folders1:
            dirs = glob.glob(folder + '/*')
            for dir in dirs:
                video_list1.extend(glob.glob(dir + '/'+self.video_folder+'/*.npy'))
                audio_list1.extend(glob.glob(dir + '/'+self.audio_folder+'/*.npy'))

        self.audio_video_list = [i for i in video_list if i.replace(self.video_folder, self.audio_folder)in audio_list]
        self.audio_video_list1 = [i for i in video_list1 if i.replace(self.video_folder, self.audio_folder)in audio_list1]
        self.audio_video_list.extend(self.audio_video_list1)
        #pdb.set_trace()

        #self.audio_video_list_ = [i for i in self.audio_video_list if (np.load(i,'r').shape[0] > opt.frame_len and np.load(i.replace('center_crop_112', 'audio_feat'),'r').shape[-1] > opt.frame_len*4)]
        #self.audio_video_list = self.audio_video_list_
        #if mode == 'train':
        #    self.audio_video_list = [i for i in audio_video_list if i[i.rfind('/')+1:i.find('.npy')] in train_names]
        #else:
        #    self.audio_video_list = audio_video_list
        #    self.audio_video_list = [i for i in audio_video_list if i[i.rfind('/') + 1:i.find('.npy')] not in train_names]

        self.num = len(self.audio_video_list)
        if self.mode == 'train':          
            self.num_chunk = self.opt.chunk_size
            self.get_chunk()
        else:
            self.num_chunk = self.num

        print('# Pairs found:', self.num)

    def __len__(self):
        return self.num_chunk

    def get_chunk(self):
        idxs = random.sample(list(np.arange(len(self.audio_video_list))),self.opt.chunk_size)
        self.videos = []
        self.audios = []
        del_index = []
        for index in idxs:
            videofile = self.audio_video_list[index]
            audiofile = self.audio_video_list[index].replace(self.video_folder, self.audio_folder)#.replace('.npy', '.wav')
            video = load_npy_lip(videofile)
            audio = np.load(audiofile) 
            if video.shape[0] > self.opt.frame_len and audio.shape[-1] > self.opt.frame_len*4:
                self.videos.append(video)
                self.audios.append(audio)
            else:
                del_index.append(index)
        for ele in sorted(del_index, reverse = True):  
        #del list1[ele] 
            del self.audio_video_list[ele]
        self.num_chunk = len(self.videos)


    def __getitem__(self, index):
        
       
        #im = load_pickle(videofile)
        im = self.videos[index]
        cc = self.audios[index]
        # print('im_shape', im.shape)
        try:
            start_frame = random.randint(0, min(cc.shape[2]/4 - self.opt.frame_len, len(im) - self.opt.frame_len))
        except:
            start_frame = 0
        # print('start from', start_frame)

        im_batch = im[start_frame:start_frame + self.opt.frame_len, :, :, :]
        # print(im_batch.shape)

        # pdb.set_trace()
        im_in = np.transpose(im_batch, (1, 0, 2, 3))

        #print('im_in', im_in.shape)
        
        '''
        sample_rate, audio = wavfile.read(audiofile)
        # print(sample_rate)
        # print('audio_feat', audio.shape)
        mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
        mfcc = np.stack([np.array(i) for i in mfcc])
        # cc = np.expand_dims(np.expand_dims(mfcc, axis = 0), axis = 0)
        cc = np.expand_dims(mfcc, axis = 0)  # , axis = 0)
        #print('cc', cc.shape)
        # start_frame = random.randint(opt.frame_len, len(im)-2*opt.frame_len)
        # cc_batch = [cc[:, :, :, start_frame * 4:(start_frame +  opt.frame_len) * 4 ]]
        '''
        
        if self.type == 'shift_replaced':
            #print('shift')
            #print(self.opt.frame_len)
            #print(start_frame)
            #print(shift_len)

            #label = random.randint(0, 1)

            cc_in = cc[:, :, start_frame * 4:(start_frame + self.opt.frame_len) * 4]
            #cc_in_replaced, _ = get_random_audio_chunk([index], self.audios, self.opt,self.audio_folder,self.video_folder,)
            im_in_replaced, _ = get_random_video_chunk([index], self.videos, self.opt,self.audio_folder,self.video_folder,)
            #print('shift_len', shift_len)

        
        if not im_in.shape[1] == self.opt.frame_len:
            print('wrong shape im',im_in.shape)
            #pdb.set_trace()
            im_in1 = np.zeros((1, self.opt.frame_len,112,112))
            im_in1[:, :im_in.shape[1],:,:] = im_in
            im_in = im_in1

        if not cc_in.shape[2] == self.opt.frame_len*4:
            print('wrong shape',cc_in.shape)
            #pdb.set_trace()
            cc_in1 = np.zeros((1, 13, self.opt.frame_len*4))
            cc_in1[:, :, :cc_in.shape[2]] = cc_in
            cc_in = cc_in1
        imtv = torch.autograd.Variable(torch.from_numpy(im_in.astype(float)).float())
        cct = torch.autograd.Variable(torch.from_numpy(cc_in.astype(float)).float())
        # label = torch.autograd.Variable(torch.tensor(1-(np.absolute(float(shift_len))/self.opt.frame_len)))
        #
        # print('ok')
        if self.type == 'shift_replaced':
            im_in_replaced = torch.autograd.Variable(torch.from_numpy(im_in_replaced.astype(float)).float())
            '''
            if not im_in_replaced.shape[1] == self.opt.frame_len:
                print('wrong shape replaced',im_in_replaced.shape)
                #pdb.set_trace()
                cc_in1 = np.zeros((1, 13, self.opt.frame_len*4))
                cc_in1[:, :, :cc_in_replaced.shape[2]] = cc_in_replaced
                cc_in_replaced = cc_in1
            '''
            return imtv, cct, im_in_replaced
            
        else:  # self.mode == 'rep'
            return imtv, cct

class AudioVideoDataset(data.Dataset):

    def __init__(self, opt, mode):
        super(AudioVideoDataset, self).__init__()

        #self.batch_size = opt.batch_size

        #import ipdb
        #ipdb.set_trace()
        self.opt = opt

        video_list = os.listdir((os.path.join(self.opt.data_dir, self.opt.work_dir, self.opt.video_dir)))
        audio_list = os.listdir((os.path.join(self.opt.data_dir, self.opt.work_dir, self.opt.audio_dir)))


        #ipdb.set_trace()

        self.audio_video_list = [i for i in video_list if i[:-4]+'.wav' in audio_list]

        self.num = len(self.audio_video_list)

        print('# Video Audio pairs found:', self.num)

    def __len__(self):
        return self.num

    def __getitem__(self, index):

        video = os.path.join(self.opt.data_dir, self.opt.work_dir, self.opt.video_dir, self.audio_video_list[index])
        audio = os.path.join(self.opt.data_dir, self.opt.work_dir, self.opt.audio_dir, self.audio_video_list[index][:-4]+'.wav')

        return video, audio

class AudioVideoDataset_mp4(data.Dataset):
    def __init__(self, opt, mode = 'train'):
        super(AudioVideoDataset_mp4, self).__init__()

        #self.batch_size = opt.batch_size

        #import ipdb
        #ipdb.set_trace()
        self.opt = opt

        self.mp4_list = glob.glob(self.opt.data_dir+'/*')
        #ipdb.set_trace()

        #self.audio_video_list = [i for i in video_list if i[:-4]+'.wav' in audio_list]

        self.num = len(self.mp4_list)

        print('# Videos found:', self.num)

    def __len__(self):
        return self.num

    def __getitem__(self, index):

        video = self.mp4_list[index]
        #audio = os.path.join(self.opt.data_dir, self.opt.work_dir, self.opt.audio_dir, self.audio_video_list[index][:-4]+'.wav')

        return video#, audio





class AudioVideoDataset_bbc(data.Dataset):
    def __init__(self, opt, mode = 'train'):
        super(AudioVideoDataset_bbc, self).__init__()

        #self.batch_size = opt.batch_size

        #import ipdb
        #ipdb.set_trace()
        self.opt = opt

        self.mp4_list = []
        dirs = glob.glob(self.opt.data_dir+'/*')
        for dir in dirs:
            mp4s = glob.glob(dir+'/*.mp4')
            self.mp4_list.extend(mp4s)

        #.mp4_list = glob.glob(self.opt.data_dir+'/*')
        #ipdb.set_trace()

        #self.audio_video_list = [i for i in video_list if i[:-4]+'.wav' in audio_list]

        self.num = len(self.mp4_list)

        print('# Videos found:', self.num)

    def __len__(self):
        return self.num

    def __getitem__(self, index):

        video = self.mp4_list[index]
        #audio = os.path.join(self.opt.data_dir, self.opt.work_dir, self.opt.audio_dir, self.audio_video_list[index][:-4]+'.wav')

        return video#, audio

class AudioVideoDataset_diar(data.Dataset):
    def __init__(self, opt, mode = 'train'):
        super(AudioVideoDataset_diar, self).__init__()

        #self.batch_size = opt.batch_size

        #import ipdb
        #ipdb.set_trace()
        self.opt = opt
        #pdb.set_trace()
        self.diar_dict = []
        with open(opt.txt_dir, 'r') as f:
            for line in f:
                if len(line) > 5:
                    if not line.find('outWavs') == -1:#line[1] == '-':
                        item = {}
                        item['audio'] = line[line.find('.')+1:line.find('wav')+3]
                    elif line[:3] == 'SPH':
                        item['video'] = [line.replace('\t\n', '').replace('SPH:', '').replace('audio', 'video').replace('.wav', '.mp4')]
                    elif line[:4] == 'INTF':
                        #print(line)
                        video_list = item['video']
                        videos = line.split('###')
                        for video in videos:
                            #if video != '':
                            video = video.replace('\t\n', '').replace('INTF:', '').replace('audio', 'video').replace('.wav', '.mp4')
                            if video != '':
                                video_list.append(video) 
                        item.update({'video': video_list})
                    elif line[:4] == 'time':
                        item['time'] = line[8:line.find('\\')]
                        #print (item['video'])
                        self.diar_dict.append(item)


        self.num = len(self.diar_dict)

        print('# Videos found:', self.num)

    def __len__(self):
        return self.num

    def __getitem__(self, index):

        item = self.diar_dict[index]
        #audio = os.path.join(self.opt.data_dir, self.opt.work_dir, self.opt.audio_dir, self.audio_video_list[index][:-4]+'.wav')

        return item#, audio