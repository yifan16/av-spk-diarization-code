import torch
import torch.nn as nn
#from utils.tools import load_npy
import numpy as np
import pdb
import time
from utils.tools import calc_pdist
import numpy
from scipy import signal


def save(model, filename):
    with open(filename, "wb") as f:
        torch.save(model, f);
        print("%s saved."%filename);

def load(filename):
    net = torch.load(filename)
    return net;
    
class S(nn.Module):
    def __init__(self, num_layers_in_fc_layers = 1024):
        super(S, self).__init__();

        self.__nFeatures__ = 24;
        self.__nChs__ = 32;
        self.__midChs__ = 32;

        self.netcnnaud = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,1), stride=(1,1)),

            nn.Conv2d(64, 192, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(1,2)),

            nn.Conv2d(192, 384, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),
            
            nn.Conv2d(256, 512, kernel_size=(5,4), padding=(0,0)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        );

        self.netfcaud = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_layers_in_fc_layers),
        );

        self.netfclip = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_layers_in_fc_layers),
        );

        self.netcnnlip = nn.Sequential(
            nn.Conv3d(3, 96, kernel_size=(5,7,7), stride=(1,2,2), padding=0),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2)),

            nn.Conv3d(96, 256, kernel_size=(1,5,5), stride=(1,2,2), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),

            nn.Conv3d(256, 256, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            nn.Conv3d(256, 256, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            nn.Conv3d(256, 256, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2)),

            nn.Conv3d(256, 512, kernel_size=(1,6,6), padding=0),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        );

    def forward_aud(self, x):

        mid = self.netcnnaud(x); # N x ch x 24 x M
        mid = mid.view((mid.size()[0], -1)); # N x (ch x 24)
        out = self.netfcaud(mid);

        return out;

    def forward_lip(self, x):

        mid = self.netcnnlip(x); 
        mid = mid.view((mid.size()[0], -1)); # N x (ch x 24)
        out = self.netfclip(mid);

        return out;

    def forward_lipfeat(self, x):

        mid = self.netcnnlip(x);
        out = mid.view((mid.size()[0], -1)); # N x (ch x 24)

        return out;

    def loadParameters(self, path):
        loaded_state = torch.load(path, map_location=lambda storage, loc: storage);

        self_state = self.state_dict();

        for name, param in loaded_state.items():

            self_state[name].copy_(param);
    
    #def load(self,filename):
    #    self.__S__ = torch.load(filename)
    #    #return net;

    def evaluate(self, opt, imtv,cct):


        self.eval();
       
        lastframe = imtv.shape[2]-6
        print('last',lastframe)
        im_feat = []
        cc_feat = []


        #pdb.set_trace()
        tS = time.time()
        for i in range(0,lastframe,opt.batch_size):
            
            im_batch = [ imtv[:,:,vframe:vframe+5,:,:] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
            #pdb.set_trace()
            im_in = torch.cat(im_batch,0)
            im_out  = self.forward_lip(im_in.cuda());
            im_feat.append(im_out.data.cpu())

            cc_batch = [ cct[:,:,:,vframe*4:vframe*4+20] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
            cc_in = torch.cat(cc_batch,0)
            cc_out  = self.forward_aud(cc_in.cuda())
            cc_feat.append(cc_out.data.cpu())

        im_feat = torch.cat(im_feat,0)
        cc_feat = torch.cat(cc_feat,0)

        # ========== ==========
        # Compute offset
        # ========== ==========
            
        #print('Compute time %.3f sec.' % (time.time()-tS))
        ##pdb.set_trace()

        dists = calc_pdist(im_feat,cc_feat,vshift=opt.vshift)
        #print('shape',len(dists))
        #print(dists[0].shape)
        mdist = torch.mean(torch.stack(dists,1),1)

        minval, minidx = torch.min(mdist,0)

        offset = opt.vshift-minidx
        conf   = torch.median(mdist) - minval

        fdist   = numpy.stack([dist[minidx].numpy() for dist in dists])
        # fdist   = numpy.pad(fdist, (3,3), 'constant', constant_values=15)
        fconf   = torch.median(mdist).numpy() - fdist
        fconfm  = signal.medfilt(fconf,kernel_size=9)
        
        numpy.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        print('Framewise conf: ')
        print(fconfm)
        print('AV offset: \t%d \nMin dist: \t%.3f\nConfidence: \t%.3f' % (offset,minval,conf))

        dists_npy = numpy.array([ dist.numpy() for dist in dists ])
        return offset.numpy(), conf.numpy(), dists_npy, minval.numpy()


    def evaluate_diar_npy(self,opt, imtv,cct):

        imtv = numpy.expand_dims(imtv, axis=0)
        cct = numpy.expand_dims(cct, axis=0)

        imtv = numpy.transpose(imtv, (0, 2,1,3,4))

        

        imtv = numpy.pad(imtv,((0,0),(0,0),(0,6),(0,0),(0,0)),'edge')

        cct = numpy.pad(cct,((0,0),(0,0),(0,0),(0,24)),'edge')

        if cct.shape[3]/float(imtv.shape[2]) < 4:
            cct = numpy.pad(cct,((0,0),(0,0),(0,0),(0,(imtv.shape[2]*4 - cct.shape[3]) )),'edge')
            print('pad cct',imtv.shape[2]*4 - cct.shape[3])


        imtv = torch.autograd.Variable(torch.from_numpy(imtv.astype(float)).float())

        cct = torch.autograd.Variable(torch.from_numpy(cct.astype(float)).float())

        




        self.eval();

        # ========== ==========
        # Load video
        # ========== ==========
        lastframe = imtv.shape[2] - 6
        print('last',lastframe)
        im_feat = []
        cc_feat = []


        #pdb.set_trace()
        tS = time.time()
        for i in range(0,lastframe,opt.batch_size):
            
            im_batch = [ imtv[:,:,vframe:vframe+5,:,:] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
            #pdb.set_trace()
            try:
                im_in = torch.cat(im_batch,0)
            except:
                pdb.set_trace()
            im_out  = self.forward_lip(im_in.cuda());
            im_feat.append(im_out.data.cpu())

            cc_batch = [ cct[:,:,:,vframe*4:vframe*4+20] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
            if len(cc_batch) < 20:
                print('short cc_batch',len(cc_batch))
                cc_batch.append(cc_batch[:20-len(cc_batch)])
            #if not cc_batch.shape[-1] ==20:
            #    pdb.set_trace()
            #    cc_batch = torch.nn.functional.pad(cc_batch,(0,0,20-cc_batch.shape[-1]))
            try:
                cc_in = torch.cat(cc_batch,0)
            except:
                pdb.set_trace()
            cc_out  = self.forward_aud(cc_in.cuda())
            cc_feat.append(cc_out.data.cpu())

        im_feat = torch.cat(im_feat,0)
        cc_feat = torch.cat(cc_feat,0)

        # ========== ==========
        # Compute offset
        # ========== ==========
            
        #print('Compute time %.3f sec.' % (time.time()-tS))
        ##pdb.set_trace()

        dists = calc_pdist(im_feat,cc_feat,vshift=opt.vshift)
        #print('shape',len(dists))
        #print(dists[0].shape)
        mdist = torch.mean(torch.stack(dists,1),1)

        minval, minidx = torch.min(mdist,0)

        offset = opt.vshift-minidx
        conf   = torch.median(mdist) - minval

        #minidx = 15

        fdist   = numpy.stack([dist[minidx].numpy() for dist in dists])
        # fdist   = numpy.pad(fdist, (3,3), 'constant', constant_values=15)
        fconf   = torch.median(mdist).numpy() - fdist
        fconfm  = signal.medfilt(fconf,kernel_size=9)
        
        numpy.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        #print('Framewise conf: ')
        #print(fconfm)
        print('AV offset: \t%d \nMin dist: \t%.3f\nConfidence: \t%.3f' % (offset,minval,conf))

        all_dists = torch.stack(dists,1).numpy()

        #low_half = numpy.sort(all_dists, axis=0)[:opt.vshift,:]
        low_half = numpy.sort(all_dists, axis=0)[opt.vshift:,:]
        frame_mean = numpy.mean(low_half,0)
        frame_max = numpy.min(all_dists,0)
        pdist = - frame_max + frame_mean


        dists_npy = numpy.array([ dist.numpy() for dist in dists ])
        return offset.numpy(), conf.numpy(), fdist, pdist, minval.numpy()
