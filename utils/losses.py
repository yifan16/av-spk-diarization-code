import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class LiftedLoss(nn.Module):
    def __init__(self, margin):
        super(LiftedLoss, self).__init__()
        self.margin = margin



    def forward(self, score, target):
        """
        Lifted loss, per "Deep Metric Learning via Lifted Structured Feature Embedding" by Song et al
        Implemented in `pytorch`
        """

        loss = 0
        counter = 0
        
        bsz = score.size(0)
        mag = (score ** 2).sum(1).expand(bsz, bsz)
        sim = score.mm(score.transpose(0, 1))
        
        dist = (mag + mag.transpose(0, 1) - 2 * sim)
        dist = torch.nn.functional.relu(dist).sqrt()
        
        for i in range(bsz):
            t_i = target[i].data[0]
            
            for j in range(i + 1, bsz):
                t_j = target[j].data[0]
                
                if t_i == t_j:
                    # Negative component
                    # !! Could do other things (like softmax that weights closer negatives)
                    l_ni = (self.margin - dist[i][target != t_i]).exp().sum()
                    l_nj = (self.margin - dist[j][target != t_j]).exp().sum()
                    l_n  = (l_ni + l_nj).log()
                    
                    # Positive component
                    l_p  = dist[i,j]
                    
                    loss += torch.nn.functional.relu(l_n + l_p) ** 2
                    counter += 1
        
        return loss / (2 * counter)

class ConstellLoss(nn.Module):
    def __init__(self, margin):
        super(ConstellLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, im_in, cc_in, target, size_average=True):
        

        dists = []
        im_in_shift = im_in[:,:,opt.vshift:opt.frame_len- opt.vshift,:,:]
        for i in range(opt.frame_len- 5):
            cc_in_shift = cc_in[:,:,:,i*4:(i+5)*4]
            pairwise_dist = torch.mm(im_in_shift, cc_in_shift)

        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self,anchor, sample,sample1, size_average=True):
        distance = (anchor - sample).pow(2).sum(1)  # .pow(.5)
        distance1 = (anchor - sample1).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance - distance1 + self.margin)
        return losses.mean() if size_average else losses.sum()


class TripletLoss1(nn.Module):
    def __init__(self, margin):
        super(TripletLoss1, self).__init__()
        self.margin = margin

    def forward(self,anchor, sample,sample1, size_average=True):
        distance = (anchor.pow(2) + sample.pow(2)).pow(0.5).sum(1)  # .pow(.5)
        distance1 =(anchor.pow(2) + sample1.pow(2)).pow(0.5).sum(1)  # .pow(.5)
        losses = F.relu(distance - distance1 + self.margin)
        return losses.mean() if size_average else losses.sum()

