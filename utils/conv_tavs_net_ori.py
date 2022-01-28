# wujian@2018

#from conf import *
import torch as th
import torch.nn as nn

import torch.nn.functional as Fn
#from audio_fea import DFComputer, iSTFT
from .resnet import resnet34,resnet18

def param(nnet, Mb=True):
    """
    Return number parameters(not bytes) in nnet
    """
    neles = sum([param.nelement() for param in nnet.parameters()])
    return neles / 10**6 if Mb else neles

class SpatiotemporalConv(nn.Module):
    """
    Spatiotemporal conv layer to process video stream
    """

    def __init__(self):
        super(SpatiotemporalConv, self).__init__()
        # 29x112x112 -> 29x56x56
        self.conv = nn.Conv3d(
            1, 64, (5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3)) #in_channels=1, out_channels=64
        self.norm = nn.BatchNorm3d(64)
        # 29x56x56 -> 29x28x28
        self.pool = nn.MaxPool3d((1, 3, 3),
                                 stride=(1, 2, 2),
                                 padding=(0, 1, 1))

    def forward(self, x):
        """
        x: NxTx1xDxD
        """
        # NxTx1xDxD => Nx1xTxDxD
        x = th.transpose(x, 1, 2)
        x = self.conv(x)
        x = self.norm(x)
        x = Fn.relu(x)
        x = self.pool(x)
        return x


        
class LipReadingNet(nn.Module):
    """
    Lip reading phoneme level networks
    """
    #def __init__(self, backend_dim=256, num_classes=44):
    def __init__(self, backend_dim=256):
        super(LipReadingNet, self).__init__()
        self.conv3d = SpatiotemporalConv()
        self.resnet = resnet18(num_classes=backend_dim)

    def forward(self, x, return_embedding=False):
        """
        x: NxTx1xD1xD2
        """
        if x.dim() != 5:
            raise RuntimeError(
                "LipReadingNet accept 5D tensor as input, got {:d}".format(
                    x.dim()))
        # NxTx1xDxD => NxCxTxDxD
        x = self.conv3d(x)
        # NxCxTxDxD => NxTxCxDxD
        x = th.transpose(x, 1, 2)
        #N, T, C, D = x.shape[:4]
        N, T, C, D1, D2 = x.shape[:5]
        # NTxCxDxD
        #x = x.reshape(N * T, C, D, D)
        x = x.reshape(N * T, C, D1, D2)
        # NTxCxDxD => NTxB
        x = self.resnet(x)
        # return lip embeddings
        if return_embedding:
            return x
        # NTxB => NxTxB
        x = x.view(N, T, -1)
        
        return x
        
class ChannelWiseLayerNorm(nn.LayerNorm):
    """
    Channel wise layer normalization
    """

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x C x T => N x T x C
        x = th.transpose(x, 1, 2)
        # LN
        x = super().forward(x)
        # N x C x T => N x T x C
        x = th.transpose(x, 1, 2)
        return x


class GlobalChannelLayerNorm(nn.Module):
    """
    Global channel layer normalization
    """

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalChannelLayerNorm, self).__init__()
        self.eps = eps
        self.normalized_dim = dim
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(th.zeros(dim, 1))
            self.gamma = nn.Parameter(th.ones(dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x 1 x 1
        mean = th.mean(x, (1, 2), keepdim=True)
        var = th.mean((x - mean)**2, (1, 2), keepdim=True)
        # N x T x C
        if self.elementwise_affine:
            x = self.gamma * (x - mean) / th.sqrt(var + self.eps) + self.beta
        else:
            x = (x - mean) / th.sqrt(var + self.eps)
        return x

    def extra_repr(self):
        return "{normalized_dim}, eps={eps}, " \
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)


def build_norm(norm, dim):
    """
    Build normalize layer
    LN cost more memory than BN
    """
    if norm not in ["cLN", "gLN", "BN"]:
        raise RuntimeError("Unsupported normalize layer: {}".format(norm))
    if norm == "cLN":
        return ChannelWiseLayerNorm(dim, elementwise_affine=True)
    elif norm == "BN":
        return nn.BatchNorm1d(dim)
    else:
        return GlobalChannelLayerNorm(dim, elementwise_affine=True)


class Conv1D(nn.Conv1d):
    """
    1D conv in ConvTasNet
    """

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))
        if squeeze:
            x = th.squeeze(x)
        return x


class ConvTrans1D(nn.ConvTranspose1d):
    """
    1D transpose conv in ConvTasNet
    """

    def __init__(self, *args, **kwargs):
        super(ConvTrans1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))
        if squeeze:
            x = th.squeeze(x)
        return x


class TasNetConv1DBlock(nn.Module):
    """
    1D convolutional block:
        Conv1x1 - PReLU - Norm - DConv - PReLU - Norm - SConv
    """

    def __init__(self,
                 in_channels=256,
                 conv_channels=512,
                 kernel_size=3,
                 dilation=1,
                 norm="cLN",
                 causal=False):
        super(TasNetConv1DBlock, self).__init__()
        # 1x1 conv
        self.conv1x1 = Conv1D(in_channels, conv_channels, 1)
        self.prelu1 = nn.PReLU()
        self.lnorm1 = build_norm(norm, conv_channels)
        dconv_pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))
        # depthwise conv
        self.dconv = nn.Conv1d(
            conv_channels,
            conv_channels,
            kernel_size,
            groups=conv_channels,
            padding=dconv_pad,
            dilation=dilation,
            bias=True)
        self.prelu2 = nn.PReLU()
        self.lnorm2 = build_norm(norm, conv_channels)
        # 1x1 conv cross channel
        self.sconv = nn.Conv1d(conv_channels, in_channels, 1, bias=True)
        # different padding way
        self.causal = causal
        self.dconv_pad = dconv_pad

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.lnorm1(self.prelu1(y))
        y = self.dconv(y)
        if self.causal:
            y = y[:, :, :-self.dconv_pad]
        y = self.lnorm2(self.prelu2(y))
        y = self.sconv(y)
        x = x + y
        return x


class DilatedLipNet(nn.Module):
    """
    Simple lip net to process lip embeddings (using dilated_conv)
    """

    def __init__(self,
                 embedding_dim=24,
                 conv_channels=128,
                 first_kernel=3,
                 num_blocks=3):
        super(DilatedLipNet, self).__init__()
        # conv with K=first_kernel
        self.conv1d = nn.Conv1d(
            embedding_dim,
            conv_channels,
            first_kernel,
            padding=((first_kernel - 1) // 2))
        # conv with K=3
        dilated_conv = []
        for d in range(num_blocks):
            dilated_conv.append(nn.PReLU())
            dilated_conv.append(nn.BatchNorm1d(conv_channels))
            dilated_conv.append(
                nn.Conv1d(
                    conv_channels,
                    conv_channels,
                    3,
                    dilation=2**d,
                    padding=2**d))
        self.dconv = nn.Sequential(*dilated_conv)

    def forward(self, x):
        # NxTxD => NxDxT
        x = th.transpose(x, 1, 2)
        # NxVxT
        x = self.conv1d(x)
        # NxVxT
        x = self.dconv(x)
        return x


class OxfordLipConv1DBlock(nn.Module):
    """
    depthwise pre-activation conv1d block used in OxfordLipNet
    """

    def __init__(self,
                 in_channels=256,
                 conv_channels=256,
                 kernel_size=3,
                 dilation=1):
        super(OxfordLipConv1DBlock, self).__init__()
        self.residual = (in_channels == conv_channels)
        self.bn = nn.BatchNorm1d(in_channels) if self.residual else None
        self.prelu = nn.PReLU() if self.residual else None
        self.dconv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            groups=in_channels,
            dilation=dilation,
            padding=(dilation * (kernel_size - 1)) // 2,
            bias=True)
        self.sconv = nn.Conv1d(in_channels, conv_channels, 1)

    def forward(self, x):
        if self.residual:
            y = self.dconv(self.bn(self.prelu(x)))
            y = self.sconv(y) + x
        else:
            y = self.dconv(x)
            y = self.sconv(y)
        return y


class OxfordLipNet(nn.Module):
    """
    Oxford like lip net to process lip embeddings
    """

    def __init__(self,
                 embedding_dim=256,
                 conv_channels=256,
                 kernel_size=3,
                 num_blocks=5):
        super(OxfordLipNet, self).__init__()
        conv1d_list = []
        for i in range(num_blocks):
            in_channels = conv_channels if i else embedding_dim
            conv1d_list.append(
                OxfordLipConv1DBlock(
                    in_channels=in_channels,
                    conv_channels=conv_channels,
                    kernel_size=kernel_size))
        self.conv1d_blocks = nn.Sequential(*conv1d_list)

    def forward(self, x):
        # NxTxD => NxDxT
        x = th.transpose(x, 1, 2)
        x = self.conv1d_blocks(x)
        return x


class AudioVisualFusion(nn.Module):
    """
    Fusion layer: audio/visual features
    """

    def __init__(self,
                 audio_features=256,
                 video_features=256*2,
                 out_features=256):
        super(AudioVisualFusion, self).__init__()
        self.audio_dim = audio_features
        self.video_dim = video_features
        self.conv1d = nn.Conv1d(audio_features + video_features, out_features,
                                1)

    def forward(self, a, v):
        if a.size(1) != self.audio_dim or v.size(1) != self.video_dim:
            raise RuntimeError("Dimention mismatch for audio/video features, "
                               "{:d}/{:d} vs {:d}/{:d}".format(
                                   a.size(1), v.size(1), self.audio_dim,
                                   self.video_dim))
        # upsample visual
        v = Fn.interpolate(v, size=a.size(-1))
        # concat: n x (A+V) x T
        y = th.cat([a, v], dim=1)
        # conv1d
        return self.conv1d(y)



def foo_conv_tas_visnet():
    x = th.rand(4, 1000)
    v = th.rand(4, 2, 256)
    nnet = ConvTavsNet(norm="BN", L=40, D=5, V=512)
    # print(nnet)
    print("ConvTavsNet #param: {:.2f}M".format(param(nnet)))
    s = nnet(x, v, spk_num_mask)
    print(s.shape)


if __name__ == "__main__":
    foo_conv_tas_visnet()
