import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
        
        self.occ_sigmoid = nn.Sigmoid()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1
    
    def initialize_occ(self, img):
        """ Two channels for occlusions """
        N, C, H, W = img.shape
        occ_false = torch.zeros((N, 2, H//8, W//8)).float().to(img.device)
        occ_true = torch.zeros((N, 2, H//8, W//8)).float().to(img.device)

        return occ_false, occ_true

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2) # shape=(N, 1, 9, 8, 8, H, W)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2) # shape=(N, 2, 8, 8, H, W)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3) # shape=(N, 2, H, 8, W, 8) this is correct!
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])        
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)
        occ_false, occ_true = self.initialize_occ(image1)
        #print('occ_true shape', occ_true.shape, 'dtype', occ_true.dtype)
        # occ_true shape torch.Size([1, 2, 46, 96]) dtype torch.float32

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        occ_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            # print('flow shape', coords0.shape, 'dtype', coords0.dtype)
            with autocast(enabled=self.args.mixed_precision):
                flow_occ = torch.cat([flow, occ_true], dim=1)
                net, up_mask, up_mask_occ, delta_flow, delta_occ = \
                    self.update_block(net, inp, corr, flow_occ, upsample=(itr==iters-1))

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow
            occ_true = occ_true + delta_occ

            # upsample predictions
            # HOW DOES THIS WORK?
            if itr == iters - 1:
                if up_mask is None:
                    flow_up = upflow8(coords1 - coords0)
                    #occ_up = upflow8(occ_true)
                else:
                    flow_up = self.upsample_flow(coords1 - coords0, up_mask)
                    #occ_up = self.upsample_flow(occ_true, up_mask_occ)

            if itr == iters - 1:
                N, _, H, W = up_mask_occ.shape
                occ_up = up_mask_occ.view(N, 1, 8, 8, H, W)
                occ_up = occ_up.permute(0, 1, 4, 2, 5, 3) # shape=(N, 1, H, 8, W, 8)
                occ_up = occ_up.reshape(N, 1, 8*H, 8*W)


                #occ_up = occ_up[:, 0:1] # second layer goes to trash. Try softmax next time. Then logsoftmax
                occ_up = self.occ_sigmoid(occ_up)
            
            if itr == iters - 1:
                flow_predictions.append(flow_up)
                occ_predictions.append(occ_up)

        #if test_mode:
        #    return coords1 - coords0, flow_up
            
        return flow_predictions, occ_predictions
