from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from raft import RAFT
import evaluate
import datasets

from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self, enabled):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

from utils.checkpoints import checkpoint_load_path, checkpoint_save_path
from utils.checkpoints import save_model_txt, load_model_txt, convert_to_txt
from utils.logfile import logfile
from pathlib import Path
import time
from time import sleep

from skimage import io

# exclude extremely large displacements
MAX_FLOW = 400

SUM_FREQ = 100
VAL_FREQ = 5000

def arr_info(img):
    logfile.log(img.shape, img.dtype, img.min(), img.max())


# Must check this (from IRR-PWC)
# def f1_score(y_true, y_pred):
#     return fbeta_score(y_true, y_pred, 1)

# def fbeta_score(y_true, y_pred, beta, eps=1e-8):
#     beta2 = beta ** 2

#     y_pred = y_pred.float()
#     y_true = y_true.float()

#     true_positive = (y_pred * y_true).sum(dim=2).sum(dim=2).sum(dim=1)
#     precision = true_positive / (y_pred.sum(dim=2).sum(dim=2).sum(dim=1) + eps)
#     recall = true_positive / (y_true.sum(dim=2).sum(dim=2) + eps)

#     return torch.mean(precision * recall / (precision * beta2 + recall + eps) * (1 + beta2))

def f1_score_bal_loss(y_pred, y_true):
    # y_pred.shape = [batch, 1, height, width]
    eps = 1e-8

    tp = -(y_true * torch.log(y_pred + eps)).sum(dim=2).sum(dim=2).sum(dim=1)
    fn = -((1 - y_true) * torch.log((1 - y_pred) + eps)).sum(dim=2).sum(dim=2).sum(dim=1)

    denom_tp = y_true.sum(dim=2).sum(dim=2).sum(dim=1) + y_pred.sum(dim=2).sum(dim=2).sum(dim=1) + eps
    denom_fn = (1 - y_true).sum(dim=2).sum(dim=2).sum(dim=1) + (1 - y_pred).sum(dim=2).sum(dim=2).sum(dim=1) + eps

    return ((tp / denom_tp).sum() + (fn / denom_fn).sum())

def sequence_loss(flow_preds, flow_gt, occ_preds, occ_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """
    
    n_predictions = len(flow_preds)
    flow_loss = 0.0
    occ_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        # print('seq flow', flow_preds[i].shape, flow_gt.shape)
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()
        
        # print('seq occ', occ_preds[i].shape, occ_gt.shape)
        #i_loss = (occ_preds[i] - occ_gt).abs()
        occ_pred = occ_preds[i]
        i_loss = f1_score_bal_loss(occ_pred, occ_gt)
        #occ_loss += i_weight * (valid[:, None] * i_loss).mean()
        occ_loss += i_weight * i_loss

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    

        # precision=tp/(tp+fp), recall=tp/(tp+fn)
        # occ_pred_bin = occ_preds[-1] > 0.5
        # occ_gt_bin = occ_gt > 0.5
        # intersect = torch.sum(occ_pred_bin * occ_gt_bin)
        # pred = torch.sum(occ_pred_bin)
        # gt = torch.sum(occ_gt_bin)
        
        
        # f1 = 0
        # if intersect != 0 and pred != 0 and gt != 0:
        #     prec = intersect / pred
        #     recall = intersect / gt
        #     f1 = 2 / (1 / prec + 1 / recall)
    f1 = torch.sum((occ_preds[-1] - occ_gt)**2, dim=1).sqrt()
    f1 = f1.view(-1)[valid.view(-1)]
    
    f_loss = flow_loss.detach()
    o_loss = occ_loss.detach()
    if (f_loss.data > o_loss.data).numpy:
       f_l_w = 1
       o_l_w = f_loss / o_loss
    else:
       f_l_w = o_loss / f_loss
       o_l_w = 1

    total_loss = (flow_loss * f_l_w + occ_loss * o_l_w) / (2 * flow_gt.size(0))
    #total_loss = (flow_loss + occ_loss * 0.1) / (2 * flow_gt.size(0))

    metrics = {
        'epe': epe.mean().item(),
        'flow_loss': f_loss.data,
        'occ_loss': o_loss.data,
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return total_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler, optimizer, path='runs/logbook', total_steps=0):
        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.running_loss = {}
        self.writer = None
        self.writer_path = path

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(
            self.total_steps+1,
            self.optimizer.param_groups[0]["lr"])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        #print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter(self.writer_path)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(self.writer_path)
        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()

def contains_nans(tensor):
    return torch.isnan(tensor).any()

def train(args):
    DEVICE = 'cuda'
    
    if args.cpu:
        DEVICE = 'cpu'
    #model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    model = nn.DataParallel(RAFT(args))
    logfile.log("Parameter Count: %d" % count_parameters(model))

    # Load model, optimizer, scheduler
    total_steps = 0
    optimizer = None
    scheduler = None
    batch_start = 0
    is_model_loaded = False

    if args.restore_ckpt is not None:
        path = checkpoint_load_path(args.restore_ckpt)
        if Path(path).exists():
            if 0:
                checkpoint = torch.load(path, map_location=torch.device(DEVICE))
                model.load_state_dict(checkpoint, strict=False)
            if 1:
                checkpoint = torch.load(path, map_location=torch.device(DEVICE))
                if 'model' in checkpoint: # New format, full save
                    total_steps = checkpoint['total_steps']
                    model = checkpoint['model']
                    optimizer = checkpoint['optimizer']
                    scheduler = checkpoint['scheduler']
                    batch_start = checkpoint['batch_start']
                    is_model_loaded = True
                    logfile.log('Continue from', total_steps, 'step')
                else: # Standard format
                    model.load_state_dict(checkpoint, strict=False)
                    logfile.log('Loaded model without steps')

                if 0: # Save only weights without state
                    PATH = 'checkpoints/01.pth'
                    torch.save(model.state_dict(), PATH)
                    exit()
            if 0:
                load_model_txt(model, path)
                PATH = 'checkpoints/01.pth'
                torch.save(model.state_dict(), PATH)
                exit()
    model.to(DEVICE)

    if not is_model_loaded:
        model.train()
        optimizer, scheduler = fetch_optimizer(args, model)

        if args.stage != 'chairs':
            model.module.freeze_bn()

        
    train_loader = datasets.fetch_dataloader(args)
    logger = Logger(model, scheduler, optimizer,
        path='{}/logbook'.format(args.output), total_steps=total_steps)
    scaler = GradScaler(enabled=args.mixed_precision)

    if 0:
        for i in range(10):
            logger.push({'epe': i})

        logger.writer.add_text('key', 'value')
        exit(0)
    
    VAL_FREQ = 40000
    STEPS = 3000

    session_steps = 0
    should_keep_training = True
    while should_keep_training:

        logfile.log('Start training from batch number %d' % batch_start)
        for i_batch, data_blob in enumerate(train_loader):
            image_index = data_blob[-1]
            data_blob = data_blob[:len(data_blob) - 1]
            logfile.log('Batch number: %d' % i_batch, 'Image index:', image_index)
            #if i_batch < batch_start: # Continue from saved batch number
            #    continue

            optimizer.zero_grad()
            image1, image2, flow, occ, valid = [x.to(DEVICE) for x in data_blob]

            # if contains nan, then loss will be undefined
            if True in [contains_nans(x) for x in [image1, image2, flow, occ]]:
                logfile.log('Skip batch, NaN found!')
                continue

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).to(DEVICE)).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).to(DEVICE)).clamp(0.0, 255.0)

            flow_predictions, occ_predictions = model(image1, image2, iters=args.iters)

            loss, metrics = sequence_loss(flow_predictions, flow, 
                                          occ_predictions, occ, valid, args.gamma)
            #logfile.log(loss, metrics)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)
            
            if total_steps % SUM_FREQ == SUM_FREQ - 1:
                logfile.log('Saving. Step', total_steps)
                PATH = checkpoint_save_path('checkpoints/%s.pth' % args.name)
                checkpoint = {
                    'total_steps': total_steps,
                    'model': model,
                    'optimizer': optimizer,
                    'scheduler': scheduler,
                    'batch_start': i_batch
                }
                torch.save(checkpoint, PATH)
                checkpoint_save_path(PATH, save_json=True)

                PATH = 'checkpoints/01.pth'
                torch.save(model.state_dict(), PATH)

                # save example images
                i1, i2, occpred, occgt = [x.detach()[0].permute(1,2,0).cpu().numpy() for x in [image1, image2, occ_predictions[-1], occ]]
                # arr_info(i1) # (368, 496, 3) float32 0.0 255.0
                # arr_info(occpred) # (368, 496, 1) float32 0.0 8.83106e-05
                # arr_info(occgt) # (368, 496, 1) float32 0.0 1.0
                
                io.imsave('{}/{}_img1.png'.format(args.output, i_batch), i1)
                io.imsave('{}/{}_img2.png'.format(args.output, i_batch), i2)
                io.imsave('{}/{}_occpred.png'.format(args.output, i_batch), occpred)
                io.imsave('{}/{}_occgt.png'.format(args.output, i_batch), occgt)
            
            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                logfile.log('Validation. Step', total_steps)
                results = {}
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        results.update(evaluate.validate_chairs(model.module))
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel_occ(model.module, args.output))
                    elif val_dataset == 'kitti': # never used
                        results.update(evaluate.validate_kitti(model.module))

                logger.write_dict(results)
                
                model.train()
                if args.stage != 'chairs':
                    model.module.freeze_bn()
            
            total_steps += 1

            session_steps += 1
            if session_steps > STEPS:
                return

            if total_steps > args.num_steps:
                should_keep_training = False
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--output', help="path for saving output")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--cpu', action='store_true', help='use only cpu')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()

    if not logfile.logfile:
        logfile.set_logfile('{}/stdout.log'.format(args.output))

    rand_seed = int(time.time() * 1000) % (2 ** 30)
    torch.manual_seed(rand_seed)
    np.random.seed(rand_seed)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)