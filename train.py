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

from utils.checkpoints import checkpoint_load_path, checkpoint_save_path, save_model_txt, load_model_txt, convert_to_txt
from pathlib import Path
import numpy as np
from time import sleep

# exclude extremly large displacements
MAX_FLOW = 400

SUM_FREQ = 1
VAL_FREQ = 5000


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler, optimizer):
        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(
            self.total_steps+1,
            self.optimizer.param_groups[0]["lr"])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

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
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()

def train(args):
    #model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    model = nn.DataParallel(RAFT(args))
    print("Parameter Count: %d" % count_parameters(model))

    total_steps = 0
    optimizer = None
    scheduler = None
    logger = None
    is_model_loaded = False

    if args.restore_ckpt is not None:
        path = checkpoint_load_path(args.restore_ckpt)
        if Path(path).exists():
            if 0:
                checkpoint = torch.load(path, map_location=torch.device('cpu'))
                model.load_state_dict(checkpoint, strict=False)
            if 1:
                checkpoint = torch.load(path)
                if 'model' in checkpoint: # New format, full save
                    total_steps = checkpoint['total_steps']
                    model = checkpoint['model']
                    optimizer = checkpoint['optimizer']
                    scheduler = checkpoint['scheduler']
                    logger = checkpoint['logger']
                    is_model_loaded = True
                    print('Continue from', total_steps, 'step')
                else: # Standard format
                    model.load_state_dict(checkpoint, strict=False)
                    print('Loaded model without steps')
            if 0:
                load_model_txt(model, path)
                PATH = 'checkpoints/01.pth'
                torch.save(model.state_dict(), PATH)

    model.cuda()
    model.train()
    train_loader = datasets.fetch_dataloader(args)

    if not is_model_loaded:
        total_steps = 0
        optimizer, scheduler = fetch_optimizer(args, model)
        logger = Logger(model, scheduler, optimizer)

        if args.stage != 'chairs':
            model.module.freeze_bn()

    PATH = checkpoint_save_path('checkpoints/%s.pth' % args.name)
    checkpoint = {
        'total_steps': total_steps,
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'logger': logger
    }
    torch.save(checkpoint, PATH)
    checkpoint_save_path(PATH, save_json=True)

    scaler = GradScaler(enabled=args.mixed_precision)

    for i in range(1000):
        logger.push({'epe': 10})
        sleep(4)
    

    SAVE_FREQ = 50
    VAL_FREQ = 5000
    MAX_STEP = 500
    add_noise = True

    should_keep_training = True
    while should_keep_training:

        print('Start training')
        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data_blob]

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            flow_predictions = model(image1, image2, iters=args.iters)            

            loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                results = {}
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        results.update(evaluate.validate_chairs(model.module))
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel(model.module))
                    elif val_dataset == 'kitti':
                        results.update(evaluate.validate_kitti(model.module))

                logger.write_dict(results)
                
                model.train()
                if args.stage != 'chairs':
                    model.module.freeze_bn()
            
            if total_steps % SAVE_FREQ == SAVE_FREQ - 1:
                PATH = checkpoint_save_path('checkpoints/%s.pth' % args.name)
                checkpoint = {
                    'total_steps': total_steps,
                    'model': model,
                    'optimizer': optimizer,
                    'scheduler': scheduler,
                    'logger': logger
                }
                torch.save(checkpoint, PATH)
                checkpoint_save_path(PATH, save_json=True)
            
            if total_steps > MAX_STEP:
                return
            
            total_steps += 1
            print('Step', total_steps)

            if total_steps > args.num_steps:
                should_keep_training = False
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)