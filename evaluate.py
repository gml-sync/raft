import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import datasets
from utils import flow_viz
from utils import frame_utils

from raft import RAFT
from utils.utils import InputPadder, forward_interpolate

from time import time
from utils.f1fast_test import F1Accumulator
from utils.logfile import logfile
from pathlib import Path
from skimage import io

def arr_info(img):
    logfile.log(img.shape, img.dtype, img.min(), img.max())

@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        
        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None
            
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            
            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []
    results = {}

    val_dataset = datasets.FlyingChairsOcc(split='validation')
    accumulator = F1Accumulator()
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, occ_gt, _, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_seq, occ_seq = model(image1, image2, iters=iters, test_mode=True) # b c h w
        flow = flow_seq[-1][0] # last prediction in sequence + first item in batch
        occ = occ_seq[-1][0]
        flow = padder.unpad(flow).cpu() # c h w
        occ = padder.unpad(occ).cpu()
        occ_gt = occ_gt[0].numpy() > 0.5 # c h w -> h w, float -> bool
        # shape=(h, w) bool
        occ = occ[0].numpy()
        # shape=(h, w) float32 in [0, 1]

        accumulator.add(occ_gt, occ)
        
        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        np_epe = epe.view(-1).numpy()
        epe_list.append(np_epe)
        #logfile.log(val_id, 'mean epe', np.mean(np_epe))

    epe = np.mean(np.concatenate(epe_list))
    results['chairs'] = epe
    logfile.log("Validation Chairs EPE: %f" % epe)

    # Max f-score and figure drawing
    precision, recall, thresholds = accumulator.get_result()
    
    max_f1 = 0
    pr = rc = th = 0
    for i, j in zip(range(len(precision)), thresholds):
        f1 = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        if f1 > max_f1:
            max_f1 = f1
            pr = precision[i]
            rc = recall[i]
            th = j

    logfile.log(max_f1, pr, rc, th)
    results['chairs_occ'] = max_f1

    return results

@torch.no_grad()
def validate_sintel(model, out_path, iters=32):
    """ Peform validation using the Sintel (train) split """
    save_dir = Path('{}/sintel_val'.format(out_path)).resolve()
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        #val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        val_dataset = datasets.MpiSintelOcc(split='training', dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            #image1, image2, flow_gt, _ = val_dataset[val_id]
            image1, image2, flow_gt, occ_gt, _, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()
            if val_id == 0:
                logfile.log(image1.size())

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu() # b c h w -> c h w

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            np_epe = epe.view(-1).numpy()
            epe_list.append(np_epe)
            logfile.log(val_id, 'mean epe', np.mean(np_epe))
            
            path = save_dir / dstype
            path.mkdir(parents=True, exist_ok=True)
            
            f = flow.permute(1,2,0).numpy()
            flow_img = flow_viz.flow_to_image(f)
            io.imsave(path / '{:04d}_flow.png'.format(val_id), flow_img)
            #io.imsave(occ_path / (str(val_id) + '.png'), occ)
            #io.imsave(occ_path / (str(val_id) + '_optimum.png'), occ > 0.36)
            #io.imsave(occ_path / (str(val_id) + '_gt.png'), occ_gt)

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)



    return results

@torch.no_grad()
def validate_sintel_occ(model, out_path, iters=32):
    """ Peform validation using the Sintel (train) split """
    save_dir = Path('{}/sintel_val'.format(out_path)).resolve()
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        #accumulator = F1Accumulator()
        val_dataset = datasets.MpiSintelOcc(split='training', dstype=dstype)
        epe_list = []
        for val_id in range(len(val_dataset)):
            accumulator = F1Accumulator()
            image1, image2, flow_gt, occ_gt, _, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            t = time()
            flow_seq, occ_seq = model(image1, image2, iters=iters, test_mode=True) # b c h w
            logfile.log('Elapsed:', time() - t)
            flow = flow_seq[-1][0] # last prediction in sequence + first item in batch
            occ = occ_seq[-1][0]
            flow = padder.unpad(flow).cpu() # c h w
            occ = padder.unpad(occ).cpu()
            occ_gt = occ_gt[0].numpy() > 0.5 # c h w -> h w, float -> bool
            # shape=(h, w) bool
            occ = occ[0].numpy()
            # shape=(h, w) float32 in [0, 1]

            accumulator.add(occ_gt, occ)
            
            path = save_dir / dstype
            path.mkdir(parents=True, exist_ok=True)
            path_occ_gt = path / 'occ_gt'
            path_occ_gt.mkdir(parents=True, exist_ok=True)
            #io.imsave(path_occ_gt / '{:04d}.png'.format(val_id), occ_gt)
            io.imsave(path / '{:04d}.png'.format(val_id), occ)
            
            f = flow.permute(1,2,0).numpy()
            flow_img = flow_viz.flow_to_image(f)
            io.imsave(path / '{:04d}_flow.png'.format(val_id), flow_img)

            f = flow_gt.permute(1,2,0).cpu().numpy()
            flow_img = flow_viz.flow_to_image(f)
            #io.imsave(path / '{:04d}_flow_gt.png'.format(val_id), flow_img)
            
            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            np_epe = epe.view(-1).numpy()
            epe_list.append(np_epe)
            #logfile.log(val_id, 'mean epe', np.mean(np_epe))

            logfile.log(accumulator.get_max())
            
            

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        logfile.log("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

        # Max f-score and figure drawing
        precision, recall, thresholds = accumulator.get_result()

        max_f1 = 0
        pr = rc = th = 0
        for i, j in zip(range(len(precision)), thresholds):
            f1 = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
            if f1 > max_f1:
                max_f1 = f1
                pr = precision[i]
                rc = recall[i]
                th = j

        logfile.log(max_f1, pr, rc, th)
        results[dstype + '_occ'] = max_f1

        plt.scatter(rc, pr, s=100)
        plt.step(recall, precision, label='RAFT Fscore={0:0.4f}'.format(max_f1), linewidth=2)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve (sintel)')
        plt.legend(loc='lower left', prop={'size': 14})
        plt.tight_layout()
        plt.savefig('naive3_sintel.png')
        #plt.show()

    return results


@torch.no_grad()
def validate_kitti(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    logfile.log("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/raft-things.pth', help="restore checkpoint")
    parser.add_argument('--output', help="path for saving output")
    parser.add_argument('--dataset', default='sintel', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    
    if not logfile.logfile:
        logfile.set_logfile('{}/stdout.log'.format(args.output))

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    # create_sintel_submission(model.module, warm_start=True)
    # create_kitti_submission(model.module)

    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module)

        elif args.dataset == 'sintel':
            validate_sintel_occ(model.module, args.output)
            #validate_sintel(model.module, args.output)
            

        elif args.dataset == 'kitti':
            validate_kitti(model.module)


