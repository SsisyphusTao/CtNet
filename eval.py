import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from utils import get_dataset, val_collate
from nets import get_pose_net
from infer import ctdet_decode, ctdet_post_process, pre_process, merge_outputs
from test import load_model


import os.path as osp
import argparse
import numpy as np
import cv2
import json
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model', default='checkpoints/ctnet_dla_140_1684.pth',
                    type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.3, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default='/ai/ailab/Share/TaoData/',
                    help='Location of VOC root directory')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')

args = parser.parse_args()

def inference():
    # load data
    Dataset = get_dataset()(args.voc_root, 'val')
    val_loader = data.DataLoader(
      Dataset, 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True,
      collate_fn=val_collate
  )
    # load net
    heads = {'hm': Dataset.num_classes,
             'wh': 2,
             'reg': 2}
    net = get_pose_net(34, heads)
    net.load_state_dict({k.replace('module.',''):v 
                        for k,v in torch.load(args.trained_model).items()})
    # load_model(net, 'ctdet_coco_dla_2x.pth')
    net.eval()
    net = nn.DataParallel(net.cuda(), device_ids=[0])
    print('Finished loading model!')
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    results = {}
    with tqdm(total=len(val_loader)) as bar:
        for i in val_loader:
            preds = net(i['input'])
            output = preds[0]
            reg = output['reg']
            dets = ctdet_decode(
            output['hm'].sigmoid_(), output['wh'], reg=reg)
            dets = dets.detach().cpu().numpy()
            dets = dets.reshape(1, -1, dets.shape[2])
            dets = ctdet_post_process(
                    dets.copy(), [i['meta'][0]['c']], [i['meta'][0]['s']],
                    i['meta'][0]['out_height'], i['meta'][0]['out_width'], 80)
            for j in range(1, 80 + 1):
                dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            results[int(i['meta'][0]['img_id'])] = merge_outputs([dets[0]])
            bar.update(1)
    Dataset.save_results(results, '.')

if __name__ == '__main__':
    inference()
    ann_file = '/ai/ailab/Share/TaoData/coco/annotations/instances_val2017.json'
    coco = COCO(ann_file)
    cocoeval = coco.loadRes('results.json')

    imgIds = sorted(coco.getImgIds())

    cocoEval = COCOeval(coco,cocoeval,'bbox')
    cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()