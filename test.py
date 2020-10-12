#%%
import torch
import cv2 as cv
from PIL import Image
import numpy as np
from nets.pose_dla_dcn import get_pose_net
from utils.image import get_affine_transform, transform_preds
from loss import _transpose_and_gather_feat, _gather_feat
from infer import _nms, _topk

import matplotlib.pyplot as plt

multi_pose = {
'default_resolution': [512, 512], 'num_classes': 1, 
'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
'dataset': 'coco_hp', 'num_joints': 17,
'flip_idx': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], 
                [11, 12], [13, 14], [15, 16]]}
heads = {'hm': 1, 'wh': 2, 'hps': 34, 'reg': 2, 'hm_hp': 17, 'hp_offset': 2}

def pre_process(image, meta=None):
    height, width = image.shape[0:2]

    inp_height, inp_width = multi_pose['default_resolution']
    c = np.array([width / 2., height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv.resize(image, (width, height))
    inp_image = cv.warpAffine(
      resized_image, trans_input, (inp_width, inp_height),
      flags=cv.INTER_LINEAR)
    inp_image = ((inp_image / 255. - multi_pose['mean']) / multi_pose['std']).astype(np.float32)
    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    images = torch.from_numpy(images)
    meta = {'c': c, 's': s, 
            'out_height': inp_height // 4, 
            'out_width': inp_width // 4}
    return images.cuda(), meta

def _topk_channel(scores, K=40):
      batch, cat, height, width = scores.size()
      
      topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

      topk_inds = topk_inds % (height * width)
      topk_ys   = (topk_inds.true_divide(width)).int().float()
      topk_xs   = (topk_inds % width).int().float()

      return topk_scores, topk_inds, topk_ys, topk_xs

def multi_pose_decode(
    heat, wh, kps, reg=None, hm_hp=None, hp_offset=None, K=100):
  batch, cat, height, width = heat.size()
  num_joints = kps.shape[1] // 2
  # heat = torch.sigmoid(heat)
  # perform nms on heatmaps
  heat = _nms(heat)
  scores, inds, clses, ys, xs = _topk(heat, K=K)

  kps = _transpose_and_gather_feat(kps, inds)
  kps = kps.view(batch, K, num_joints * 2)
  kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)
  kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)
  if reg is not None:
    reg = _transpose_and_gather_feat(reg, inds)
    reg = reg.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
    ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
  else:
    xs = xs.view(batch, K, 1) + 0.5
    ys = ys.view(batch, K, 1) + 0.5
  wh = _transpose_and_gather_feat(wh, inds)
  wh = wh.view(batch, K, 2)
  clses  = clses.view(batch, K, 1).float()
  scores = scores.view(batch, K, 1)

  bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                      ys - wh[..., 1:2] / 2,
                      xs + wh[..., 0:1] / 2, 
                      ys + wh[..., 1:2] / 2], dim=2)
  if hm_hp is not None:
      hm_hp = _nms(hm_hp)
      thresh = 0.1
      kps = kps.view(batch, K, num_joints, 2).permute(
          0, 2, 1, 3).contiguous() # b x J x K x 2
      reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
      hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K) # b x J x K
      if hp_offset is not None:
          hp_offset = _transpose_and_gather_feat(
              hp_offset, hm_inds.view(batch, -1))
          hp_offset = hp_offset.view(batch, num_joints, K, 2)
          hm_xs = hm_xs + hp_offset[:, :, :, 0]
          hm_ys = hm_ys + hp_offset[:, :, :, 1]
      else:
          hm_xs = hm_xs + 0.5
          hm_ys = hm_ys + 0.5
        
      mask = (hm_score > thresh).float()
      hm_score = (1 - mask) * -1 + mask * hm_score
      hm_ys = (1 - mask) * (-10000) + mask * hm_ys
      hm_xs = (1 - mask) * (-10000) + mask * hm_xs
      hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
          2).expand(batch, num_joints, K, K, 2)
      dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
      min_dist, min_ind = dist.min(dim=3) # b x J x K
      hm_score = hm_score.gather(2, min_ind).unsqueeze(-1) # b x J x K x 1
      min_dist = min_dist.unsqueeze(-1)
      min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
          batch, num_joints, K, 1, 2)
      hm_kps = hm_kps.gather(3, min_ind)
      hm_kps = hm_kps.view(batch, num_joints, K, 2)
      l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
      t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
      r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
      b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
      mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
             (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
             (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
      mask = (mask > 0).float().expand(batch, num_joints, K, 2)
      kps = (1 - mask) * hm_kps + mask * kps
      kps = kps.permute(0, 2, 1, 3).contiguous().view(
          batch, K, num_joints * 2)
  detections = torch.cat([bboxes, scores, kps, clses], dim=2)
    
  return detections

def multi_pose_post_process(dets, c, s, h, w):
  # dets: batch x max_dets x 40
  # return list of 39 in image coord
  ret = []
  for i in range(dets.shape[0]):
    bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))
    pts = transform_preds(dets[i, :, 5:39].reshape(-1, 2), c[i], s[i], (w, h))
    top_preds = np.concatenate(
      [bbox.reshape(-1, 4), dets[i, :, 4:5], 
       pts.reshape(-1, 34)], axis=1).astype(np.float32).tolist()
    ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
  return ret

def load_model(model, model_path, optimizer=None, resume=False, 
               lr=None, lr_step=None):
  start_epoch = 0
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  state_dict = {}
  
  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  msg = 'If you see this, your model does not fully load the ' + \
        'pre-trained weight. Please make sure ' + \
        'you have correctly specified --arch xxx ' + \
        'or set the correct --num_classes for your own dataset.'
  for k in state_dict:
    if k in model_state_dict:
      if state_dict[k].shape != model_state_dict[k].shape:
        print('Skip loading parameter {}, required shape{}, '\
              'loaded shape{}. {}'.format(
          k, model_state_dict[k].shape, state_dict[k].shape, msg))
        state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k) + msg)
  for k in model_state_dict:
    if not (k in state_dict):
      print('No param {}.'.format(k) + msg)
      state_dict[k] = model_state_dict[k]
  model.load_state_dict(state_dict, strict=False)

  # resume optimizer parameters
  if optimizer is not None and resume:
    if 'optimizer' in checkpoint:
      optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
      start_lr = lr
      for step in lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print('Resumed optimizer with start lr', start_lr)
    else:
      print('No optimizer parameters in checkpoint.')
  if optimizer is not None:
    return model, optimizer, start_epoch
  else:
    return model

if __name__ == '__main__':
  net = get_pose_net(34, heads).cuda()
  # net.load_state_dict({k.replace('module.',''):v 
  #         for k,v in torch.load('multi_pose_dla_3x.pth').items()})
  load_model(net, 'multi_pose_dla_3x.pth')
  net.eval()
  img = cv.imread('cxk.jpeg')
  x, meta = pre_process(img, 1)
  output = net(x)[0]

  with torch.no_grad():
      output['hm'].sigmoid_()
      output['hm_hp'].sigmoid_()

      reg = output['reg']
      hm_hp = output['hm_hp']
      hp_offset = output['hp_offset']

      dets = multi_pose_decode(
              output['hm'], output['wh'], output['hps'],
              reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=100)

  dets = multi_pose_post_process(
      dets.cpu().numpy(), [meta['c']], [meta['s']],
      meta['out_height'], meta['out_width'])
  result = np.array(dets[0][1], dtype=np.float32).reshape(-1, 39)

  for s in result:
    if s[4] > 0.5:
      # r = np.random.randint(50,200)
      # g = np.random.randint(50,200)
      # b = np.random.randint(50,200)
      p = lambda i :(s[5+i*2], s[5+2*i+1])
      l = lambda a,b:cv.line(img, p(a), p(b), (0,0,255),2)
      for i in range(17):
        cv.circle(img, (s[5+i*2], s[5+2*i+1]), 3, (0,0,255), -1)
        # cv.putText(img, str(i), (s[5+i*2], s[5+2*i+1]), 0, 0.3, (0,0,255))
      l(0,1)
      l(0,2)
      l(1,3)
      l(2,4)
      l(5,6)
      l(5,7)
      l(7,9)
      l(6,8)
      l(8,10)
      l(11,12)
      l(11,13)
      l(13,15)
      l(12,14)
      l(14,16)

      neck = (int((p(5)[0]+p(6)[0])*0.5), int((p(5)[1]+p(6)[1])*0.5))
      belly = (int((p(11)[0]+p(12)[0])*0.5), int((p(12)[1]+p(12)[1])*0.5))
      cv.line(img, p(0), neck, (0,0,255),2)
      cv.line(img, neck, belly, (0,0,255),2)

  cv.imwrite('output.jpg', img)
""" 
0:   nose
1,2: eyes
3,4: ears
5,6: shoulders
7,8: elbows
9,10:wrists
11,12:hips
13,14:knees
15,16:ankles
from right to left
"""