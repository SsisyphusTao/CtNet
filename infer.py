import torch
import cv2 as cv
import numpy as np
from nets import get_pose_net
from utils.image import get_affine_transform, transform_preds
from loss import _transpose_and_gather_feat, _gather_feat

heads = {'hm': 20,
            'wh': 2,
            'reg': 2}
mean = np.array([0.485, 0.456, 0.406],
                dtype=np.float32).reshape(1, 1, 3)
std  = np.array([0.229, 0.224, 0.225],
                dtype=np.float32).reshape(1, 1, 3)

net = get_pose_net(34, heads).cuda()
net.load_state_dict({k.replace('module.',''):v 
        for k,v in torch.load('checkpoints/ctnet_dla_20_396461.pth').items()})

img = cv.imread('000040.jpg')
height, width = img.shape[0:2]
inp_height = 384
inp_width = 384
print(inp_height, inp_width)
c = np.array([width // 2, height // 2], dtype=np.float32)
s = np.array([inp_width, inp_height], dtype=np.float32)

meta = {'c': c, 's': s, 
        'out_height': inp_height // 4, 
        'out_width': inp_width // 4}

trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
resized_image = cv.resize(img, (width, height))
inp_image = cv.warpAffine(
  resized_image, trans_input, (inp_width, inp_height),
  flags=cv.INTER_LINEAR)
inp_image = ((inp_image / 255. - mean) / std).astype(np.float32)
img = torch.from_numpy(inp_image).permute(2, 0, 1)
img = img.unsqueeze(0).cuda()
print(img.size())
output = net(img)
hm = output[0]['hm'].sigmoid_()#.detach().cpu().numpy()[0]
wh = output[0]['wh']
reg = output[0]['reg']

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = torch.nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
      
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
      
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
      reg = _transpose_and_gather_feat(reg, inds)
      reg = reg.view(batch, K, 2)
      xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
      ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
      xs = xs.view(batch, K, 1) + 0.5
      ys = ys.view(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
      wh = wh.view(batch, K, cat, 2)
      clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
      wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
      wh = wh.view(batch, K, 2)
    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2, 
                        ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
    print(torch.cat([scores, clses], dim=2)[0][0:10])
    return detections

def ctdet_post_process(dets, c, s, h, w, num_classes):
  # dets: batch x max_dets x dim
  # return 1-based class det dict
  ret = []
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(
          dets[i, :, 0:2], c[i], s[i], (w, h))
    dets[i, :, 2:4] = transform_preds(
          dets[i, :, 2:4], c[i], s[i], (w, h))
    classes = dets[i, :, -1]
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :4].astype(np.float32),
        dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)
  return ret

def merge_outputs(detections):
    results = {}
    for j in range(1, 20 + 1):
      results[j] = np.concatenate(
        [detection[j] for detection in detections], axis=0).astype(np.float32)
    scores = np.hstack(
      [results[j][:, 4] for j in range(1, 20 + 1)])
    if len(scores) > 50:
      kth = len(scores) - 50
      thresh = np.partition(scores, kth)[kth]
      for j in range(1, 20 + 1):
        keep_inds = (results[j][:, 4] >= thresh)
        results[j] = results[j][keep_inds]
    return results

dets = ctdet_decode(hm, wh, reg)
dets = dets.detach().cpu().numpy()
dets = dets.reshape(1, -1, dets.shape[2])
dets = ctdet_post_process(
      dets.copy(), [meta['c']], [meta['s']],
      meta['out_height'], meta['out_width'], 20)
for j in range(1, 20 + 1):
    dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
dets = merge_outputs([dets[0]])
# print(dets.keys())
# print(dets[0][1])