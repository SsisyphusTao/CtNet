import torch
import cv2 as cv
import numpy as np
from nets import get_pose_net

heads = {'hm': 20,
            'wh': 2,
            'reg': 2}
mean = np.array([0.485, 0.456, 0.406],
                dtype=np.float32).reshape(1, 1, 3)
std  = np.array([0.229, 0.224, 0.225],
                dtype=np.float32).reshape(1, 1, 3)

net = get_pose_net(50, heads).cuda()
net.load_state_dict({k.replace('module.',''):v 
        for k,v in torch.load('check/checkctnet_res18_end_661.pth').items()})

img = cv.imread('000015.jpg')
img = show = cv.resize(img, (384,384))
# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = img.astype(np.float32)
img /= 255.
img = (img - mean) / std
img = torch.from_numpy(img).permute(2, 0, 1)
img = img.unsqueeze(0).cuda()

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

output = net(img)
hm = _sigmoid(output[0]['hm']).detach().cpu().numpy()[0]
wh = output[0]['wh'].detach().cpu().numpy()[0]
reg = output[0]['reg'].detach().cpu().numpy()[0]

i, ix, iy = np.where(hm==np.max(hm))

w = wh[0][ix, iy]
h = wh[1][ix, iy]
x = ix[0] + reg[0][ix,iy]
y = iy[0] + reg[1][ix,iy]
print(np.max(hm), i[0])
cv.rectangle(show, ((x-w/2)*4, (y-h/2)*4), ((x+w/2)*4, (y+w/2)*4), 255)
cv.imwrite('output.jpg', show)