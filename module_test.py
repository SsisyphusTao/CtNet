import torch
from nets import get_pose_net

heads = {'hm': 21,
         'wh': 2 * 21,
         'reg': 2}
net = get_pose_net(50, heads)

t = torch.randn(2,3,300,300)

net(t)