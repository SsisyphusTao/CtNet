from utils import creat_dataset
from losses 
import torch
from torch import nn
import torch.optim as optim
import torch.utils.data as data
from nets import Ctnet_mbv2
import argparse
import time
import os.path as osp

parser = argparse.ArgumentParser(
    description='CenterNet task')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset_root', default=osp.join(osp.expanduser('~'),'data','VOCdevkit'),
                    help='Path of training set')
parser.add_argument('--batch_size', default=64, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--epochs', default=200, type=int,
                    help='the number of training epochs')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=16, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    help='initial learning rate')
parser.add_argument('--save_folder', default='checkpoints/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def train_one_epoch(loader, net, criterion, optimizer, epoch):
    loss_amount = 0
    # load train data
    for iteration, batch in enumerate(loader):
        images, targets = batch
        images = images.cuda()
        # forward
        t0 = time.time()
        predictions = net(images)
        # backprop
        optimizer.zero_grad()
        
        #TODO:criterion

        loss.backward()
        optimizer.step()
        t1 = time.time()
        loss_amount += loss_bare.cpu().detach().numpy()
        if iteration % 10 == 0 and not iteration == 0:
            print('Loss: %.6f | iter: %03d | timer: %.4f sec. | epoch: %d' %
                    (loss_amount/iteration, iteration, t1-t0, epoch))
    print('Loss: %.6f --------------------------------------------' % (loss_amount/iteration))
    return '_%d' % (loss_amount/iteration*1000)

def train():
    start_time = time.time()
    net = Ctnet_mbv2()
    if args.resume:
        missing, unexpected = net.load_state_dict({k.replace('module.',''):v 
        for k,v in torch.load(args.resume).items()}, strict=False)
        if missing:
            print('Missing:', missing)
        if unexpected:
            print('Unexpected:', unexpected)
    net.train()
    net = nn.DataParallel(net.cuda(), device_ids=[0,1,2,3])
    torch.backends.cudnn.benchmark = True

    dataset = creat_dataset()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.lr
    # adjust_learning_rate = optim.lr_scheduler.MultiStepLR(optimizer, [120, 180], 0.1, args.start_iter)
    adjust_learning_rate = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, args.start_iter)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False)

    print('Loading the dataset...')
    print('Training CenterNet on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True,
                                  pin_memory=True)

    # create batch iterator
    for iteration in range(args.start_iter + 1, args.epochs):
        loss = train_one_epoch(data_loader, net, criterion, optimizer, iteration)
        adjust_learning_rate.step()
        if (not (iteration-args.start_iter) == 0 and iteration % 10 == 0):
            print('Saving state, iter:', iteration)
            torch.save(net.state_dict(), args.save_folder + 'ctnet_mbv2_' +
                       repr(iteration) + loss + '.pth')
    torch.save(net.state_dict(),
                args.save_folder + 'ctnet_mbv2_end' + loss + '.pth')
    end_time=time.time()
    print('%d' % (time.gmtime(end_time).tm_yday-time.gmtime(start_time).tm_yday), time.strftime('%H:%M:%S', time.gmtime(end_time-start_time)))

if __name__ == '__main__':
    train()