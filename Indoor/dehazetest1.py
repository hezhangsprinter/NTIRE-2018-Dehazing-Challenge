from __future__ import print_function
import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable

from misc import *
# import models.dehaze1113 as net
import dehaze1113 as net

import pdb



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False,
  default='pix2pix',  help='')
parser.add_argument('--dataroot', required=False,
  default='', help='path to trn dataset')
parser.add_argument('--valDataroot', required=False,
  default='', help='path to val dataset')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--valBatchSize', type=int, default=1, help='input batch size')
parser.add_argument('--originalSize', type=int,
  default=1024, help='the height / width of the original input image')
parser.add_argument('--imageSize', type=int,
  default=1024, help='the height / width of the cropped input image to network')
parser.add_argument('--inputChannelSize', type=int,
  default=3, help='size of the input channels')
parser.add_argument('--outputChannelSize', type=int,
  default=3, help='size of the output channels')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=400, help='number of epochs to train for')
parser.add_argument('--lambdaGAN', type=float, default=0.01, help='lambdaGAN')
parser.add_argument('--lambdaIMG', type=float, default=1, help='lambdaIMG')
parser.add_argument('--poolSize', type=int, default=50, help='Buffer size for storing previously generated samples from G')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--exp', default='sample', help='folder to output images and model checkpoints')
parser.add_argument('--display', type=int, default=5, help='interval for displaying train-logs')
parser.add_argument('--evalIter', type=int, default=500, help='interval for evauating(generating) images from valDataroot')
opt = parser.parse_args()
print(opt)


create_exp_dir(opt.exp)
opt.manualSeed = random.randint(1, 10000)
# opt.manualSeed = 101
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)

# get dataloader
dataloader = getLoader(opt.dataset,
                       opt.dataroot,
                       opt.originalSize,
                       opt.imageSize,
                       opt.batchSize,
                       opt.workers,
                       mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                       split='train',
                       shuffle=True,
                       seed=opt.manualSeed)
opt.dataset='pix2pix_val'
# opt.dataset='pix2pix'

valDataloader = getLoader(opt.dataset,
                          opt.valDataroot,
                          opt.imageSize, #opt.originalSize,
                          opt.imageSize,
                          opt.valBatchSize,
                          opt.workers,
                          mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                          split='val',
                          shuffle=False,
                          seed=opt.manualSeed)

# get logger
trainLogger = open('%s/train.log' % opt.exp, 'w')



ngf = opt.ngf
ndf = opt.ndf
inputChannelSize = opt.inputChannelSize
outputChannelSize= opt.outputChannelSize



netG=net.Dense_rain_cvprw3()




netG.apply(weights_init)
if opt.netG != '':
  netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = net.D(inputChannelSize + outputChannelSize, ndf)
netD.apply(weights_init)
if opt.netD != '':
  netD.load_state_dict(torch.load(opt.netD))
print(netD)


netD_tran = net.D_tran(inputChannelSize + outputChannelSize, ndf)
netD_tran.apply(weights_init)

print(netD_tran)


netG.eval()
netG.train()

netD.train()
netD_tran.train()
criterionBCE = nn.BCELoss()
criterionCAE = nn.L1Loss()

target= torch.FloatTensor(opt.batchSize, outputChannelSize, opt.imageSize, opt.imageSize)
input = torch.FloatTensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)




val_target= torch.FloatTensor(opt.valBatchSize, outputChannelSize, opt.imageSize, opt.imageSize)
val_input = torch.FloatTensor(opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)
label_d = torch.FloatTensor(opt.batchSize)


target = torch.FloatTensor(opt.batchSize, outputChannelSize, opt.imageSize, opt.imageSize)
input = torch.FloatTensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)
depth = torch.FloatTensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)
ato = torch.FloatTensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)


val_target = torch.FloatTensor(opt.valBatchSize, outputChannelSize, opt.imageSize, opt.imageSize)
val_input = torch.FloatTensor(opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)
val_depth = torch.FloatTensor(opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)
val_ato = torch.FloatTensor(opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)




# NOTE: size of 2D output maps in the discriminator
sizePatchGAN = 30
real_label = 1
fake_label = 0

# image pool storing previously generated samples from G
imagePool = ImagePool(opt.poolSize)

# NOTE weight for L_cGAN and L_L1 (i.e. Eq.(4) in the paper)
lambdaGAN = opt.lambdaGAN
lambdaIMG = opt.lambdaIMG

netD.cuda()
netG.cuda()
netD_tran.cuda()
criterionBCE.cuda()
criterionCAE.cuda()


target, input, depth, ato = target.cuda(), input.cuda(), depth.cuda(), ato.cuda()
val_target, val_input, val_depth, val_ato = val_target.cuda(), val_input.cuda(), val_depth.cuda(), val_ato.cuda()

target = Variable(target, volatile=True)
input = Variable(input,volatile=True)
depth = Variable(depth,volatile=True)
ato = Variable(ato,volatile=True)



label_d = Variable(label_d.cuda())




# net_label=net2.vgg19ca()
# # net_label=net2.vgg16()
# net_label.load_state_dict(torch.load('./checkpoints_test_classificanew/netG_epoch_9.pth'))
# # net_label.load_state_dict(torch.load('./checkpoints_test_classificanew_vgg16_scratch/netG_epoch_5.pth'))
#
# net_label=net_label.cuda()


# residue_net=net.Dense_rain5()
# residue_net.load_state_dict(torch.load('./checkpoints_train_all_new_modelnewdata_nolabel/netG_epoch_0.pth'))
# residue_net=residue_net.cuda()
#
# residue_net=net.Dense_rain4()
# residue_net.load_state_dict(torch.load('./checkpoints_train_all_new_modelnewdata_heavy/netG_epoch_6.pth'))
# residue_net=residue_net.cuda()

def norm_ip(img, min, max):
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min)

    return img


def norm_range(t, range):
    if range is not None:
        norm_ip(t, range[0], range[1])
    else:
        norm_ip(t, t.min(), t.max())
    return norm_ip(t, t.min(), t.max())
import time

ganIterations = 0
for epoch in range(1):
  heavy, medium, light=200, 200, 200
  for i, data in enumerate(valDataloader, 0):
    if 1:
      t0 = time.time()

      data_val = data

      val_input_cpu, val_target_cpu, path = data_val

      val_target_cpu, val_input_cpu = val_target_cpu.float().cuda(), val_input_cpu.float().cuda()
      val_batch_output = torch.FloatTensor(val_input.size()).fill_(0)

      val_input.resize_as_(val_input_cpu).copy_(val_input_cpu)
      val_target=Variable(val_target_cpu, volatile=True)


      z=0
      label_cpu = torch.FloatTensor(opt.batchSize).fill_(z)


      label_cpu2=0

      label_d.data.fill_(label_cpu2)

      label_cpu = label_cpu.long().cuda()
      label_cpu = Variable(label_cpu)


      for idx in range(val_input.size(0)):
        single_img = val_input[idx,:,:,:].unsqueeze(0)
        val_inputv = Variable(single_img, volatile=True)

        # pdb.set_trace()
        x_hat_val = netG(val_target)

        # val_batch_output[idx,:,:,:].copy_(x_hat_val.data)
      # vutils.save_image(x_hat_val.data, './image_heavy/'+str(i)+'.jpg', normalize=True, scale_each=False,  padding=0, nrow=1)
      from PIL import Image
      resukt=torch.cat([val_inputv,x_hat_val],3)
      tensor = x_hat_val.data.cpu()
      # tensor2 = val_target.data
      # tensor = val_inputv.data.cpu()

      # grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
      #                  normalize=normalize, range=range, scale_each=scale_each)
      # resukt=x_hat_val
      # resukt=x_hat_val
      # residual=x_hat_val

      # resukt=torch.cat([val_inputv,val_target],3)

      # print(val_target.size())
      # tensor = residual.data.cpu()
      from PIL import Image

      directory = './indoor/our_cvprw_test3/'
      if not os.path.exists(directory):
          os.makedirs(directory)


      # pdb.set_trace()
      name=''.join(path)
      filename='./indoor/our_cvprw_test3/'+str(i)+'.png'

      tensor = torch.squeeze(tensor)
      tensor=norm_range(tensor, None)
      print('Patch:'+str(i))


      ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
      im = Image.fromarray(ndarr)
      im.save(filename)
      t1 = time.time()
      print('running time:'+str(t1-t0))
