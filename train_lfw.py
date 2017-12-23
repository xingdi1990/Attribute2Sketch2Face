from __future__ import print_function, division
import argparse
from data.data_loader import CreateDataLoader
from models.models import create_model
import os
import time
import torch
from random import shuffle
import numpy as np
from util.visualizer import Visualizer

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='attr_sketch_face',  help='name of the model to be used: stackgan, attr2img')
parser.add_argument('--gpu_ids', type=str , default= '0' ,  help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--phase', type=str, default='train',   help='train, val, test, etc')
parser.add_argument('--name', type=str, default='lfw_attr2sketch2face', help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--loadSize', type=int, default=64, help='scale images to this size')
parser.add_argument('--fineSize', type=int, default=64, help='then crop to this size')
parser.add_argument('--attrB_dim', type=int, default=17, help='# of input attribute channels')
parser.add_argument('--attrA_dim', type=int, default=6, help='# of input attribute channels')
parser.add_argument('--sketch_nc', type=int, default=1, help='# of output image channels')
parser.add_argument('--image_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--nt', type=int, default=256, help='#  of dim for text features ')
parser.add_argument('--nz', type=int, default=1024, help='#  of dim for Z')
parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='# of descriminator filters in first conv layer')
parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization')
parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--niter', type=int, default=1, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=1,help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lambda_A', type=float, default=100.0, help='weight for L1 loss')
parser.add_argument('--display_winsize', type=int, default=64,  help='display window size')
parser.add_argument('--display_single_pane_ncols', type=int, default=0,
                         help='if positive, display all images in a single visdom web panel with certain number of images per row.')
parser.add_argument('--display_id', type=int, default=100, help='window id of the web display')
parser.add_argument('--display_port', type=int, default=8099, help='visdom port of the web display')
parser.add_argument('--display_freq', type=int, default=100,help='frequency of showing training results on screen')
parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
parser.add_argument('--save_latest_freq', type=int, default=3100, help='frequency of saving the latest results')
parser.add_argument('--save_epoch_freq', type=int, default=2,help='frequency of saving checkpoints at the end of epochs')
parser.add_argument('--no_html', action='store_true',help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
parser.add_argument('--isTrain', type = bool,default=True,help='training flag')
parser.add_argument('--manualSeed', type=int, default =None, help='manual seed')
parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
parser.add_argument('--which_epoch', type=str, default='latest',
                     help='which epoch to load? set to latest to use latest cached model')
parser.add_argument('--pool_size', type=int, default=62, help='the size of image buffer that stores previously generated images')
parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
opt = parser.parse_args()
print(opt)


face_dataset = CreateDataLoader(opt, csv_fileA='./dataset/lfw/fine_grained_attribute_trainA.txt',root_dirA='./dataset/lfw/trainA/',
                                     csv_fileB='./dataset/lfw/fine_grained_attribute_trainB.txt', root_dirB='./dataset/lfw/trainB/')

dataset_size = len(face_dataset)
print('#training images = %d' % dataset_size)

if opt.manualSeed is None:
    opt.manualSeed = np.random.randint(1, 10000)

np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)

model = create_model(opt)
visualizer = Visualizer(opt)
train_num = dataset_size

#### stage-1 ###########
print('---------- Stage1 training -------------')
stage = 1
total_steps = 0
for epoch in range(1, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    # batch_indices = np.random.choice(train_num, train_num, replace=False)
    batch_idx = range(np.int(train_num/opt.batchSize))
    shuffle(batch_idx)
    epoch_iter = 0
    for i in batch_idx:
        # print(i)
        idx = np.arange(i*opt.batchSize,(i+1)*opt.batchSize)
        iter_start_time = time.time()
        total_steps += 1*opt.batchSize
        # epoch_iter = total_steps - dataset_size * (epoch - 1)
        epoch_iter += opt.batchSize
        model.set_input(face_dataset[idx])
        model.optimize_stage1_parameters()
        # if total_steps % opt.display_freq == 0:
        visualizer.display_current_results(model.get_current_visuals(stage), epoch, stage)


        # if total_steps % opt.print_freq == 0:
        errors = model.get_current_errors(stage)
        # t = (time.time() - iter_start_time) / opt.batchSize
        t = time.time() - iter_start_time
        visualizer.print_current_errors(epoch, epoch_iter, errors, t)
        if opt.display_id > 0:
           visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors, stage)

        # print(total_steps)
        if total_steps % opt.save_latest_freq == 0:
           print('saving the latest model (epoch %d, total_steps %d)' %
                 (epoch, total_steps))
           model.save('latest', stage)

    if epoch % opt.save_epoch_freq == 0:
       print('saving the model at the end of epoch %d, iters %d' %
             (epoch, total_steps))
       model.save('latest', stage)
       model.save(epoch, stage)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
         (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    if epoch > opt.niter:
       model.update_learning_rate()

stage1_epoch = opt.niter+opt.niter_decay
# # ########## stage-2 ###########
print('---------- Stage2 training -------------')
stage = 2
model.old_lr = opt.lr

total_steps = 0
for epoch in range(1, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    # batch_indices = np.random.choice(train_num, train_num, replace=False)
    batch_idx = range(np.int(train_num/opt.batchSize))
    shuffle(batch_idx)
    epoch_iter = 0
    for i in batch_idx:
        # print(i)
        idx = np.arange(i*opt.batchSize,(i+1)*opt.batchSize)
        iter_start_time = time.time()
        total_steps += 1*opt.batchSize
        # epoch_iter = total_steps - dataset_size * (epoch - 1)
        epoch_iter += opt.batchSize
        model.set_input(face_dataset[idx])
        model.optimize_stage2_parameters()
        # if total_steps % opt.display_freq == 0:
        visualizer.display_current_results(model.get_current_visuals(stage), epoch, stage)

        # if total_steps % opt.print_freq == 0:
        errors = model.get_current_errors(stage)
        # t = (time.time() - iter_start_time) / opt.batchSize
        t = time.time() - iter_start_time
        visualizer.print_current_errors(epoch, epoch_iter, errors, t)
        if opt.display_id > 0:
            visualizer.plot_current_errors(stage1_epoch+epoch, float(epoch_iter) / dataset_size, opt, errors, stage)

        # print(total_steps)
        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest',stage)

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest',stage)
        model.save(stage1_epoch+epoch, stage)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    if epoch > opt.niter:
        model.update_learning_rate()

stage2_epoch = (opt.niter+opt.niter_decay)*2
print('---------- Stage3 training -------------')
stage = 3
model.old_lr = opt.lr

total_steps = 0
for epoch in range(1, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    # batch_indices = np.random.choice(train_num, train_num, replace=False)
    batch_idx = range(np.int(train_num/opt.batchSize))
    shuffle(batch_idx)
    epoch_iter = 0
    for i in batch_idx:
        # print(i)
        idx = np.arange(i*opt.batchSize,(i+1)*opt.batchSize)
        iter_start_time = time.time()
        total_steps += 1*opt.batchSize
        # epoch_iter = total_steps - dataset_size * (epoch - 1)
        epoch_iter += opt.batchSize
        model.set_input(face_dataset[idx])
        model.optimize_stage3_parameters()
        model.optimize_stage2_parameters()
        model.optimize_stage1_parameters()
        # if total_steps % opt.display_freq == 0:
        visualizer.display_current_results(model.get_current_visuals(stage), epoch, stage)

        # if total_steps % opt.print_freq == 0:
        errors = model.get_current_errors(stage)
        # t = (time.time() - iter_start_time) / opt.batchSize
        t = time.time() - iter_start_time
        visualizer.print_current_errors(epoch, epoch_iter, errors, t)
        if opt.display_id > 0:
            visualizer.plot_current_errors(stage2_epoch+epoch, float(epoch_iter) / dataset_size, opt, errors, stage)

        # print(total_steps)
        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest',stage)

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest',stage)
        model.save(stage2_epoch+epoch, stage)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    if epoch > opt.niter:
        model.update_learning_rate()
