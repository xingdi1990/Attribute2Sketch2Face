import numpy as np
import os
import argparse
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import torch
from util import html

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='attr_sketch_face',  help='name of the model to be used: stackgan, attr2img')
parser.add_argument('--name', type=str, default='cuhk_attr_sketch_face', help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--gpu_ids', type=str , default= '0' ,  help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--loadSize', type=int, default=64, help='scale images to this size')
parser.add_argument('--fineSize', type=int, default=64, help='then crop to this size')
parser.add_argument('--attrB_dim', type=int, default=17, help='# of input attribute channels')
parser.add_argument('--attrA_dim', type=int, default=6, help='# of input attribute channels')
parser.add_argument('--sketch_nc', type=int, default=1, help='# of output image channels')
parser.add_argument('--image_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--nt', type=int, default=256, help='#  of dim for text features')
parser.add_argument('--nz', type=int, default=1024, help='#  of dim for Z')
parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='# of descriminator filters in first conv layer')
parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--display_winsize', type=int, default=64,  help='display window size')
parser.add_argument('--display_single_pane_ncols', type=int, default=0,
                         help='if positive, display all images in a single visdom web panel with certain number of images per row.')
parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')

parser.add_argument('--how_many', type=int, default=6880, help='how many test images to run')
parser.add_argument('--phase', type=str, default='test',   help='train, val, test, etc')
parser.add_argument('--which_epoch', type=str, default='120',
                     help='which epoch to load? set to latest to use latest cached model')
parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
parser.add_argument('--isTrain', type = bool,default=False,help='training flag')
parser.add_argument('--no_html', action='store_true',help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
parser.add_argument('--manualSeed', type=int, default =777, help='manual seed')
parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
parser.add_argument('--pool_size', type=int, default=62, help='the size of image buffer that stores previously generated images')
opt = parser.parse_args()
print(opt)


# face_dataset = CreateDataLoader(opt, csv_fileA='./dataset/lfw/fine_grained_attribute_testA.txt',root_dirA='./dataset/lfw/testA/',
#                                      csv_fileB='./dataset/lfw/fine_grained_attribute_testB.txt', root_dirB='./dataset/lfw/testB/')
face_dataset = CreateDataLoader(opt, csv_fileA='./dataset/CUHK/fine_grained_attribute_testA.txt',root_dirA='./dataset/CUHK/testA/',
                                     csv_fileB='./dataset/CUHK/fine_grained_attribute_testB.txt', root_dirB='./dataset/CUHK/testB/')

dataset_size = len(face_dataset)
print('#test images = %d' % dataset_size)

if opt.manualSeed is None:
    opt.manualSeed = np.random.randint(1, 10000)

np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)

model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
test_num = dataset_size
stage = 3
for i in range(test_num):
    idx = [i]
    print(i)
    if i >= opt.how_many:
        break
    model.set_input(face_dataset[idx])
    model.test()
    visuals = model.get_current_visuals(stage)
    # print(visuals)
    img_pathA, img_pathB  = model.get_image_paths()
    # print('process images... %s %s' % img_pathA % img_pathB )
    #visualizer.save_images(webpage, visuals, img_pathA)
    visualizer.save_inception_images(webpage, visuals, img_pathA)
    # visualizer.save_images(webpage, visuals, img_pathB)

webpage.save()
