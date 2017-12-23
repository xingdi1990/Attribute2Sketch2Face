from __future__ import print_function
import torch

import torch.nn as nn
from PIL import Image
import inspect, re
import numpy as np
import math
import os
import collections
from torchvision import transforms
#from vggface import Vggface
from torch.utils.serialization import load_lua
from vgg16 import Vgg16


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) ) * 255.0
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    return image_numpy.astype(imtype)

def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD



def GaussianCriterion(input, target):
    Gelement = 0.5*(input[1] + math.log(2 * math.pi))
    Gelement += (target + -1 * input[0]).pow(2)/(torch.exp(input[1]))*0.5
    output = torch.sum(Gelement)
    return output
def KLDCriterion(input):
    # KLDelement = (input[2] + 1):add(-1, torch.pow(input[1], 2)):add(-1, torch.exp(input[2]))
    KLDelement = (input[1] + 1)-1*input[0].pow(2)-1*torch.exp(input[1])
    output = 0.5* torch.sum(KLDelement)
    return output
# def init_vgg16(model_folder):
# 	"""load the vgg16 model feature"""
# 	if not os.path.exists(os.path.join(model_folder, 'vgg16.weight')):
# 		if not os.path.exists(os.path.join(model_folder, 'vgg16.t7')):
# 			os.system(
# 				'wget http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/vgg16.t7 -O ' + os.path.join(model_folder, 'vgg16.t7'))
# 		vgglua = load_lua(os.path.join(model_folder, 'vgg16.t7'))
# 		vgg = Vgg16()
# 		for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
# 			dst.data[:] = src
# 		torch.save(vgg.state_dict(), os.path.join(model_folder, 'vgg16.weight'))



def preprocess(image_numpy):
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    # print(len(image_numpy.shape))
    if image_numpy.shape[-1] != 3:
        image_numpy = np.stack([image_numpy, image_numpy, image_numpy],2)
        preprocess = transforms.Compose(transform_list)
        image_out = preprocess(image_numpy)
        out = image_out[0]
        return out
    else:
        preprocess = transforms.Compose(transform_list)
        image_out = preprocess(image_numpy)
        out = image_out
        return out

def init_vgg16(model_folder):
    """load the vgg16 model feature"""
    if not os.path.exists(os.path.join(model_folder, 'vgg16.weight')):
        if not os.path.exists(os.path.join(model_folder, 'vgg16.t7')):
            os.system(
				'wget http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/vgg16.t7 -O ' + os.path.join(model_folder, 'vgg16.t7'))
            vgglua = load_lua(os.path.join(model_folder, 'vgg16.t7'))
            vgg = Vgg16()
            for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
                dst.data[:] = src
            torch.save(vgg.state_dict(), os.path.join(model_folder, 'vgg16.weight'))


def vgg_initial(image_tensor):
    tensortype = type(image_tensor.data)
    image_tensor_out = tensortype(1, 3, 224, 224)

    # print(image_tensor.data[0])
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Scale(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    img_tensor = transform(image_tensor.data[0].cpu())
    image_tensor_out[0] = img_tensor
    image_tensor_out.cuda()
    return image_tensor_out

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    if image_numpy.shape[-1]==1:
        image_numpy = image_numpy[:,:,0]
        image_pil = Image.fromarray(image_numpy,'L')
        image_pil.save(image_path)
    else:

        image_pil = Image.fromarray(image_numpy)
        image_pil.save(image_path)

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
