import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import models
from util.vgg16 import Vgg16
import util.util as util
import os

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def define_E(attrB, sketch_nc, nz, nt, ngf, gpu_ids=[]):
    netE1 = None
    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        assert(torch.cuda.is_available())
    netE1 = E1(sketch_nc, nz, nt, ngf, attrB, gpu_ids=gpu_ids)
    if len(gpu_ids) > 0:
        netE1.cuda()
    netE1.apply(weights_init)
    return netE1


def define_G( attrB, attrA, sketch_nc, image_nc, nz, nt, ngf, gpu_ids=[]):
    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        assert(torch.cuda.is_available())
    netG1 = G1(sketch_nc, nz, nt, ngf, attrB, gpu_ids=gpu_ids)
    netG2 = Dense(nt, gpu_ids=gpu_ids)
    netG3 = ConUnetGenerator(image_nc, image_nc, attrB+attrA, ngf, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=gpu_ids)

    if len(gpu_ids) > 0:
        netG1.cuda()
        netG2.cuda()
        netG3.cuda()
    netG1.apply(weights_init)
    netG2.apply(weights_init)
    netG3.apply(weights_init)
    return netG1, netG2, netG3

def define_D(attrB, attrA, sketch_nc, image_nc, nz, nt, ndf, gpu_ids=[]):
    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        assert(torch.cuda.is_available())
    print(gpu_ids)
    netD2 = NLayerDiscriminator(sketch_nc+image_nc, ndf, n_layers = 4, norm_layer = nn.BatchNorm2d , use_sigmoid=True, gpu_ids=gpu_ids )
    netD3 = NLayerDiscriminator(image_nc*2, ndf, n_layers= 4, norm_layer=nn.BatchNorm2d, use_sigmoid=True, gpu_ids=gpu_ids)
    if len(gpu_ids) > 0:
        netD2.cuda()
        netD3.cuda()
    netD2.apply(weights_init)
    netD3.apply(weights_init)
    return netD2, netD3

def define_P(gpu_ids=[]):
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert(torch.cuda.is_available())
    netP =Vgg16()
    util.init_vgg16('./')
    netP.load_state_dict(torch.load(os.path.join('./', "vgg16.weight")))
    for param in netP.parameters():
        param.requires_grad = False
    if use_gpu:
        netP.cuda()
    return netP

###################################################################################
## this following code is from Han Zhang's work: https://github.com/hanzhanggit/StackGAN-Pytorch
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        # nn.Upsample(scale_factor=2, mode='nearest'),
        nn.UpsamplingNearest2d(scale_factor=2),
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True))
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(True),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out

class CA_NET1(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self, attrB, nt, gpu_ids=[]):
        super(CA_NET1, self).__init__()
        # we define the conditional vector dimension
        self.c_dim = nt
        self.fc1 = nn.Linear(attrB, self.c_dim * 2, bias=False)
        # self.fc = F.linear(attrB, self.c_dim * 2, bias=None)
        self.relu = nn.ReLU()
        self.gpu_ids = gpu_ids

    def encode(self, text_embedding):
        x = self.relu(self.fc1(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if len(self.gpu_ids) > 0:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar

## this following code is from  Jun-Yan Zhu and Taesung Park https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=False, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        # if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        return self.model(input)


############# Networks for E1 #############
class E1(nn.Module):
    def __init__(self, sketch_nc, nz, nt, ngf, attrB, gpu_ids=[]):
        super(E1, self).__init__()
        self.gpu_ids = gpu_ids
        self.gf_dim = ngf
        self.ef_dim = nt
        self.z_dim = nz
        self.c_dim = self.gf_dim * 16
        self.define_module(sketch_nc, attrB, gpu_ids )
        self.fc1 = nn.Linear(self.gf_dim * 16, self.gf_dim * 16 * 2, bias=True)
        self.relu = nn.ReLU()

        # # TEXT.DIMENSION -> GAN.CONDITION_DIM
        # self.ca_net1 = CA_NET1(attrB, nt, gpu_ids)
    def define_module(self, sketch_nc, attrB, gpu_ids):
        ninput = self.gf_dim * 16 + self.ef_dim
        ngf = self.gf_dim
        self.encoderIMG = nn.Sequential(
            nn.Conv2d(sketch_nc, ngf, 5, 1, 2, bias=False),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 x 32
            nn.Conv2d(ngf, ngf * 2, 5, 1, 2, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16 x 16
            nn.Conv2d(ngf * 2, ngf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8 x 8
            nn.Conv2d(ngf * 4, ngf * 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),  # 4 x 4
            nn.Conv2d(ngf * 8, ngf * 16, 4, 1, 0, bias=False),
            nn.ReLU(True),  # 1 x 1
            nn.Dropout(0.5)
        )
        self.sp = nn.Sequential(
            conv3x3(ngf // 16, sketch_nc),
            nn.Tanh(),
        )
        self.cc = nn.Sequential(
            conv3x3(ngf // 16, sketch_nc),
        )
        self.encoderZ = nn.Sequential(
            nn.Linear(self.z_dim, ngf * 16, bias=False),
            nn.BatchNorm1d(ngf * 16),
            nn.ReLU(True),
        )
        self.fc_text = nn.Sequential(
            nn.Linear(attrB, self.ef_dim, bias=False),
            nn.BatchNorm1d(self.ef_dim),
            nn.ReLU(True),
        )
        self.fc = nn.Sequential(
            nn.Linear(ninput, ngf * 4 * 4, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4),
            nn.ReLU(True))
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if len(self.gpu_ids) > 0:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def encode_latent(self, text_embedding):
        x = self.relu(self.fc1(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def forward(self, noise, text_embedding, sketch):
        # c_code, mu, logvar = self.ca_net1(text_embedding)
        encode_text = self.fc_text(text_embedding)
        encode_img = self.encoderIMG(sketch)
        encode_img = encode_img.view(-1, self.gf_dim * 16)

        i_c_code = torch.cat((encode_img, encode_text ), 1)
        l_code = self.fc(i_c_code)
        l_mu , l_logvar = self.encode_latent(l_code)
        l_code = self.reparametrize(l_mu, l_logvar)

        encode_n = self.encoderZ(noise)
        z_c_code = torch.cat((encode_n, encode_text ), 1)
        z_code = self.fc(z_c_code)
        z_mu , z_logvar = self.encode_latent(z_code)
        z_code = self.reparametrize(z_mu, z_logvar)

        return [z_code, z_mu, z_logvar], [l_code, l_mu, l_logvar], encode_text


############# Networks for G1 #############
class G1(nn.Module):
    def __init__(self, sketch_nc, nz, nt, ngf, attrB,gpu_ids=[]):
        super(G1, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = nt
        self.z_dim = nz
        self.define_module(attrB, nt, sketch_nc, gpu_ids)

    def define_module(self, attrB, nt, sketch_nc, gpu_ids):
        ninput = self.gf_dim * 16 + self.ef_dim
        ngf = self.gf_dim
        # TEXT.DIMENSION -> GAN.CONDITION_DIM
        self.ca_net1 = CA_NET1(attrB, nt, gpu_ids)
        self.decoder = nn.Sequential(
            upBlock(ngf, ngf // 2),
            upBlock(ngf // 2, ngf // 4),
            upBlock(ngf // 4, ngf // 8),
            upBlock(ngf // 8, ngf // 16),
        )
        self.sp = nn.Sequential(
            conv3x3(ngf // 16, sketch_nc),
            nn.Tanh(),
        )
        self.cc = nn.Sequential(
            conv3x3(ngf // 16, sketch_nc),
        )
    def forward(self, l_code, z_code):
        l_code = l_code.view(-1, self.gf_dim, 4, 4)
        recon_img = self.decoder(l_code)
        cc = self.cc(recon_img)
        sp = self.sp(recon_img)

        z_code = z_code.view(-1, self.gf_dim, 4, 4)
        fake_img = self.decoder(z_code)
        fsp = self.sp(fake_img)
        fcc = self.cc(fake_img)
        return [sp, cc], [fsp, fcc],

class G2(nn.Module):
    def __init__(self, input_nc, output_nc, nz, nt, ngf, attr, gpu_ids=[]):
        super(G2, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = nt
        self.z_dim = nz
        self.output_nc = output_nc
        # self.STAGE1_G = STAGE1_G(input_nc, output_nc, nz, nt, ngf, norm_layer, use_dropout, gpu_ids)
        # # fix parameters of stageI GAN
        # for param in self.STAGE1_G.parameters():
        #     param.requires_grad = False
        self.define_module(input_nc, output_nc, nt, attr, gpu_ids)

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(2):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self, input_nc, output_nc, nt, attr, gpu_ids):
        ngf = self.gf_dim
        # TEXT.DIMENSION -> GAN.CONDITION_DIM
        self.ca_net = CA_NET2(attr, nt, gpu_ids)

        self.encoder = nn.Sequential(
            conv3x3(input_nc, ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            #            nn.Conv2d(ngf * 8, ngf * 16, 4, 2, 1, bias=False),
            #            nn.BatchNorm2d(ngf * 16),
            #            nn.ReLU(True),
        )
        self.hr_joint = nn.Sequential(
            conv3x3(self.ef_dim + ngf * 8, ngf * 8),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True))
        self.residual = self._make_layer(ResBlock, ngf * 8)

        self.upsample1 = upBlock(ngf * 8, ngf * 4)

        self.upsample2 = upBlock(ngf * 4, ngf * 2)

        self.upsample3 = upBlock(ngf * 2, ngf * 1)

        # self.upsample4 = upBlock(ngf * 1, ngf // 2)
        # --> 3 x 64 x 64
        self.img = nn.Sequential(
            conv3x3(ngf * 1, self.output_nc),
            nn.Tanh())

    def forward(self, stage1_img, text_embedding):
        encoded_img = self.encoder(stage1_img)

        c_code, mu, logvar = self.ca_net(text_embedding)
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 8, 8)
        i_c_code = torch.cat([encoded_img, c_code], 1)
        h_code = self.hr_joint(i_c_code)
        h_code = self.residual(h_code)

        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)


        fake_img = self.img(h_code)
        return fake_img, mu, logvar

# ############# Networks for D2 #############
#class D2(nn.Module):
#    def __init__(self, attrA, image_nc, sketch_nc, nz, nt, ndf, gpu_ids = []):
#        super(D2, self).__init__()
#        self.df_dim = ndf
#        self.ef_dim = nt*2
#        self.define_module(image_nc, sketch_nc)

#    def define_module(self, image_nc, sketch_nc):
#        ndf, nef = self.df_dim, self.ef_dim
#        self.encode_img = nn.Sequential(
#            nn.Conv2d(image_nc, ndf, 4, 2, 1, bias=False),
#            nn.LeakyReLU(0.2, inplace=True), # 32 * 32 * ndf * 2
#            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#            nn.BatchNorm2d(ndf * 2),
#            nn.LeakyReLU(0.2, inplace=True),  # 16 * 16 * ndf * 2
#            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
#            nn.BatchNorm2d(ndf * 4),
#            nn.LeakyReLU(0.2, inplace=True),  # 8 * 8 * ndf * 4
#            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
#            nn.BatchNorm2d(ndf * 8),
#            nn.LeakyReLU(0.2, inplace=True),  # 4 * 4 * ndf * 8
#            conv3x3(ndf * 8, ndf * 4),
#            nn.BatchNorm2d(ndf * 4),
#            nn.LeakyReLU(0.2, inplace=True),   # 4 * 4 * ndf * 4
#            conv3x3(ndf * 4, ndf * 2),
#            nn.BatchNorm2d(ndf * 2),
#            nn.LeakyReLU(0.2, inplace=True)   # 4 * 4 * ndf * 2
#        )

#        self.get_cond_logits = D_GET_LOGITS(ndf * 2, nef, bcondition=True)
#        self.get_uncond_logits = D_GET_LOGITS(ndf * 2, nef, bcondition=False)

#    def forward(self, image):
#        img_embedding = self.encode_img(image)
#        output = img_embedding

#        return output

class ConUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, attr, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(ConUnetGenerator, self).__init__()
        self.attD = attr
        self.down1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, True),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*4, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(0.2, True),
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(ngf*4, ngf*8, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(ngf*8),
            nn.LeakyReLU(0.2, True),
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf*8),
            nn.LeakyReLU(0.2, True),
        )
        self.hr_joint = nn.Sequential(
            conv3x3(attr+ngf*8, ngf*8),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
        )
        self.residual = self._make_layer(ResBlock, ngf*8)
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(ngf*16, ngf*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 16, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(2):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)
    def forward(self, input, attr):
        m1 = self.down1(input)
        m2 = self.down2(m1)
        m3 = self.down3(m2)
        m4 = self.down4(m3)
        m5 = self.down5(m4)
        attr = attr.view(-1, self.attD, 1, 1)
        attr = attr.repeat(1, 1, 2, 2)
        mf = self.hr_joint(torch.cat([m5, attr], 1))
        mf = self.residual(mf)
        u5 = self.up5(torch.cat([mf, m5],1))
        u4 = self.up4(torch.cat([u5, m4], 1))
        u3 = self.up3(torch.cat([u4, m3], 1))
        u2 = self.up2(torch.cat([u3, m2], 1))
        output = self.up1(torch.cat([u2, m1], 1))
        return output


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                                        padding=0, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.upsample_nearest(out, scale_factor=2)

class Dense(nn.Module):
    def __init__(self, nt, gpu_ids=[]):
        super(Dense, self).__init__()
        self.ef_dim = nt

        ############# 64x64  ##############
        haze_class = models.densenet121(pretrained=True)

        self.conv0 = haze_class.features.conv0
        self.norm0 = haze_class.features.norm0
        self.relu0 = haze_class.features.relu0
        self.pool0 = haze_class.features.pool0

        ############# Block1-down 16x16  ##############
        self.dense_block1 = haze_class.features.denseblock1
        self.trans_block1 = haze_class.features.transition1

        ############# Block2-down 8x8  ##############
        self.dense_block2 = haze_class.features.denseblock2
        self.trans_block2 = haze_class.features.transition2

        ############# Block3-down  4x4 ##############
        self.dense_block3 = haze_class.features.denseblock3
        self.trans_block3 = haze_class.features.transition3

        ############# Block4-up  512x2x2  ##############
        self.hr_joint = nn.Sequential(
            conv3x3(nt+512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.residual = self._make_layer(ResBlock, 512)

        self.bottle_block4 = BottleneckBlock(512, 256)
        self.trans_block4 = TransitionBlock(768, 128)

        ############# Block5-up  4x4 ##############
        self.bottle_block5 = BottleneckBlock(384, 256)
        self.trans_block5 = TransitionBlock(640, 128)

        ############# Block6-up 8x8   ##############
        self.bottle_block6 = BottleneckBlock(256, 128)
        self.trans_block6 = TransitionBlock(384, 64)

        ############# Block7-up 16x16   ##############
        self.bottle_block7 = BottleneckBlock(64, 64)
        self.trans_block7 = TransitionBlock(128, 32)

        ## 128 X  128
        ############# Block8-up c 32x32  ##############
        self.bottle_block8 = BottleneckBlock(32, 32)
        self.trans_block8 = TransitionBlock(64, 16)

        self.tanh = nn.Tanh()


    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(2):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)
    def forward(self, x, attr):
        ## 256x256
        x0 = self.pool0(self.relu0(self.norm0(self.conv0(x))))

        ## 64 X 64
        x1 = self.dense_block1(x0)
        # print x1.size()
        x1 = self.trans_block1(x1)

        ###  32x32
        x2 = self.trans_block2(self.dense_block2(x1))
        # print  x2.size()


        ### 16 X 16
        x3 = self.trans_block3(self.dense_block3(x2))

        # x3=Variable(x3.data,requires_grad=True)
        # print(x3.size())
        attr = attr.view(-1, self.ef_dim,1,1)
        attr = attr.repeat(1,1,2,2)
        xf = self.hr_joint(torch.cat([x3, attr],1))
        xf = self.residual(xf)

        ## 8 X 8
        x4 = self.trans_block4(self.bottle_block4(xf))
        # x4 = self.trans_block4(self.bottle_block4(x3))

        x42 = torch.cat([x4, x2], 1)
        ## 16 X 16
        x5 = self.trans_block5(self.bottle_block5(x42))

        x52 = torch.cat([x5, x1], 1)
        ##  32 X 32
        x6 = self.trans_block6(self.bottle_block6(x52))

        ##  64 X 64
        x7 = self.trans_block7(self.bottle_block7(x6))

        ##  128 X 128
        x8 = self.trans_block8(self.bottle_block8(x7))

        # print x8.size()
        # print x.size()

        x8 = torch.cat([x8, x], 1)

        # print x8.size()

        x9 = self.relu(self.conv_refin(x8))

        out = self.tanh(self.refine3(x9))

        return output

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
