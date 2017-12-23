import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util

from . import asf_networks as asf_networks

class attrsketchfaceModel():
    def name(self):
        return 'attrsketchfaceModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        if opt.phase == "train":
            self.isTrain = True
        else:
            self.isTrain = False

        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.FloatTensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.nz = opt.nz
        self.nb = opt.batchSize
        self.real_label = Variable(torch.FloatTensor(self.nb,1).fill_(1).cuda(),requires_grad=False)
        self.fake_label = Variable(torch.FloatTensor(self.nb,1).fill_(0).cuda(),requires_grad=False)
        size = opt.fineSize

        self.input_attrB = self.Tensor(self.nb, opt.attrB_dim)
        self.input_attrA = self.Tensor(self.nb, opt.attrA_dim)
        self.input_sketch = self.Tensor(self.nb, opt.sketch_nc, size, size)
        self.input_image = self.Tensor(self.nb, opt.image_nc, size, size)
        self.input_noise = self.Tensor(self.nb, opt.nz)

        self.netE1 = asf_networks.define_E(opt.attrB_dim, opt.sketch_nc,  opt.nz, opt.nt, opt.ngf, self.gpu_ids)
        self.netG1, self.netG2, self.netG3 = asf_networks.define_G(opt.attrB_dim, opt.attrA_dim, opt.sketch_nc, opt.image_nc, opt.nz, opt.nt, opt.ngf, self.gpu_ids)
        self.netD2, self.netD3 = asf_networks.define_D(opt.attrB_dim, opt.attrB_dim, opt.sketch_nc, opt.image_nc, opt.nz, opt.nt, opt.ndf, self.gpu_ids)
        self.netP = asf_networks.define_P(self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netE1, 'E1', opt.which_epoch)
            self.load_network(self.netG1, 'G1', opt.which_epoch)
            self.load_network(self.netG2, 'G2', opt.which_epoch)
            self.load_network(self.netG3, 'G3', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD2, 'D2', opt.which_epoch)
                self.load_network(self.netD3, 'D3', opt.which_epoch)

        # loss criterion
        self.criterionBCE =  torch.nn.BCELoss()
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionGAN = asf_networks.GANLoss(use_lsgan=False, tensor=self.Tensor)

        print('---------- Networks initialized -------------')
        asf_networks.print_network(self.netE1)
        asf_networks.print_network(self.netG1)
        asf_networks.print_network(self.netG2)
        asf_networks.print_network(self.netG3)
        if self.isTrain:
            asf_networks.print_network(self.netD2)
	    asf_networks.print_network(self.netD3)
        print('-----------------------------------------------')

        # initialize optimizers
        self.old_lr = opt.lr
        self.optimizer_E1 = torch.optim.Adam(self.netE1.parameters(),
                                            lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_G1 = torch.optim.Adam(self.netG1.parameters(),
                                            lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_G2 = torch.optim.Adam(self.netG2.parameters(),
                                            lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(),
                                            lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_G3 = torch.optim.Adam(self.netG3.parameters(),
                                            lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D3 = torch.optim.Adam(self.netD3.parameters(),
                                            lr=opt.lr, betas=(opt.beta1, 0.999))


    def set_input(self, input):
        self.input_sketch = input['sketch']
        self.input_sketch = self.input_sketch.cuda()
        self.input_attrB = input['attributeB']
        self.input_attrB = self.input_attrB.cuda()
        self.input_image = input['image']
        self.input_image = self.input_image.cuda()
        self.input_attrA = input['attributeA']
        self.input_attrA = self.input_attrA.cuda()
        self.input_attrAB = torch.cat([self.input_attrA,self.input_attrB],1)
        self.input_noise = torch.FloatTensor(self.nb, self.nz).normal_(0, 1)
        self.input_noise = self.input_noise.cuda()
        self.image_pathA = input['nameA']
        self.image_pathB = input['nameB']

    def forward_CVAE(self):
        self.attrB = Variable(self.input_attrB)
        self.noise = Variable(self.input_noise)
        self.real_sketch = Variable(self.input_sketch)
        self.z, self.latent, self.enc_text = self.netE1.forward(self.noise, self.attrB, self.real_sketch)
        self.recon, self.fake = self.netG1.forward(self.latent[0], self.z[0])

    def backward_CVAE(self):
        self.KLloss_z = util.KL_loss(self.z[1], self.z[2])
        self.KLloss_latent = util.KL_loss(self.latent[1], self.latent[2])

        self.Gaussian_loss1 = util.GaussianCriterion(self.recon, self.real_sketch) * 0.0001
        self.Gaussian_loss2 = util.GaussianCriterion(self.fake, self.real_sketch) * 0.0001
        self.Gaussian_loss = self.Gaussian_loss1 + self.Gaussian_loss2
        self.loss_G = self.KLloss_z + self.KLloss_latent + self.Gaussian_loss
        self.loss_G.backward()

    def forward2(self):
        self.attrB = Variable(self.input_attrB)
        self.noise = Variable(self.input_noise)
        self.real_sketch = Variable(self.input_sketch)
        
        self.z, self.latent, self.enc_text = self.netE1.forward(self.noise, self.attrB, self.real_sketch)
        self.recon, self.fake = self.netG1.forward(self.latent[0], self.z[0])

        self.recon2 = self.netG2.forward(torch.cat([self.recon[0],self.recon[0],self.recon[0]],1), self.enc_text)
        self.fake2 = self.netG2.forward(torch.cat([self.fake[0],self.fake[0],self.fake[0]],1),self.enc_text)
    def backward_D2(self):
        #fake
        fake_AB = torch.cat((self.recon[0], self.recon2), 1)
        self.pred_fake = self.netD2.forward(fake_AB.detach())
        self.loss_D2_fake = self.criterionGAN(self.pred_fake, False)

        # Real
        self.real_sketch = torch.cat([self.real_sketch, self.real_sketch, self.real_sketch], 1)
        real_AB = torch.cat((self.recon[0], self.real_sketch), 1)
        self.pred_real = self.netD2.forward(real_AB.detach())
        self.loss_D2_real = self.criterionGAN(self.pred_real, True)

        self.loss_D2 = (self.loss_D2_fake + self.loss_D2_real) * 0.5

        self.loss_D2.backward()

    def backward_G2(self):

        fake_AB = torch.cat((self.recon[0], self.recon2), 1)
        self.pred_fake = self.netD2.forward(fake_AB)
        self.loss_G2_GAN = self.criterionGAN(self.pred_fake, True)

        self.real_resize = Variable(util.vgg_initial(self.real_sketch))
        self.perp_real = self.netP.forward(self.real_resize)[0]
        self.fake_resize = Variable(util.vgg_initial(self.recon2))
        self.perp_fake = self.netP.forward(self.fake_resize)[0]
        self.loss_percp = self.criterionL1(self.perp_fake, self.perp_real) * 10

        self.loss_G2_L1 = self.criterionL1(self.recon2, self.real_sketch) * self.opt.lambda_A

        self.loss_G2 = self.loss_G2_L1 + self.loss_G2_GAN + self.loss_percp

        self.loss_G2.backward()

    def forward3(self):
        self.attrB = Variable(self.input_attrB)
        self.attr = Variable(self.input_attrAB)
        self.noise = Variable(self.input_noise)
        self.real_sketch = Variable(self.input_sketch)
        self.real_image = Variable(self.input_image)
        self.z, self.latent, self.enc_text = self.netE1.forward(self.noise, self.attrB, self.real_sketch)
        self.recon, self.fake = self.netG1.forward(self.latent[0], self.z[0])
        self.recon2 = self.netG2.forward(torch.cat([self.recon[0],self.recon[0],self.recon[0]],1), self.enc_text)
        self.fake2 = self.netG2.forward(torch.cat([self.fake[0],self.fake[0],self.fake[0]],1), self.enc_text)

        self.recon_img = self.netG3.forward(self.recon2,self.attr)

        self.fake_img = self.netG3.forward(self.fake2, self.attr)

    def backward_D3(self):
        #fake
        fake_AB3 = torch.cat((self.recon2, self.recon_img), 1)

        self.pred_fake = self.netD3.forward(fake_AB3.detach())

        self.loss_D3_fake = self.criterionGAN(self.pred_fake, False)

        # Real
        real_AB3 = torch.cat((self.recon2.detach(), self.real_image), 1)
        self.pred_real = self.netD3.forward(real_AB3)
        self.loss_D3_real = self.criterionGAN(self.pred_real, True)

        self.loss_D3 = (self.loss_D3_fake + self.loss_D3_real) * 0.5

        self.loss_D3.backward()

    def backward_G3(self):

        fake_AB3 = torch.cat((self.recon2, self.recon_img), 1)
        self.pred_fake = self.netD3.forward(fake_AB3)
        self.loss_G3_GAN = self.criterionGAN(self.pred_fake, True)

        self.real_resize = Variable(util.vgg_initial(self.real_image))
        self.perp_real = self.netP.forward(self.real_resize)[0]
        self.fake_resize = Variable(util.vgg_initial(self.recon_img))
        self.perp_fake = self.netP.forward(self.fake_resize)[0]
        self.loss_percp = self.criterionL1(self.perp_fake,self.perp_real) * 5

        self.loss_G3_L1 = self.criterionL1(self.recon_img, self.real_image) * self.opt.lambda_A

        self.loss_G3 = self.loss_G3_L1 + self.loss_G3_GAN + self.loss_percp

        self.loss_G3.backward()


    def optimize_stage1_parameters(self):
        self.forward_CVAE()

        self.optimizer_G1.zero_grad()
        self.optimizer_E1.zero_grad()
        self.backward_CVAE()
        self.optimizer_G1.step()
        self.optimizer_E1.step()


    def optimize_stage2_parameters(self):
        self.forward2()

        self.optimizer_D2.zero_grad()
        self.backward_D2()
        self.optimizer_D2.step()

        self.optimizer_G2.zero_grad()
        self.backward_G2()
        self.optimizer_G2.step()

    def optimize_stage3_parameters(self):
        self.forward3()

        self.optimizer_D3.zero_grad()
        self.backward_D3()
        self.optimizer_D3.step()

        self.optimizer_G3.zero_grad()
        self.backward_G3()
        self.optimizer_G3.step()


    def get_current_errors(self, stage):
        if stage == 1:
            return OrderedDict([
                                ('G_GAN', 0),
                                ('G_Gaussian', self.Gaussian_loss.data[0]),
                                ('KLD_latent', self.KLloss_latent.data[0]),
                                ('KLD_z', self.KLloss_z.data[0]),
                                ('D_real', 0),
                                ('D_wrong', 0),
                                ('D_fake', 0),
                                ])
        elif stage == 2:
            return OrderedDict([('G_GAN', self.loss_G2_GAN.data[0]),
                                ('G_Gaussian', self.loss_G2_L1.data[0]),
                                ('perceptual_loss', self.loss_percp.data[0]),
                                ('KLD_latent', 0),
                                ('KLD_z', 0),
                                ('D_real', self.loss_D2_real.data[0]),
                                ('D_wrong', 0),
                                ('D_fake', self.loss_D2_fake.data[0]),
                                ])
        elif stage == 3:
            return OrderedDict([('G_GAN', self.loss_G3_GAN.data[0]),
                                ('G_Gaussian', self.loss_G3_L1.data[0]),
                                ('perceptual_loss', self.loss_percp.data[0]),
                                ('KLD_latent', 0),
                                ('KLD_z', 0),
                                ('D_real', self.loss_D3_real.data[0]),
                                ('D_wrong', 0),
                                ('D_fake', self.loss_D3_fake.data[0]),
                                ])

    def get_current_visuals(self,stage):
        if stage == 1:
            fake_B = util.tensor2im(self.fake[0].data)
            recon_B = util.tensor2im(self.recon[0].data)
            real_B = util.tensor2im(self.real_sketch.data)
            return OrderedDict([ ('fake', fake_B), ('real', real_B),('recon', recon_B) ])
        elif stage == 2:
            fake_sketch = util.tensor2im(self.fake[0].data)
            fake_sketch2 = util.tensor2im(self.fake2.data)
            real_sketch = util.tensor2im(self.real_sketch.data)
            reconstruct_image2 = util.tensor2im(self.recon2.data)
            reconstruct_image = util.tensor2im(self.recon[0].data)
            return OrderedDict([ ('fake_sketch', fake_sketch),
                                 ('fake_sketch2', fake_sketch2),
                                 ('recon_sketch2', reconstruct_image2),('recon_sketch', reconstruct_image),
                                 ('real_sketch', real_sketch),])
        elif stage == 3:
            fake_sketch = util.tensor2im(self.fake[0].data)
            fake_sketch2 = util.tensor2im(self.fake2.data)
            fake_img = util.tensor2im(self.fake_img.data)

            real_sketch = util.tensor2im(self.real_sketch.data)
            real_img = util.tensor2im(self.real_image.data)

            reconstruct_sketch2 = util.tensor2im(self.recon2.data)
            reconstruct_sketch = util.tensor2im(self.recon[0].data)
            reconstruct_img = util.tensor2im(self.recon_img.data)
            return OrderedDict([ ('fake_sketch', fake_sketch),('fake_sketch2', fake_sketch2),
                                 ('real_sketch', real_sketch),('fake_img', fake_img),
                                 ('recon_CVAE', reconstruct_sketch), ('recon_CGAN1', reconstruct_sketch2),
                                 ('recon_CGAN2', reconstruct_img),('real_image', real_img)])

    def test(self):
        self.netG1.eval()
        self.netE1.eval()
        self.netG2.eval()
        self.netG3.eval()

        self.attrB = Variable(self.input_attrB, volatile=True)
        self.attrA = Variable(self.input_attrA, volatile=True)
        self.attr = Variable(torch.cat([self.input_attrA, self.input_attrB],1), volatile = True)
        self.noise = Variable(self.input_noise , volatile=True)
        self.real_sketch = Variable(self.input_sketch, volatile=True)


        self.z, self.latent, self.enc_text = self.netE1.forward(self.noise, self.attrB, self.real_sketch)
        self.recon, self.fake = self.netG1.forward(self.latent[0], self.z[0])
        # self.recon2 = self.netG2.forward(self.recon[0])
        # self.fake2 = self.netG2.forward(self.fake)
        self.recon2 = self.netG2.forward(torch.cat([self.recon[0],self.recon[0],self.recon[0]],1),self.enc_text)
        self.fake2 = self.netG2.forward(torch.cat([self.fake[0],self.fake[0],self.fake[0]],1), self.enc_text)
        # self.recon_img = self.netG3.forward(torch.cat([self.recon2,self.recon2,self.recon2],1))
        self.recon_img = self.netG3.forward(self.recon2, self.attr)
        # self.fake_img = self.netG3.forward(torch.cat([self.fake2,self.fake2,self.fake2],1))
        self.fake_img = self.netG3.forward(self.fake2, self.attr)
        self.real_image = Variable(self.input_image, volatile=True)


    def get_image_paths(self):
        return self.image_pathA, self.image_pathB

    def save(self, label ,stage):
        if stage == 1:
            self.save_network(self.netE1, 'E1', label, self.gpu_ids)
            self.save_network(self.netG1, 'G1', label, self.gpu_ids)

        elif stage == 2:
            self.save_network(self.netE1, 'E1', label, self.gpu_ids)
            self.save_network(self.netG1, 'G1', label, self.gpu_ids)
            self.save_network(self.netG2, 'G2', label, self.gpu_ids)
            self.save_network(self.netD2, 'D2', label, self.gpu_ids)
        elif stage == 3:
            self.save_network(self.netE1, 'E1', label, self.gpu_ids)
            self.save_network(self.netG1, 'G1', label, self.gpu_ids)
            self.save_network(self.netG2, 'G2', label, self.gpu_ids)
            self.save_network(self.netD2, 'D2', label, self.gpu_ids)
            self.save_network(self.netG3, 'G3', label, self.gpu_ids)
            self.save_network(self.netD3, 'D3', label, self.gpu_ids)

    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda()

    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        print(save_path)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        # for param_group in self.optimizer_D1.param_groups:
        #     param_group['lr'] = lr
        for param_group in self.optimizer_G1.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D2.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G2.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D3.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G3.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
