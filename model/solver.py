# reference:
# https://github.com/icpm/super-resolution
# https://github.com/twtygqyy/pytorch-vdsr
from math import log10
import random
import torch
import torch.backends.cudnn as cudnn
import os
from .SRMD import SRMD
from .RRDB import Classifier, weights_init_kaiming
from .RaGAN import RaGAN
from .PatchGAN import PatchGAN
from .VGG19 import FeatureExtractor
from utils import progress_bar, get_platform_path, get_logger, shave, print_options
from torchvision.transforms import transforms
from PIL import Image
from torchvision import utils as vutils
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
from torch.autograd import Variable
import numpy as np
import itertools

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor


class MDDRBasic(object):
    def __init__(self, config, device=None):
        super(MDDRBasic, self).__init__()
        self.CUDA = torch.cuda.is_available()
        self.device = device

        # ==============================================================
        # =                  models configuration                      =
        # ==============================================================
        # classifier
        self.classifier = None
        self.channels = config.channels
        self.grow_channels = config.grow_channels
        self.scale_ratio = config.scale_ratio
        self.RoIPooling_shape = config.RoIPooling_shape
        self.n_classes = config.n_classes

        # SR generator: SRMD
        self.generator = None
        self.num_blocks = config.num_blocks
        self.num_channels = config.n_classes + config.data_channel
        self.conv_dim = config.conv_dim
        self.upscale_factor = config.upscaleFactor[0]

        # discriminator: RaGAN
        self.discriminator = None
        self.input_shape = (config.data_channel, config.img_size, config.img_size)

        # feature extractor: VGG19
        self.feature_extractor = None
        self.feature_layer = config.feature_layer

        # information discriminator: PatchGAN from CycleGAN
        self.info_discriminator = None
        self.ndf = config.ndf
        self.n_layers = config.n_layers

        # degradation generator:
        self.degradator = None
        self.input_nc = config.data_channel * 2

        self.color_space = config.color_space
        self.test_upscaleFactor = config.test_upscaleFactor
        self.model_name = "{}-{}x".format(config.model, self.upscale_factor)

        # checkpoint configuration
        self.resume = config.resume
        self.checkpoint_name = "{}.pth".format(self.model_name)
        self.best_quality = 0
        self.start_epoch = 0
        self.n_epochs = config.n_epochs
        self.checkpoint_interval = config.checkpoint_interval

        # logger configuration
        _, _, _, log_dir = get_platform_path()
        self.logger = get_logger("{}/{}.log".format(log_dir, self.model_name))
        self.logger.info(print_options(config))

        # tensorboard writer
        self.writer = SummaryWriter(config.tensorboard_log_dir)
        self.tensorboard_image_interval = config.tensorboard_image_interval
        self.tensorboard_draw_model = config.tensorboard_draw_model
        self.tensorboard_input = config.tensorboard_input
        self.tensorboard_image_size = config.tensorboard_image_size
        self.tensorboard_image_sample = config.tensorboard_image_sample

        # distributed
        self.distributed = config.distributed
        self.local_rank = config.local_rank

    def progress_bar(self, index, length, message):
        if not self.distributed or self.local_rank == 0:
            progress_bar(index, length, message)

    def load_model(self):
        _, _, checkpoint_dir, _ = get_platform_path()
        print('==> Resuming from checkpoint...')
        assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('{}/{}'.format(checkpoint_dir, self.checkpoint_name))
        new_state_dict = OrderedDict()
        for key, value in checkpoint['net'].items():
            key = key.replace('module.', '')
            new_state_dict[key] = value
        self.generator.load_state_dict(new_state_dict)
        self.best_quality = checkpoint['psnr']
        self.start_epoch = checkpoint['epoch'] + 1
        self.logger.info("Start from epoch {}, best PSNR: {}".format(self.start_epoch, self.best_quality))

    @staticmethod
    def psrn(mse):
        return 10 * log10(1 / mse)


class MDDRTrainer(MDDRBasic):
    def __init__(self, config, train_loader=None, test_loader=None, device=None):
        super(MDDRTrainer, self).__init__(config, device)

        # ==============================================================
        # =                parameters configuration                    =
        # ==============================================================
        # classifier
        self.lr_c = config.lr_c
        self.beta1_c = config.beta1_c
        self.beta2_c = config.beta2_c
        self.warmup_c = config.warmup_c
        self.criterion_c = None
        self.optimizer_c = None
        self.scheduler_c = None

        # generator
        self.lr_g = config.lr_g
        self.beta1_g = config.beta1_g
        self.beta2_g = config.beta2_g
        self.warmup_g = config.warmup_g
        self.criterion_g = None
        self.optimizer_g = None
        self.scheduler_g = None

        # discriminator
        self.lr_d = config.lr_d
        self.beta1_d = config.beta1_d
        self.beta2_d = config.beta2_d
        self.criterion_d = None
        self.optimizer_d = None
        self.scheduler_d = None

        # info discriminator
        self.lr_i = config.lr_i
        self.beta1_i = config.beta1_i
        self.beta2_i = config.beta2_i
        self.criterion_i = None
        self.optimizer_i = None
        self.scheduler_i = None

        self.lambda_adv = float(config.lambda_adv)
        self.lambda_pixel = float(config.lambda_pixel)

        self.seed = config.seed

        # data loader
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.repeat = config.repeat
        self.batch_size = config.batch_size

        # models init
        self.build_model()

    def build_model(self):

        # classifier
        self.classifier = Classifier(n_classes=self.n_classes, RoIPooling_shape=self.RoIPooling_shape,
                                     channels=self.channels, grow_channels=self.grow_channels,
                                     scale_ratio=self.scale_ratio)
        self.criterion_c = torch.nn.CrossEntropyLoss()
        self.optimizer_c = torch.optim.Adam(self.classifier.parameters(), lr=self.lr_c,
                                            betas=(self.beta1_c, self.beta2_c))
        self.classifier.apply(weights_init_kaiming)

        # generator
        self.generator = SRMD(num_blocks=self.num_blocks, num_channels=self.num_channels,
                              conv_dim=self.conv_dim, scale_factor=self.upscale_factor).to(self.device)
        self.criterion_g = torch.nn.MSELoss()
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr_g,
                                            betas=(self.beta1_g, self.beta2_g))
        self.generator.apply(weights_init_kaiming)

        # discriminator
        self.discriminator = RaGAN(input_shape=self.input_shape)
        self.criterion_d = torch.nn.BCEWithLogitsLoss()
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_d,
                                            betas=(self.beta1_d, self.beta2_d))
        self.discriminator.apply(weights_init_kaiming)

        # feature extractor
        self.feature_extractor = FeatureExtractor(feature_layer=self.feature_layer)
        self.criterion_content = torch.nn.L1Loss()

        # degradation generator
        # self.degradator =

        # info_discriminator
        self.info_discriminator = PatchGAN(input_nc=self.input_nc, ndf=self.ndf, n_layers=self.n_layers)
        self.criterion_i = torch.nn.L1Loss()
        self.optimizer_i = torch.optim.Adam(
            itertools.chain(self.generator.parameters(), self.discriminator.parameters()), lr=self.lr_i,
            betas=(self.beta1_i, self.beta2_i))
        self.info_discriminator.apply(weights_init_kaiming)

        if self.resume:
            self.load_model()

        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.classifier.cuda()
            self.generator.cuda()
            self.discriminator.cuda()
            self.info_discriminator.cuda()
            self.feature_extractor.cuda()
            self.criterion_content.cuda()
            self.criterion_c.cuda()
            self.criterion_g.cuda()
            self.criterion_d.cuda()
            self.criterion_i.cuda()

    def save_model(self, epoch, avg_psnr, name):
        _, _, checkpoint_dir, _ = get_platform_path()
        model_out_path = '{}/{}'.format(checkpoint_dir, name)
        state = {
            'net': self.classifier.state_dict(),
            'psnr': avg_psnr,
            'epoch': epoch
        }
        torch.save(state, model_out_path)
        self.logger.info("checkpoint saved to {}".format(model_out_path))

    def set_train(self):
        self.classifier.train()
        self.generator.train()
        self.discriminator.train()
        self.info_discriminator.train()
        self.feature_extractor.eval()

    def train(self, epoch):
        self.set_train()
        train_loss = 0
        data_size = len(self.train_loader) * self.repeat
        correct = 0
        for index, (idx, img, target) in enumerate(self.train_loader):
            batches_done = epoch * data_size + index
            assert img.shape == target.shape, 'the shape of input is not equal to the shape of output'
            idx, img, target = idx.to(self.device), img.to(self.device, dtype=torch.double), target.to(self.device,
                                                                                                       dtype=torch.double)
            batch_size, channel, height, width = img.shape

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((batch_size, *self.discriminator.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((batch_size, *self.discriminator.output_shape))), requires_grad=False)

            # ------------------
            #  Train Classifier
            # ------------------
            self.optimizer_c.zero_grad()
            label = self.classifier(img)
            loss_c = self.criterion_c(label, idx)
            loss_c.backward()
            self.optimizer_c.step()
            if batches_done < self.warmup_c:
                _, predict = label.max(1)
                correct += predict.eq(idx).sum().item()
                print('correct: {}/{}, {}, loss_c：{}'.format(correct, (index + 1) * self.batch_size,
                                                             100 * correct / ((index + 1) * self.batch_size), loss_c))
                continue

            # ------------------
            #  Train Generators
            # ------------------
            self.generator.zero_grad()
            prior = torch.ones([batch_size, self.n_classes, height, width]).to(self.device)
            for i in range(batch_size):
                for j in range(self.n_classes):
                    prior[i, j, :, :] = prior[i, j, :, :] * label[i, j]
            gen_hr = self.generator(torch.cat([prior.detach(), img], dim=1))

            # Measure pixel-wise loss against ground truth
            loss_pixel = self.criterion_g(gen_hr, target)

            if batches_done < self.warmup_g:
                # Warm-up (pixel-wise loss only)
                loss_pixel.backward()
                self.optimizer_g.step()
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]"
                    % (epoch, self.n_epochs, batches_done, data_size, loss_pixel.item())
                )
                continue

            # Extract validity predictions from discriminator
            pred_real = self.discriminator(gen_hr).detach()
            pred_fake = self.discriminator(target)

            # Adversarial loss (relativistic average GAN)
            loss_GAN = self.criterion_d(pred_fake - pred_real.mean(0, keepdim=True), valid)

            # Content loss
            gen_features = self.feature_extractor(gen_hr)
            real_features = self.feature_extractor(target).detach()
            loss_content = self.criterion_content(gen_features, real_features)

            # Total generator loss
            loss_G = loss_content + self.lambda_adv * loss_GAN + self.lambda_pixel * loss_pixel
            loss_G.backward()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            self.optimizer_d.zero_grad()

            pred_real = self.discriminator(target)
            pred_fake = self.discriminator(gen_hr.detach())

            # Adversarial loss for real and fake images (relativistic average GAN)
            loss_real = self.criterion_d(pred_real - pred_fake.mean(0, keepdim=True), valid)
            loss_fake = self.criterion_d(pred_fake - pred_real.mean(0, keepdim=True), fake)

            # Total loss
            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            self.optimizer_d.step()

            # ---------------------
            #  Information Loss
            # ---------------------
            # self.optimizer_i.zero_grad()
            #
            # posterior = self.info_discriminator(torch.cat([img, gen_hr.detach(), ], dim=1))
            # info_loss = self.criterion_i(posterior, )

            if not self.distributed or self.local_rank == 0:
                progress_bar(index, len(self.train_loader), 'Loss: %.4f' % (train_loss / (index + 1)))

        avg_train_loss = train_loss / len(self.train_loader)
        print("    Average Loss: {:.4f}".format(avg_train_loss))

        return avg_train_loss

    def set_test(self):
        self.classifier.eval()
        self.generator.eval()
        self.discriminator.eval()
        self.info_discriminator.eval()

    def test(self):
        self.set_test()
        psnr = 0
        # random sample output to tensorboard
        save_inputs, save_outputs, save_targets = [], [], []
        with torch.no_grad():
            for index, (idx, img, target) in enumerate(self.test_loader):
                assert img.shape == target.shape, 'the shape of input is not equal to the shape of output'
                idx, img, target = idx.to(self.device), img.to(self.device), target.to(self.device)
                output = self.model(img).clamp(0.0, 1.0)
                output, target = shave(output, target, self.test_upscaleFactor)
                loss = self.criterion(output, target)
                psnr += self.psrn(loss.item() / target.shape[2] / target.shape[3])
                if not self.distributed or self.local_rank == 0:
                    progress_bar(index, len(self.test_loader), 'PSNR: %.4f' % (psnr / (index + 1)))
                    if index < self.tensorboard_image_sample:
                        save_inputs.append(img)
                        save_outputs.append(output)
                        save_targets.append(target)

        avg_psnr = psnr / len(self.test_loader)
        print("    Average PSNR: {:.4f} dB".format(avg_psnr))
        return avg_psnr, save_inputs, save_outputs, save_targets

    def run(self):
        for epoch in range(self.start_epoch, self.n_epochs + self.start_epoch):
            print('\n===> Epoch {} starts:'.format(epoch))
            avg_train_loss = self.train(epoch)
            avg_psnr, save_input, save_output, save_target = self.test()
            self.scheduler.step()

            if not self.distributed or self.local_rank == 0:

                # save to logger
                self.logger.info(
                    "Epoch [{}/{}]: lr={:.6f} loss={:.6f} PSNR={:.6f}".format(epoch, self.n_epochs + self.start_epoch,
                                                                              self.optimizer.param_groups[0]['lr'],
                                                                              avg_train_loss, avg_psnr))

                # save best models
                if avg_psnr > self.best_quality:
                    self.best_quality = avg_psnr
                    self.save_model(epoch, avg_psnr, self.checkpoint_name)

                # save interval models
                if epoch % self.checkpoint_interval == 0:
                    name = self.checkpoint_name.replace('.pth', '_{}.pth'.format(epoch))
                    self.save_model(epoch, avg_psnr, name)

                # tensorboard scalars
                self.writer.add_scalar('train_loss', avg_train_loss, epoch)
                self.writer.add_scalar('psnr', avg_psnr, epoch)

                # tensorboard graph
                if epoch == 0 and self.tensorboard_draw_model and \
                        len(save_input) > 0 and len(save_target) > 0:
                    self.writer.add_graph(model=self.model, input_to_model=[save_input[0]])

                # tensorboard images
                if epoch % self.tensorboard_image_interval == 0:

                    assert len(save_input) == len(save_output) == len(save_target), \
                        'the size of save_input and save_output and save_target is not equal.'
                    for i in range(len(save_target)):
                        save_input[i] = F.interpolate(save_input[i], size=self.tensorboard_image_size,
                                                      mode='bicubic', align_corners=True)
                        save_output[i] = F.interpolate(save_output[i], size=self.tensorboard_image_size,
                                                       mode='bicubic', align_corners=True)
                        save_target[i] = F.interpolate(save_target[i], size=self.tensorboard_image_size,
                                                       mode='bicubic', align_corners=True)
                        images = torch.cat((save_input[i], save_output[i], save_target[i]))
                        grid = vutils.make_grid(images)
                        self.writer.add_image('image-{}'.format(i), grid, epoch)
