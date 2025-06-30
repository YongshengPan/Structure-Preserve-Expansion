# coding=utf-8
import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from core.simulator import thd_dice_loss, dice_coeff, thd_dice
import random
import os
import config


def extend_by_dim(krnlsz, model_type='3d', half_dim=1):
    """
    生成合适的kernel size，以适应不同维度的卷积操作
    '2d'：返回一个 2D 核大小，例如 krnlsz=5 时，返回 (5, 5)。
    '3d'：返回一个 3D 核大小，例如 krnlsz=5 时，返回 (5, 5, 5)。
    '2.5d'：返回一个介于 2D 和 3D 之间的核大小。
    例如，krnlsz=5 且 half_dim=2 时，返回 (2, 5, 5)，其中第一个维度仅使用 half_dim 控制的大小，而其他维度保持原 krnlsz。
    其他类型：默认返回一个 1D 尺寸列表，即 (krnlsz,)
    """
    if model_type == '2d':
        outsz = [krnlsz] * 2
    elif model_type == '3d':
        outsz = [krnlsz] * 3
    elif model_type == '2.5d':
        outsz = [(np.array(krnlsz) * 0 + 1) * half_dim] + [krnlsz] * 2
    else:
        outsz = [krnlsz]
    return tuple(outsz)


def getKernelbyType(model_type='3D'):
    """
    :param model_type:
    :return: 卷积块，归一化层，池化层
    """
    if model_type == '3d':
        ConvBlock, InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
        max_pool, avg_pool = nn.MaxPool3d, nn.AvgPool3d
    elif model_type == '2.5d':
        ConvBlock, InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
        max_pool, avg_pool = nn.MaxPool3d, nn.AvgPool3d
    else:
        ConvBlock, InstanceNorm = nn.Conv2d, nn.InstanceNorm2d
        max_pool, avg_pool = nn.MaxPool2d, nn.AvgPool2d
    return ConvBlock, InstanceNorm, max_pool, avg_pool


def build_end_activation(input, activation='linear', alpha=None):
    """
    最后一层的激活函数
    """
    if activation == 'softmax':
        output = F.softmax(input, dim=1)
    elif activation == 'sigmoid':
        output = torch.sigmoid(input)
    elif activation == 'elu':
        if alpha is None: alpha = 0.01
        output = F.elu(input, alpha=alpha)
    elif activation == 'lrelu':
        if alpha is None: alpha = 0.01
        output = F.leaky_relu(input, negative_slope=alpha)
    elif activation == 'relu':
        output = F.relu(input)
    elif activation == 'tanh':
        output = F.tanh(input) #* (3-2.0*torch.relu(1-torch.relu(input*100)))
    else:
        output = input
    return output


class TriDis_Mix(nn.Module):
    """
    能是对输入张量进行标准化（归一化）处理，
    并在某些条件下引入随机均值和方差，从而实现数据扰动以增强模型的鲁棒性。
    这种方法类似于数据增强技术，通常用于提升模型的泛化能力。
    """
    def __init__(self, prob=0.5, alpha=0.1, eps=1.0e-6,):
        super(TriDis_Mix, self).__init__()

        self.prob = prob
        self.alpha = alpha
        self.eps = eps
        self.mu = []
        self.var = []

    def forward(self, x: torch.Tensor, ):
        shp = x.shape
        smpshp = [shp[0], shp[1]] + [1]*(len(shp)-2)
        if torch.rand(1) > self.prob:
            return x
        mu = torch.mean(x, dim=[d for d in range(2, len(shp))], keepdim=True)
        var = torch.var(x, dim=[d for d in range(2, len(shp))], keepdim=True)
        sig = torch.sqrt(var+self.eps)
        if (sig == 0).any():
            print(sig)
        x_normed = (x-mu.detach()) / sig.detach()
        mu_r = torch.rand(smpshp, device=mu.device)
        sig_r = torch.rand(smpshp, device=mu.device)
        lmda = torch.distributions.Beta(self.alpha, self.alpha).sample(smpshp)
        bern = torch.bernoulli(lmda).to(mu.device)
        mu_mix = mu_r*bern + mu * (1.0-bern)
        sig_mix = sig_r * bern + sig * (1.0 - bern)
        return x_normed * sig_mix + mu_mix


class StackConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='same', mid_channels=None,
            model_type='3d', residualskip=False,  device=None, dtype=None):
        super(StackConvBlock, self).__init__()

        self.change_dimension = in_channels != out_channels
        self.model_type = model_type
        self.residualskip = residualskip
        padding = {'same': kernel_size//2, 'valid': 0}[padding] if padding in ['same', 'valid'] else padding
        mid_channels = out_channels if mid_channels is None else mid_channels

        if self.model_type == '3d':
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
        elif self.model_type == '2.5d':
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
        else:
            self.ConvBlock, self.InstanceNorm = nn.Conv2d, nn.InstanceNorm2d

        def extdim(krnlsz, halfdim=1):
            return extend_by_dim(krnlsz, model_type=model_type.lower(), half_dim=halfdim)
        self.short_cut_conv = self.ConvBlock(in_channels, out_channels, 1, extdim(stride))
        self.norm0 = self.InstanceNorm(out_channels, affine=True)
        self.conv1 = self.ConvBlock(in_channels, mid_channels, extdim(kernel_size, 3), extdim(stride), padding=extdim(padding, 1), padding_mode='reflect')
        self.norm1 = self.InstanceNorm(mid_channels, affine=True)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv2 = self.ConvBlock(mid_channels, out_channels, extdim(kernel_size, 3), extdim(1), padding=extdim(padding, 1), padding_mode='reflect')
        self.norm2 = self.InstanceNorm(out_channels, affine=True, track_running_stats=False)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        if self.residualskip and self.change_dimension:
            short_cut_conv = self.norm0(self.short_cut_conv(x))
        else:
            short_cut_conv = x
        o_c1 = self.relu1(self.norm1(self.conv1(x)))
        o_c2 = self.norm2(self.conv2(o_c1))
        if self.residualskip:
            out_res = self.relu2(o_c2+short_cut_conv)
        else:
            out_res = self.relu2(o_c2)
        return out_res


class Generic_UNetwork(nn.Module):
    def __init__(self, in_channels, output_channel=2, basedim=8, downdepth=2, model_type='3D', isresunet=True,
                 istransunet=False, activation_function='sigmoid', use_max_pool=False, use_attention=False, use_triD=False, use_skip=True):
        super(Generic_UNetwork, self).__init__()
        self.output_channel = output_channel
        self.model_type = model_type.lower()
        self.downdepth = downdepth
        self.activation_function = activation_function
        self.isresunet = isresunet
        self.istransunet = istransunet
        self.use_max_pool = use_max_pool
        self.use_triD = use_triD
        self.use_attention = use_attention
        if self.model_type == '2d':
            self.ConvBlock, self.InstanceNorm = nn.Conv2d, nn.InstanceNorm2d
            self.MaxPool, self.AvgPool = nn.MaxPool2d, nn.AvgPool2d
        else:
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
            self.MaxPool, self.AvgPool = nn.MaxPool3d, nn.AvgPool3d
        self.tridis_mix = TriDis_Mix(prob=0.5, alpha=0.1, eps=1.0e-6)
        self.bulid_network(in_channels, basedim, downdepth, output_channel)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def extdim(self, krnlsz, halfdim=1):
        return extend_by_dim(krnlsz, model_type=self.model_type, half_dim=halfdim)

    def bulid_network(self, in_channels, basedim, downdepth=2, output_channel=2):
        self.begin_conv = StackConvBlock(in_channels, basedim, 7, 1, model_type=self.model_type, residualskip=self.isresunet)
        if self.use_max_pool:
            self.encoding_block = nn.ModuleList([nn.Sequential(
                self.MaxPool(self.extdim(3), self.extdim(2), padding=self.extdim(1, 0)),
                StackConvBlock(basedim * 2 ** convidx, basedim * 2 ** (convidx + 1), 3, 1,
                               model_type=self.model_type, residualskip=self.isresunet)) for
                convidx in range(0, downdepth)])
        else:
            self.encoding_block = nn.ModuleList([nn.Sequential(
                StackConvBlock(basedim * 2 ** convidx, basedim * 2 ** (convidx + 1), 3, 2,
                               model_type=self.model_type, residualskip=self.isresunet)) for
                convidx in range(0, downdepth)])

        trans_dim = basedim * 2 ** downdepth
        if self.istransunet:
            self.trans_block = nn.Sequential(nn.TransformerEncoder(nn.TransformerEncoderLayer(trans_dim, 8), 12))
        else:
            self.trans_block = nn.Sequential(
                StackConvBlock(trans_dim, trans_dim, 1, 1, model_type=self.model_type, residualskip=self.isresunet),
                StackConvBlock(trans_dim, trans_dim, 1, 1, model_type=self.model_type, residualskip=self.isresunet),
                StackConvBlock(trans_dim, trans_dim, 1, 1, model_type=self.model_type, residualskip=self.isresunet),
                StackConvBlock(trans_dim, trans_dim, 1, 1, model_type=self.model_type, residualskip=self.isresunet),
                StackConvBlock(trans_dim, trans_dim, 1, 1, model_type=self.model_type, residualskip=self.isresunet),
                StackConvBlock(trans_dim, trans_dim, 1, 1, model_type=self.model_type, residualskip=self.isresunet),
            )

        self.decoding_block = nn.ModuleList([
            StackConvBlock(basedim * 2 ** (convidx + 2), basedim * 2 ** convidx, 3, 1, model_type=self.model_type,
                           mid_channels=basedim * 2 ** (convidx + 1), residualskip=self.isresunet) for convidx in range(0, downdepth)
        ])
        
        self.end_conv = StackConvBlock(basedim * 2, basedim, 3, 1, model_type=self.model_type, residualskip=self.isresunet)
        # self.start_conv = nn.Conv3d(in_channels, in_channels, kernel_size=(3,1,1), stride=(2, 1, 1), padding=(1, 0, 0))
        self.class_conv = self.ConvBlock(basedim, output_channel, self.extdim(7), stride=1, dilation=1, padding=self.extdim(3, 0), padding_mode='reflect')

    def forward(self, x):
        # x = self.start_conv(x)
        o_c1 = self.begin_conv(x)
        feats = [o_c1, ]
        for convidx in range(0, len(self.encoding_block)):
            if self.training and self.use_triD:
                o_c1 = self.tridis_mix(o_c1)
            o_c1 = self.encoding_block[convidx](o_c1)
            # print("---", o_c1.shape)
            # x = F.interpolate(x, scale_factor=self.extdim(1/2), mode="trilinear")
            feats.append(o_c1)
        if self.istransunet:
            o_c2 = torch.transpose(o_c1.view([*o_c1.size()[0:2], -1]), 1, 2)
            o_c2 = self.trans_block(o_c2)
            o_c2 = torch.transpose(o_c2, 1, 2).view(o_c1.size())
        else:
            o_c2 = self.trans_block(o_c1)

        for convidx in range(self.downdepth, 0, -1):
            o_c2 = torch.concat((o_c2, feats[convidx]), dim=1)
            o_c2 = self.decoding_block[convidx - 1](o_c2)
            o_c2 = F.interpolate(o_c2, scale_factor=self.extdim(2), mode="trilinear")

        o_c3 = self.end_conv(torch.concat((o_c2, feats[0]), dim=1))
        # o_c3 = self.end_conv2(o_c3)
        o_cls = self.class_conv(o_c3)
        prob = build_end_activation(o_cls, self.activation_function)
        return [o_cls, prob, ]


class AdverserialNetwork(nn.Module):
    def __init__(self, in_channel, basedim=8, downdepth=2, model_type='3D', activation_function=None):
        super(AdverserialNetwork, self).__init__()
        self.model_type = model_type.lower()
        self.activation_function = activation_function
        if self.model_type == '2d':
            self.ConvBlock, self.InstanceNorm = nn.Conv2d, nn.InstanceNorm2d
            self.MaxPool, self.AvgPool = nn.MaxPool2d, nn.AvgPool2d
        else:
            self.ConvBlock, self.InstanceNorm = nn.Conv3d, nn.InstanceNorm3d
            self.MaxPool, self.AvgPool = nn.MaxPool3d, nn.AvgPool3d

        self.adverserial_network = self.bulid_network(in_channel, basedim, downdepth)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def extdim(self, krnlsz, halfdim=1):
        return extend_by_dim(krnlsz, model_type=self.model_type, half_dim=halfdim)

    def bulid_network(self, channel, basedim=8, depth=3):
        adverserial_network = nn.Sequential(
            nn.Sequential(self.ConvBlock(channel, basedim, self.extdim(4), stride=2, dilation=1,
                                         padding=self.extdim(2, 0), padding_mode='reflect'),
                          nn.LeakyReLU()),
            *[nn.Sequential(
                self.ConvBlock(basedim * 2 ** dpt, basedim * 2 ** (dpt + 1), self.extdim(4), stride=2, dilation=1,
                               padding=self.extdim(2, 0), padding_mode='reflect'),
                self.InstanceNorm(basedim * 2 ** (dpt + 1)),
                nn.LeakyReLU()
            ) for dpt in range(0, depth)],
            self.ConvBlock(basedim * 2 ** depth, 1, self.extdim(4), stride=1, dilation=1,
                               padding=self.extdim(2, 0), padding_mode='reflect'))
        return adverserial_network

    def forward(self, x, **kwargs):
        features = []
        for layer in self.adverserial_network:
            x = layer(x)
            features.append(x)
        prob = build_end_activation(x, self.activation_function)
        return prob, features


class Trusteeship:
    def __init__(self, module: nn.Module, loss_fn, volin=('CT1',), volout=('CT2',),
                 metrics=('mae', ), advmodule: nn.Module=None, device="cpu", ckpt_prefix='', volume_names=None):
        self.volin, self.volout = volin, volout
        self.module = module
        # self.advmodule = advmodule  # 单卡（推理时）
        self.advmodule = advmodule.module  # 多卡（训练时）
        self.device = device
        self.ckpt_prefix = ckpt_prefix
        self.to_device(device)
        self.loss_fun = loss_fn
        self.metrics = metrics
        self.volume_names = volume_names
        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=1e-3)

    def to_device(self, device):
        self.module.to(device)
        if self.advmodule is not None:
            self.advmodule.to(device)

    def loss_function(self, pred, target, add_con=None, classes=None):
        # print(pred[1].shape, target.shape)
        total_loss = torch.mean(pred[1])*0
        if total_loss.isnan():
            print(total_loss)
        if 'dice' in self.loss_fun:
            total_loss += self._dice_loss_(pred[1], target, classes)
        if 'crep' in self.loss_fun:
            # total_loss += nn.functional.binary_cross_entropy_with_logits(pred[0], target)
            total_loss += self._cross_entropy_(pred[0], target, classes)
        if 'mse' in self.loss_fun:
            total_loss += nn.functional.mse_loss(pred[1], target)
        if 'mae' in self.loss_fun:
            total_loss += nn.functional.l1_loss(pred[1], target)
        if 'thd' in self.loss_fun:
            total_loss += thd_dice_loss(pred[1], target, thres=(-1000, 0.1, 0.85, 0.99, 1.15, 1000))
        if 'pdc' in self.loss_fun:
            pr, tr = torch.relu(pred[1]), torch.relu(target)
            coeff = (torch.minimum(pr, tr)) / (torch.maximum(pr, tr) + 1.0e-6)
            total_loss += 1-coeff.mean()
        return total_loss

    def metrics_function(self, pred, target, add_con=None, classes=None):
        metrics = {}
        if 'dice' in self.metrics:
            metrics['dice'] = dice_coeff(pred[1], target, classes)
        if 'crep' in self.metrics:
            # total_loss += nn.functional.binary_cross_entropy_with_logits(pred[0], target)
            metrics['crep'] += self._cross_entropy_(pred[0], target, classes)
        if 'rmse' in self.metrics:
            metrics['mse'] = torch.sqrt(nn.functional.mse_loss(pred[1], target))
        if 'mae' in self.metrics:
            metrics['mae'] = nn.functional.l1_loss(pred[1], target)
        if 'thd' in self.metrics:
            metrics['thd'] = thd_dice(pred[1], target,  thres=(-1000, 0.1, 0.85, 0.99, 1.15, 1000))
        return metrics

    def gradient_loss(self, img1, img2):
        # 计算梯度
        grad1 = torch.gradient(img1, dim=(2, 3, 4))
        grad2 = torch.gradient(img2, dim=(2, 3, 4))
        # 计算梯度差的L2损失
        loss = 0
        for g1, g2 in zip(grad1, grad2):
            loss += torch.mean((g1 - g2) ** 2)
        return loss


    def train_step_g1(self, datadict):
        # print(datadict['CT1'].shape, datadict['CT2'].shape, datadict['CT3'].shape)
        modalities = {modal: datadict[modal].to(self.device) for modal in datadict}
        inputs = torch.concat([modalities[modal] for modal in datadict if modal in self.volin], axis=1)
        outputs = torch.concat([modalities[modal] for modal in datadict if modal in self.volout], axis=1)
        pred = self.module(inputs,)
        lambda3 = 0.02
        gen_loss = self.loss_function(pred, outputs)
        loss_gradient = self.gradient_loss(pred[1], outputs)  # 预测结果和GT的梯度
        loss = gen_loss + lambda3 * loss_gradient
        D_loss = 0
        if 'adv' in self.loss_fun or 'msl' in self.loss_fun:
            adv_grth, adv_pred = self.advmodule(outputs), self.advmodule(pred[1].detach())
            # adv_loss是 1/2 * ( |D(G(input)) - 1| + |D(input_shift)| )
            adv_loss = torch.mean(torch.abs(adv_pred[0] - 1) + torch.abs(adv_grth[0]))
            D_loss += adv_loss
            self.advmodule.optimizer.zero_grad()
            adv_loss.backward()
            self.advmodule.optimizer.step()
            adv_pred = self.advmodule(pred[1])
            fmloss = sum([torch.mean(torch.abs(adv_pred[1][ly] - adv_grth[1][ly].detach()))
                                                    for ly in range(len(adv_pred[1]) - 1)])
        else:
            fmloss = 0
        loss = loss + fmloss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        res = [loss, D_loss, gen_loss, fmloss, loss_gradient]
        return res

    def train_step(self, datadict, freeze_model):
        # print(datadict['CT1'].shape, datadict['CT2'].shape, datadict['CT3'].shape)
        modalities = {modal: datadict[modal].to(self.device) for modal in datadict}
        inputs = torch.concat([modalities[modal] for modal in datadict if modal in self.volin], axis=1)
        outputs = torch.concat([modalities[modal] for modal in datadict if modal in self.volout], axis=1)
        # 用新的训练好模型替代 self.module
        with torch.no_grad():
            inputs = freeze_model(inputs)[1].to(self.device)  # new model inference
        pred = self.module(inputs,)
        lambda3 = 0.02
        gen_loss = self.loss_function(pred, outputs)
        loss_gradient = self.gradient_loss(pred[1], outputs)  # 预测结果和GT的梯度
        loss = gen_loss + lambda3 * loss_gradient
        D_loss = 0
        if 'adv' in self.loss_fun or 'msl' in self.loss_fun:
            adv_grth, adv_pred = self.advmodule(outputs), self.advmodule(pred[1].detach())
            # adv_loss是 1/2 * ( |D(G(input)) - 1| + |D(input_shift)| )
            adv_loss = torch.mean(torch.abs(adv_pred[0] - 1) + torch.abs(adv_grth[0]))
            D_loss += adv_loss
            self.advmodule.optimizer.zero_grad()
            adv_loss.backward()
            self.advmodule.optimizer.step()
            adv_pred = self.advmodule(pred[1])
            fmloss = sum([torch.mean(torch.abs(adv_pred[1][ly] - adv_grth[1][ly].detach()))
                                                    for ly in range(len(adv_pred[1]) - 1)])
        else:
            fmloss = 0
        loss = loss + fmloss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        res = [loss, D_loss, gen_loss, fmloss, loss_gradient]
        return res

    def train_step_diffusion(self, datadict, sqrt_alpha_cp):
        t = random.randint(0, len(sqrt_alpha_cp[0])-1)
        modalities = {modal: datadict[modal].to(self.device) for modal in datadict}
        inputs = torch.concat([modalities[modal] for modal in datadict if modal in self.volin], axis=1)
        outputs = torch.concat([modalities[modal] for modal in datadict if modal in self.volout], axis=1)

        outputs_t = outputs*sqrt_alpha_cp[0][t]+(torch.randn(outputs.size(), device=outputs.device)+1) * sqrt_alpha_cp[1][t]
        pred = self.module(torch.concat((inputs, outputs_t), dim=1),)  # , coord.unique()
        loss = nn.functional.mse_loss(outputs*sqrt_alpha_cp[0][t]+(pred[0]+1) * sqrt_alpha_cp[1][t], outputs_t) / (sqrt_alpha_cp[1][t] * sqrt_alpha_cp[1][t])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def __save_memory_infer(self, inputs, split_size=None, stride=(96, 32, 32), outdevice=None):
        predicts = None
        if outdevice is None:
            outdevice = inputs.device
        size = inputs.size()
        weight = 1
        for dm in range(0, len(split_size)):
            weight = weight * torch.exp(-(torch.arange(0, split_size[dm])/split_size[dm]-0.5)**2).view([1]*dm+[-1]+[1]*(len(split_size)-dm-1))
        weight = weight.to(outdevice)
        iterXYZ = list(product(range(0, size[2]-split_size[0]+stride[0], stride[0]), range(0, size[3]-split_size[1]+stride[1], stride[1]), range(0, size[4]-split_size[2]+stride[2], stride[2])))
        for iXYZ in iterXYZ:
            iXYZ = np.minimum(iXYZ, [size[2] - split_size[0], size[3] - split_size[1], size[4] - split_size[2]])
            pred = self.module(inputs[:, :, iXYZ[0]:iXYZ[0] + split_size[0], iXYZ[1]:iXYZ[1] + split_size[1], iXYZ[2]:iXYZ[2] + split_size[2]].to(self.device),)
            if predicts is None:
                predicts = torch.zeros(pred[0].size()[0:2] + inputs.size()[2::], device=outdevice), torch.zeros(pred[1].size()[0:2] + inputs.size()[2::], device=outdevice)
                weights = torch.zeros((1, 1) + inputs.size()[2::], device=outdevice)+1.0e-6
            predicts[0][:, :, iXYZ[0]:iXYZ[0] + split_size[0], iXYZ[1]:iXYZ[1] + split_size[1], iXYZ[2]:iXYZ[2] + split_size[2]] += pred[0].to(outdevice)
            predicts[1][:, :, iXYZ[0]:iXYZ[0] + split_size[0], iXYZ[1]:iXYZ[1] + split_size[1], iXYZ[2]:iXYZ[2] + split_size[2]] += pred[1].to(outdevice)
            weights[:, :, iXYZ[0]:iXYZ[0] + split_size[0], iXYZ[1]:iXYZ[1] + split_size[1], iXYZ[2]:iXYZ[2] + split_size[2]] += 1
        predicts = predicts[0] / weights, predicts[1] / weights
        return predicts

    def eval_step(self, datadict, split_size=None, stride=(96, 32, 32), outdevice='cpu'):
        modalities = {modal: datadict[modal] for modal in datadict}
        inputs = torch.concat([modalities[modal] for modal in datadict if modal in self.volin], axis=1)
        outputs = torch.concat([modalities[modal] for modal in datadict if modal in self.volout], axis=1)
        outdevice = outputs.device
        if split_size is not None:
            predicts = self.__save_memory_infer(inputs, split_size=split_size, stride=stride, outdevice=outdevice)
        else:
            predicts = self.module(inputs.to(self.device), )
        loss = self.loss_function(predicts, outputs)
        # self.metrics = self.metrics_function(predicts, outputs)
        return loss, predicts

    def infer_step(self, datadict, split_size=None, stride=(96, 32, 32),):
        inputs = torch.concat([datadict[modal] for modal in datadict if modal in self.volin], axis=1)
        outdevice = inputs.device
        if split_size is not None:
            predicts = self.__save_memory_infer(inputs, split_size=split_size, stride=stride,)
        else:
            predicts = self.module(inputs.to(self.device),)
            predicts = [item.to(outdevice) for item in predicts]
        return predicts

    def _dice_loss_(self, pred, target, classes=None):
        smooth = 1.
        coeff = 0
        all_classes = range(pred.size(1)) if classes is None else classes
        for idx in range(len(all_classes)):
            m1, m2 = pred[:, idx], (target[:, 0] == all_classes[idx])*1.0
            intersection = (m1 * m2).sum()
            union = m1.sum() + m2.sum()
            coeff += (2. * intersection + smooth) / (union + smooth)
        return 1-coeff/len(all_classes)

    def _cross_entropy_(self, pred, target, classes=None):
        smooth = 1.
        crossentropy = 0
        all_classes = range(pred.size(1)) if classes is None else classes
        for idx in range(len(all_classes)):
            m1, m2 = pred[:, idx], (target[:, 0] == all_classes[idx])*1.0
            crossentropy += F.binary_cross_entropy_with_logits(m1, m2)
        coeff = crossentropy / len(all_classes)
        return coeff

    def load_dict(self, dict_name, weights_store_file="", strict=True):
        # print(self.loss_fun, self.ckpt_prefix, dict_name)
        # print(os.path.join('weightsTotal', '_'.join(self.loss_fun), self.ckpt_prefix, dict_name))
        state_dict = torch.load(os.path.join(weights_store_file, '_'.join(self.loss_fun),
                                             self.ckpt_prefix, dict_name),
                                        weights_only=False, map_location='cpu')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_k = k.replace("module.", "")  # 删除 "module." 前缀
            new_state_dict[new_k] = v
        # self.module.load_state_dict(state_dict, strict=strict)  # 单卡
        self.module.load_state_dict(new_state_dict, strict=strict)  # 多卡

    def save_dict(self, dict_name):
        # path_temp = os.path.join(config.s4to1weightsStoreFile, '_'.join(self.loss_fun), self.ckpt_prefix)
        path_temp = os.path.join(config.weightsStoreFile, '_'.join(self.loss_fun), self.ckpt_prefix)
        if not os.path.exists(path_temp):
            os.makedirs(path_temp)
        torch.save(self.module.state_dict(), os.path.join(path_temp, self.ckpt_prefix+'_'+dict_name))
        return os.path.join(path_temp, self.ckpt_prefix+'_'+dict_name)

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()
