# Code adapted from:
# https://github.com/zhenglab/cotere-net

import torch
import torch.nn as nn
import numpy as np
import copy
import math

from timm.models.layers import trunc_normal_

GN_CHANNELS = 16

class ImplicitRelation(nn.Module):
    def __init__(self, input_size, unit_type="C", mlp_r=1):
        super(ImplicitRelation, self).__init__()
        
        assert unit_type in ["C", "T"], "Relation Unit Type should be C or T."
        self.unit_type = unit_type
        
        if self.unit_type == "C":
            self.pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.pooling = nn.AdaptiveAvgPool3d((1, None, 1))
        
        middle_size = int(input_size / mlp_r)
        self.relation = nn.Sequential(
            nn.Linear(input_size, middle_size),
            nn.GroupNorm(middle_size // GN_CHANNELS, middle_size),
            # nn.ReLU(),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(middle_size, input_size),
            nn.GroupNorm(input_size // GN_CHANNELS, input_size))
        
        self.activate = nn.Sigmoid()
        
    def forward(self, x):
        x_in = x

        if self.unit_type == "T":
            x_in = x_in.view(x.size(0), x.size(1), x.size(3), x.size(2), x.size(4))
        
        x_out = self.pooling(x_in)
        x_out = x_out.view(x_out.size(0), -1)

        x_out = self.relation(x_out)
        x_out = self.activate(x_out)
        
        if self.unit_type == "C":
            return x_out.view(x.size(0), x.size(1), 1, 1, 1)
        else:
            return x_out.view(x.size(0), x.size(1), x.size(2), 1, 1)

class ImplicitRelationNew(nn.Module):
    def __init__(self, input_size, unit_type="C", mlp_r=1):
        super(ImplicitRelationNew, self).__init__()
        
        assert unit_type in ["C", "T"], "Relation Unit Type should be C or T."
        self.unit_type = unit_type
        
        if self.unit_type == "C":
            self.pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.pooling = nn.AdaptiveAvgPool3d((1, None, 1))
        
        self.conv = nn.Conv3d(input_size, input_size, kernel_size=1, bias=False)
        self.bn = nn.GroupNorm(input_size // GN_CHANNELS, input_size)
        
        self.activate = nn.Sigmoid()
        
    def forward(self, x):
        x_in = x
        
        if self.unit_type == "T":
            x_in = x_in.view(x.size(0), x.size(1), x.size(3), x.size(2), x.size(4))
        
        x_out = self.pooling(x_in)
        x_out = self.conv(x_out)
        x_out = self.bn(x_out)
        x_out = self.activate(x_out)

        if self.unit_type == "C":
            return x_out.view(x.size(0), x.size(1), 1, 1, 1)
        else:
            return x_out.view(x.size(0), x.size(1), x.size(2), 1, 1)

class ExplicitRelation(nn.Module):
    def __init__(self, num_filters, conv_size, norm_module=nn.BatchNorm3d, freeze_bn=False):
        super(ExplicitRelation, self).__init__()
        
        self.pooling = nn.AdaptiveAvgPool3d((1, None, None))
        
        pad_size = (conv_size - 1) // 2
        self.conv = nn.Conv3d(num_filters, num_filters, kernel_size=(1, conv_size, conv_size), stride=(1, 1, 1), padding=(0, pad_size, pad_size), bias=False)
        # self.bn = norm_module(num_features=num_filters, track_running_stats=(not freeze_bn))
        self.bn = nn.GroupNorm(num_filters // GN_CHANNELS, num_filters)
        self.activate = nn.Sigmoid()
        
    def forward(self, x):
        x_in = x
        
        x_out = self.pooling(x_in)
        x_out = self.conv(x_out)
        x_out = self.bn(x_out)

        x_out = self.activate(x_out)
        return x_out

class COTERE(nn.Module):
    def __init__(self, input_filters, output_filters, input_seq_length, 
                 cotere_type="CTSR", cotere_c_mlp_r=1, cotere_t_mlp_r=1, cotere_s_conv_size=1, 
                 norm_module=nn.BatchNorm3d, freeze_bn=False):
        super(COTERE, self).__init__()

        self.unit = nn.ModuleDict()
        self.collaborate_type = "mul_sum"

        c_mlp_r = cotere_c_mlp_r
        t_mlp_r = cotere_t_mlp_r
        s_conv_size = cotere_s_conv_size

        # self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU(negative_slope=0.2)

        if "C" in cotere_type:
            self.unit["C"] = ImplicitRelation(output_filters, unit_type="C", mlp_r=c_mlp_r)
            # self.unit["C"] = ImplicitRelationNew(output_filters, unit_type="C", mlp_r=c_mlp_r)
        
        if "T" in cotere_type:
            self.unit["T"] = ImplicitRelation(output_filters * input_seq_length, unit_type="T", mlp_r=t_mlp_r)
            # self.unit["T"] = ImplicitRelationNew(output_filters, unit_type="T", mlp_r=t_mlp_r)
        
        if "S" in cotere_type:
            self.unit["S"] = ExplicitRelation(output_filters, s_conv_size, norm_module=norm_module, freeze_bn=freeze_bn)

    def forward(self, x):
        ctsr_in = x

        unit_in = self.relu(ctsr_in)
        unit_keys = list(self.unit.keys())
        unit_out_dict = dict()
        for unit_key in unit_keys:
            unit_out_dict[unit_key] = self.unit[unit_key](unit_in)

        unit_out = None
        if self.collaborate_type == "mul_sum":
            if "T" in unit_keys and "S" in unit_keys:
                unit_out = unit_out_dict["T"] + unit_out_dict["S"]
                if "C" in unit_keys:
                    unit_out = unit_out_dict["C"] * unit_out
            else:
                unit_out = unit_out_dict[unit_keys[0]]
                if len(unit_keys) == 2:
                    unit_out = unit_out * unit_out_dict[unit_keys[1]]
        else:
            unit_out = sum(unit_out_dict.values())

        return ctsr_in * unit_out

def build_conv(block, 
            in_filters, 
            out_filters, 
            kernels, 
            strides=(1, 1, 1), 
            pads=(0, 0, 0), 
            conv_idx=1, 
            block_type="3d",
            norm_module=nn.BatchNorm3d,
            freeze_bn=False):
    
    if block_type == "2.5d":
        i = 3 * in_filters * out_filters * kernels[1] * kernels[2]
        i /= in_filters * kernels[1] * kernels[2] + 3 * out_filters
        middle_filters = int(i)
        
        # 1x3x3 layer
        conv_middle = "conv{}_middle".format(str(conv_idx))
        block[conv_middle] = nn.Conv3d(
            in_filters,
            middle_filters,
            kernel_size=(1, kernels[1], kernels[2]),
            stride=(1, strides[1], strides[2]),
            padding=(0, pads[1], pads[2]),
            bias=False)
        
        bn_middle = "bn{}_middle".format(str(conv_idx))
        # block[bn_middle] = norm_module(num_features=middle_filters, track_running_stats=(not freeze_bn))
        block[bn_middle] = nn.GroupNorm(middle_filters // GN_CHANNELS, middle_filters)
        
        relu_middle = "relu{}_middle".format(str(conv_idx))
        # block[relu_middle] = nn.ReLU(inplace=True)
        # block[relu_middle] = nn.SiLU(inplace=True)
        block[relu_middle] = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # 3x1x1 layer
        conv = "conv{}".format(str(conv_idx))
        block[conv] = nn.Conv3d(
            middle_filters,
            out_filters,
            kernel_size=(kernels[0], 1, 1),
            stride=(strides[0], 1, 1),
            padding=(pads[0], 0, 0),
            bias=False)
    elif block_type == "3d":
        conv = "conv{}".format(str(conv_idx))
        block[conv] = nn.Conv3d(
            in_filters,
            out_filters,
            kernel_size=kernels,
            stride=strides,
            padding=pads,
            bias=False)
    elif block_type == "i3d":
        conv = "conv{}".format(str(conv_idx))
        block[conv] = nn.Conv3d(
            in_filters,
            out_filters,
            kernel_size=kernels,
            stride=strides,
            padding=pads,
            bias=False)

def build_basic_block(block, 
                    input_filters, 
                    output_filters, 
                    use_temp_conv=False,
                    down_sampling=False, 
                    block_type='3d', 
                    norm_module=nn.BatchNorm3d,
                    freeze_bn=False):
    
    if down_sampling:
        strides = (2, 2, 2)
    else:
        strides = (1, 1, 1)
    
    build_conv(block,
            input_filters, 
            output_filters, 
            kernels=(3, 3, 3), 
            strides=strides, 
            pads=(1, 1, 1),
            conv_idx=1,
            block_type=block_type,
            norm_module=nn.BatchNorm3d,
            freeze_bn=freeze_bn)
    # block["bn1"] = norm_module(num_features=output_filters, track_running_stats=(not freeze_bn))
    # block["relu"] = nn.ReLU(inplace=True)
    block["bn1"] = nn.GroupNorm(output_filters // GN_CHANNELS, output_filters)
    # block["relu"] = nn.SiLU(inplace=True)
    block["relu"] = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    build_conv(block,
            output_filters, 
            output_filters, 
            kernels=(3, 3, 3), 
            strides = (1, 1, 1),
            pads=(1, 1, 1), 
            conv_idx=2, 
            block_type=block_type,
            norm_module=nn.BatchNorm3d,
            freeze_bn=freeze_bn)
    # block["bn2"] = norm_module(num_features=output_filters, track_running_stats=(not freeze_bn))
    block["bn2"] = nn.GroupNorm(output_filters // GN_CHANNELS, output_filters)
    
    if (output_filters != input_filters) or down_sampling:
        block["shortcut"] = nn.Conv3d(
            input_filters,
            output_filters,
            kernel_size=(1, 1, 1),
            stride=strides,
            padding=(0, 0, 0),
            bias=False)
        # block["shortcut_bn"] = norm_module(num_features=output_filters, track_running_stats=(not freeze_bn))
        block["shortcut_bn"] = nn.GroupNorm(output_filters // GN_CHANNELS, output_filters)

def build_bottleneck(block, 
                    input_filters, 
                    output_filters, 
                    base_filters, 
                    use_temp_conv=False,
                    down_sampling=False, 
                    block_type='3d', 
                    norm_module=nn.BatchNorm3d,
                    freeze_bn=False):
    
    if down_sampling:
        if block_type == 'i3d':
            strides = (1, 2, 2)
        else:
            strides = (2, 2, 2)
    else:
        strides = (1, 1, 1)
    
    temp_conv_size = 3 if block_type == 'i3d' and use_temp_conv else 1
    
    build_conv(block,
            input_filters, 
            base_filters, 
            kernels=(temp_conv_size, 1, 1) if block_type == 'i3d' else (1, 1, 1), 
            pads=(temp_conv_size // 2, 0, 0) if block_type == 'i3d' else (0, 0, 0),
            conv_idx=1,
            norm_module=norm_module,
            freeze_bn=freeze_bn)
    # block["bn1"] = norm_module(num_features=base_filters, track_running_stats=(not freeze_bn))
    # block["relu1"] = nn.ReLU(inplace=True)
    block["bn1"] = nn.GroupNorm(base_filters // GN_CHANNELS, base_filters)
    # block["relu1"] = nn.SiLU(inplace=True)
    block["relu1"] = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    build_conv(block,
            base_filters, 
            base_filters, 
            kernels=(1, 3, 3) if block_type == 'i3d' else (3, 3, 3), 
            strides=strides,
            pads=(0, 1, 1) if block_type == 'i3d' else (1, 1, 1), 
            conv_idx=2, 
            block_type=block_type,
            norm_module=norm_module,
            freeze_bn=freeze_bn)
    # block["bn2"] = norm_module(num_features=base_filters, track_running_stats=(not freeze_bn))
    # block["relu2"] = nn.ReLU(inplace=True)
    block["bn2"] = nn.GroupNorm(base_filters // GN_CHANNELS, base_filters)
    # block["relu2"] = nn.SiLU(inplace=True)
    block["relu2"] = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    build_conv(block,
            base_filters, 
            output_filters, 
            kernels=(1, 1, 1), 
            conv_idx=3,
            norm_module=norm_module,
            freeze_bn=freeze_bn)
    # block["bn3"] = norm_module(num_features=output_filters, track_running_stats=(not freeze_bn))
    block["bn3"] = nn.GroupNorm(output_filters // GN_CHANNELS, output_filters)
    
    if (output_filters != input_filters) or down_sampling:
        block["shortcut"] = nn.Conv3d(
            input_filters,
            output_filters,
            kernel_size=(1, 1, 1),
            stride=strides,
            padding=(0, 0, 0),
            bias=False)
        # block["shortcut_bn"] = norm_module(num_features=output_filters, track_running_stats=(not freeze_bn))
        block["shortcut_bn"] = nn.GroupNorm(output_filters // GN_CHANNELS, output_filters)

def init_module_weights(m):
    if isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm3d):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def init_module_weights_new(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.GroupNorm, nn.BatchNorm3d)):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv3d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()

class BasicConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 upsampling=False,
                 act_norm=False,
                 act_inplace=True):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if upsampling is True:
            self.conv = nn.Sequential(*[
                nn.Conv2d(in_channels, out_channels*4, kernel_size=kernel_size,
                          stride=1, padding=padding, dilation=dilation),
                nn.PixelShuffle(2)
            ])
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation)

        self.norm = nn.GroupNorm(out_channels // GN_CHANNELS, out_channels)
        self.act = nn.SiLU(inplace=act_inplace)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.GroupNorm, nn.BatchNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class COTEREConvSC(nn.Module):

    def __init__(self,
                 C_in,
                 C_out,
                 kernel_size=3,
                 downsampling=False,
                 upsampling=False,
                 act_norm=True,
                 act_inplace=True):
        super(COTEREConvSC, self).__init__()

        stride = 2 if downsampling is True else 1
        padding = (kernel_size - stride + 1) // 2

        self.conv = BasicConv2d(C_in, C_out, kernel_size=kernel_size, stride=stride,
                                upsampling=upsampling, padding=padding,
                                act_norm=act_norm, act_inplace=act_inplace)

    def forward(self, x):
        y = self.conv(x)
        return y

class COTEREBasicBlock(nn.Module):
    
    def __init__(self, 
                 input_filters, 
                 output_filters,
                 input_seq_length=12,
                 cotere_type="CTSR",
                 cotere_c_mlp_r=1, 
                 cotere_t_mlp_r=1, 
                 cotere_s_conv_size=1, 
                 cotere_embed_pos="A",
                 use_temp_conv=False,
                 down_sampling=False, 
                 block_type='3d',
                 norm_module=nn.BatchNorm3d,
                 freeze_bn=False):
        
        super(COTEREBasicBlock, self).__init__()
        
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.cotere_type = cotere_type
        self.down_sampling = down_sampling
        self.block = nn.ModuleDict()
        
        build_basic_block(self.block, 
                          input_filters, 
                          output_filters, 
                          use_temp_conv,
                          down_sampling, 
                          block_type, 
                          norm_module, 
                          freeze_bn)
        
        if self.cotere_type != 'NONE':
            self.cotere_embed_pos = cotere_embed_pos
            if self.cotere_embed_pos == "B":
                self.cotere = COTERE(input_filters, input_filters, input_seq_length, 
                                     cotere_type, cotere_c_mlp_r, cotere_t_mlp_r, cotere_s_conv_size, 
                                     norm_module=norm_module, freeze_bn=freeze_bn)
            else:
                self.cotere = COTERE(output_filters, output_filters, input_seq_length, 
                                     cotere_type, cotere_c_mlp_r, cotere_t_mlp_r, cotere_s_conv_size, 
                                     norm_module=norm_module, freeze_bn=freeze_bn)
     
        for m in self.modules():
            init_module_weights_new(m)
    
    def forward(self, x):
        residual = x
        out = x
        
        if self.cotere_type != 'NONE' and self.cotere_embed_pos == "B":
            out = self.cotere(out)
        
        for k, v in self.block.items():
            if "shortcut" in k:
                break
            out = v(out)

            if k == "bn2" and self.cotere_type != 'NONE' and self.cotere_embed_pos == "A":
                out = self.cotere(out)

        if (self.input_filters != self.output_filters) or self.down_sampling:
            residual = self.block["shortcut"](residual)
            residual = self.block["shortcut_bn"](residual)
        
        if self.cotere_type != 'NONE' and self.cotere_embed_pos == "D":
            residual = self.cotere(residual)
            
        out += residual
        out = self.block["relu"](out)
        
        if self.cotere_type != 'NONE' and self.cotere_embed_pos == "C":
            out = self.cotere(out)

        return out

class COTEREBottleneckBlock(nn.Module):
    
    def __init__(self, 
                 input_filters, 
                 output_filters, 
                 base_filters, 
                 input_seq_length=12,
                 cotere_type="CTSR",
                 cotere_c_mlp_r=1, 
                 cotere_t_mlp_r=1, 
                 cotere_s_conv_size=1, 
                 cotere_embed_pos="A",
                 use_temp_conv=True,
                 down_sampling=False, 
                 block_type='i3d', 
                 norm_module=nn.BatchNorm3d,
                 freeze_bn=False):
        
        super(COTEREBottleneckBlock, self).__init__()
        
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.base_filters = base_filters
        self.cotere_type = cotere_type
        self.down_sampling = down_sampling
        self.block = nn.ModuleDict()
        
        build_bottleneck(self.block, 
                          input_filters, 
                          output_filters, 
                          base_filters, 
                          use_temp_conv, 
                          down_sampling, 
                          block_type, 
                          norm_module, 
                          freeze_bn)
        
        if self.cotere_type != 'NONE':
            self.cotere_embed_pos = cotere_embed_pos
            if self.cotere_embed_pos == "B":
                self.cotere = COTERE(input_filters, input_filters, input_seq_length, 
                                     cotere_type, cotere_c_mlp_r, cotere_t_mlp_r, cotere_s_conv_size, 
                                     norm_module=norm_module, freeze_bn=freeze_bn)
            else:
                self.cotere = COTERE(output_filters, output_filters, input_seq_length, 
                                     cotere_type, cotere_c_mlp_r, cotere_t_mlp_r, cotere_s_conv_size, 
                                     norm_module=norm_module, freeze_bn=freeze_bn)
        
        for m in self.modules():
            init_module_weights_new(m)
    
    def forward(self, x):
        residual = x
        out = x
        
        if self.cotere_type != 'NONE' and self.cotere_embed_pos == "B":
            out = self.cotere(out)
        
        for k, v in self.block.items():
            if "shortcut" in k:
                break
            out = v(out)

            if k == "bn3" and self.cotere_type != 'NONE' and self.cotere_embed_pos == "A":
                out = self.cotere(out)

        if (self.input_filters != self.output_filters) or self.down_sampling:
            residual = self.block["shortcut"](residual)
            residual = self.block["shortcut_bn"](residual)
        
        if self.cotere_type != 'NONE' and self.cotere_embed_pos == "D":
            residual = self.cotere(residual)
            
        out += residual
        out = self.block["relu1"](out)
        
        if self.cotere_type != 'NONE' and self.cotere_embed_pos == "C":
            out = self.cotere(out)

        return out
