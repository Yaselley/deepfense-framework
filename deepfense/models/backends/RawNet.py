# code from https://github.com/clovaai/aasist/blob/main/models/RawNet2Spoof.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from deepfense.models.backends.registry import register_backend


class Residual_block(nn.Module):
    def __init__(self, nb_filts, pool, first=False):
        super(Residual_block, self).__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm1d(num_features=nb_filts[0])

        self.lrelu = nn.LeakyReLU(negative_slope=0.3)

        self.conv1 = nn.Conv1d(
            in_channels=nb_filts[0],
            out_channels=nb_filts[1],
            kernel_size=3,
            padding=1,
            stride=1,
        )

        self.bn2 = nn.BatchNorm1d(num_features=nb_filts[1])
        self.conv2 = nn.Conv1d(
            in_channels=nb_filts[1],
            out_channels=nb_filts[1],
            padding=1,
            kernel_size=3,
            stride=1,
        )

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(
                in_channels=nb_filts[0],
                out_channels=nb_filts[1],
                padding=0,
                kernel_size=1,
                stride=1,
            )

        else:
            self.downsample = False
        self.mp = nn.MaxPool1d(pool)

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu(out)
        else:
            out = x

        out = self.conv1(x)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        return out


@register_module("rawnet")
class RawNet(nn.Module):
    def __init__(self, d_args):
        super().__init__()

        pool = d_args.get("pool", 3)

        self.first_bn = nn.BatchNorm1d(num_features=d_args["filts"][0])
        self.selu = nn.SELU(inplace=True)
        self.block0 = nn.Sequential(
            Residual_block(nb_filts=d_args["filts"][1], pool=pool, first=True))
        self.block1 = nn.Sequential(
            Residual_block(nb_filts=d_args["filts"][1], pool=pool))
        self.block2 = nn.Sequential(
            Residual_block(nb_filts=d_args["filts"][2], pool=pool))
        d_args["filts"][2][0] = d_args["filts"][2][1]
        self.block3 = nn.Sequential(
            Residual_block(nb_filts=d_args["filts"][2], pool=pool))
        self.block4 = nn.Sequential(
            Residual_block(nb_filts=d_args["filts"][2], pool=pool))
        self.block5 = nn.Sequential(
            Residual_block(nb_filts=d_args["filts"][2], pool=pool))
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc_attention0 = self._make_attention_fc(
            in_features=d_args["filts"][1][-1],
            l_out_features=d_args["filts"][1][-1])
        self.fc_attention1 = self._make_attention_fc(
            in_features=d_args["filts"][1][-1],
            l_out_features=d_args["filts"][1][-1])
        self.fc_attention2 = self._make_attention_fc(
            in_features=d_args["filts"][2][-1],
            l_out_features=d_args["filts"][2][-1])
        self.fc_attention3 = self._make_attention_fc(
            in_features=d_args["filts"][2][-1],
            l_out_features=d_args["filts"][2][-1])
        self.fc_attention4 = self._make_attention_fc(
            in_features=d_args["filts"][2][-1],
            l_out_features=d_args["filts"][2][-1])
        self.fc_attention5 = self._make_attention_fc(
            in_features=d_args["filts"][2][-1],
            l_out_features=d_args["filts"][2][-1])

        self.bn_before_gru = nn.BatchNorm1d(
            num_features=d_args["filts"][2][-1])
        self.gru = nn.GRU(
            input_size=d_args["filts"][2][-1],
            hidden_size=d_args["gru_node"],
            num_layers=d_args["nb_gru_layer"],
            batch_first=True,
        )

        self.fc1_gru = nn.Linear(in_features=d_args["gru_node"],
                                 out_features=d_args["nb_fc_node"])

        self.outdim = d_args["nb_fc_node"]

    def get_outdim(self,):
        return self.outdim

    def forward(self, x, Freq_aug=None):

        nb_samp = x.shape[0]
        len_seq = x.shape[1]
        x = x.view(nb_samp, 1, len_seq)

        x = self.Sinc_conv(x)
        x = F.max_pool1d(torch.abs(x), 3)
        x = self.first_bn(x)
        x = self.selu(x)

        x0 = self.block0(x)
        y0 = self.avgpool(x0).view(x0.size(0),
                                   -1)  # torch.Size([batch, filter])
        y0 = self.fc_attention0(y0)
        y0 = self.sig(y0).view(y0.size(0), y0.size(1),
                               -1)  # torch.Size([batch, filter, 1])
        x = x0 * y0 + y0  # (batch, filter, time) x (batch, filter, 1)

        x1 = self.block1(x)
        y1 = self.avgpool(x1).view(x1.size(0),
                                   -1)  # torch.Size([batch, filter])
        y1 = self.fc_attention1(y1)
        y1 = self.sig(y1).view(y1.size(0), y1.size(1),
                               -1)  # torch.Size([batch, filter, 1])
        x = x1 * y1 + y1  # (batch, filter, time) x (batch, filter, 1)

        x2 = self.block2(x)
        y2 = self.avgpool(x2).view(x2.size(0),
                                   -1)  # torch.Size([batch, filter])
        y2 = self.fc_attention2(y2)
        y2 = self.sig(y2).view(y2.size(0), y2.size(1),
                               -1)  # torch.Size([batch, filter, 1])
        x = x2 * y2 + y2  # (batch, filter, time) x (batch, filter, 1)

        x3 = self.block3(x)
        y3 = self.avgpool(x3).view(x3.size(0),
                                   -1)  # torch.Size([batch, filter])
        y3 = self.fc_attention3(y3)
        y3 = self.sig(y3).view(y3.size(0), y3.size(1),
                               -1)  # torch.Size([batch, filter, 1])
        x = x3 * y3 + y3  # (batch, filter, time) x (batch, filter, 1)

        x4 = self.block4(x)
        y4 = self.avgpool(x4).view(x4.size(0),
                                   -1)  # torch.Size([batch, filter])
        y4 = self.fc_attention4(y4)
        y4 = self.sig(y4).view(y4.size(0), y4.size(1),
                               -1)  # torch.Size([batch, filter, 1])
        x = x4 * y4 + y4  # (batch, filter, time) x (batch, filter, 1)

        x5 = self.block5(x)
        y5 = self.avgpool(x5).view(x5.size(0),
                                   -1)  # torch.Size([batch, filter])
        y5 = self.fc_attention5(y5)
        y5 = self.sig(y5).view(y5.size(0), y5.size(1),
                               -1)  # torch.Size([batch, filter, 1])
        x = x5 * y5 + y5  # (batch, filter, time) x (batch, filter, 1)

        x = self.bn_before_gru(x)
        x = self.selu(x)
        x = x.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:, -1, :]
        last_hidden = self.fc1_gru(x)

        return last_hidden

    def _make_attention_fc(self, in_features, l_out_features):

        l_fc = []

        l_fc.append(
            nn.Linear(in_features=in_features, out_features=l_out_features))

        return nn.Sequential(*l_fc)

    def _make_layer(self, nb_blocks, nb_filts, first=False):
        layers = []
        # def __init__(self, nb_filts, first = False):
        for i in range(nb_blocks):
            first = first if i == 0 else False
            layers.append(Residual_block(nb_filts=nb_filts, first=first))
            if i == 0:
                nb_filts[0] = nb_filts[1]

        return nn.Sequential(*layers)