# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------
Copyright 2022 Benjamin Alexander Albert [Karchin Lab]
All Rights Reserved

BigMHC Academic License

bigmhc.py
------------------------------------------------------------------------
"""

import os
import torch

from collections import OrderedDict


class AttentionLSTMCell(torch.nn.Module):

    def __init__(self, inp, out):
        super().__init__()
        self.att = torch.nn.MultiheadAttention(
            embed_dim=inp,
            num_heads=1,
            batch_first=True)
        self.lstm = torch.nn.LSTM(
            input_size=inp,
            hidden_size=out,
            batch_first=True,
            bidirectional=True)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x,_ = self.att(
            query=x,
            key=x,
            value=x,
            need_weights=False)
        y,_ = self.lstm(x)
        return y


class Dense(torch.nn.Module):

    def __init__(self, inp, out, act, drp):
        super().__init__()
        self.linear = torch.nn.Linear(inp, out)
        self.dropout = torch.nn.Dropout(drp)
        self.act = act

    def forward(self, x):
        return torch.cat((x,self.dropout(self.act(self.linear(x)))),-1)


class DenseBlock(torch.nn.Module):
    def __init__(self, inp, out, act, drp=0.5, layers=1):
        super().__init__()
        self.denseLayers = torch.nn.ModuleList(
            [Dense(
                inp=inp+(x*out),
                out=out,
                act=act,
                drp=drp) for x in range(layers)])

    def forward(self, x):
        for dense in self.denseLayers:
            x = dense(x)
        return x


class BigMHC(torch.nn.Module):

    class AnchorBlock(torch.nn.Module):
        def __init__(self, mhclen, minpep, enclen, hidlen, layers):
            super().__init__()
            self.enclen = enclen
            self.minpep = minpep
            self.denseBlock = DenseBlock(
                inp=mhclen + (minpep * enclen),
                out=hidlen,
                act=torch.nn.Tanh(),
                layers=layers)

        def buildInput(self, mhc, pep):
            return torch.cat(
                (mhc,
                 pep[:, :(self.minpep//2) * self.enclen],
                 pep[:, -(self.minpep - (self.minpep//2)) * self.enclen:]), 1)

        def forward(self, mhc, pep):
            return self.denseBlock(self.buildInput(mhc, pep))

    class PMHCLSTM(torch.nn.Module):
        def __init__(self, mhclen, minpep, enclen, hidlen, ncells):
            super().__init__()
            self.minpep = minpep
            self.enclen = enclen
            self.narrowCells = torch.nn.ModuleList(
                [AttentionLSTMCell(
                    inp=2*hidlen*x + mhclen + enclen,
                    out=hidlen) for x in range(ncells)])
            self.wideCells  = torch.nn.ModuleList(
                [AttentionLSTMCell(
                    inp=2*hidlen*x + mhclen + (minpep*enclen),
                    out=hidlen) for x in range(ncells)])
    
        def buildInput(self, mhc, pep, win):
            window = win * self.enclen
            slices = (pep.shape[1] - window) // self.enclen + 1
            output = torch.zeros(
                size=(mhc.shape[0], slices, mhc.shape[1] + window),
                dtype=torch.float32,
                device=mhc.device)
            output[:, :, window:] = torch.unsqueeze(mhc, 1)
            for i in range(slices):
                output[:, i, :window] = pep[:, i * self.enclen:i * self.enclen + window]
            return output

        def forward(self, mhc, pep):
            narrowinp = self.buildInput(mhc, pep, 1)
            wideinp   = self.buildInput(mhc, pep, self.minpep)
            for idx in range(len(self.narrowCells)):
                narrowout = self.narrowCells[idx](narrowinp)
                wideout   = self.wideCells[idx](wideinp)
                if idx < len(self.narrowCells)-1:
                    narrowinp = torch.cat((narrowinp, narrowout), -1)
                    wideinp   = torch.cat((wideinp,   wideout  ), -1)
            return torch.cat((narrowout[:,-1,:], wideout[:,-1,:]),-1)

    class Condenser(torch.nn.Module):
        def __init__(self, mhclen, minpep, enclen, hidlen, layers):
            super().__init__()
            self.enclen = enclen
            self.act = torch.nn.Tanh()
            inp = mhclen + (minpep*enclen) + hidlen*(4+layers)
            self.preAttentionDenseBlock = DenseBlock(
                inp=inp,
                out=hidlen,
                act=self.act,
                layers=layers)
            self.att = torch.nn.Linear(inp + hidlen*layers, mhclen)
            self.out = torch.nn.Linear(mhclen, 1)

        def forward(self, mhc, anchorOutput, lstmOutput):
            attention = self.att(self.preAttentionDenseBlock(
                torch.cat((anchorOutput, lstmOutput), -1)))
            attention = torch.masked_select(
                self.out.weight * attention,
                mhc.bool()).reshape(mhc.shape[0],-1)
            return \
                (torch.sigmoid(
                    torch.sum(attention, dim=1) + self.out.bias),
                attention)

    def __init__(
            self,
            mhclen=388,
            minpep=8,
            enclen=20,
            hidlen=512,
            layers=2):

        super().__init__()

        self.lstm = BigMHC.PMHCLSTM(
            mhclen=mhclen,
            minpep=minpep,
            enclen=enclen,
            hidlen=hidlen,
            ncells=layers)
        self.anchorBlock = BigMHC.AnchorBlock(
            mhclen=mhclen,
            minpep=minpep,
            enclen=enclen,
            hidlen=hidlen,
            layers=layers)
        self.condenser = BigMHC.Condenser(
            mhclen=mhclen,
            minpep=minpep,
            enclen=enclen,
            hidlen=hidlen,
            layers=layers)
    
    def forward(self, mhc, pep):
        mhc = mhc.float()
        pep = pep.float()
        return self.condenser(
            mhc=mhc,
            anchorOutput=self.anchorBlock(mhc,pep),
            lstmOutput=self.lstm(mhc,pep))

    def save(self, fp):
        if not os.path.exists(fp):
            os.makedirs(fp)
        state = self.state_dict()
        for idx, (k,v) in enumerate(self.state_dict().items()):
            torch.save(v, os.path.join(fp, "{}_{}".format(idx,k)))

    @staticmethod
    def load(fp):
        state = []
        for lyr in sorted(os.listdir(fp)):
            name = lyr[lyr.index('_')+1:]
            state.append((name, torch.load(os.path.join(fp,lyr))))
        model = BigMHC()
        model.load_state_dict(OrderedDict(state))
        return model
