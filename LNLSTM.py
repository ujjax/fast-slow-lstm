from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

import helper
import config

class LN_LSTMCell(object):
    """docstring for LN_LSTMCell"""
    def __init__(self,num_units, use_zoneout, is_training,
                 zoneout_keep_h, zoneout_keep_c, f_bias = 0.5):
        super(LN_LSTMCell, self).__init__()

        self.num_units = num_units
        self.f_bias = f_bias

        self.use_zoneout  = use_zoneout
        self.zoneout_keep_h = zoneout_keep_h
        self.zoneout_keep_c = zoneout_keep_c

        self.is_training = is_training

    def __call__(self, x ,state):
        h, c = state
        h_size = self.num_units
        x_size = int(x.size()[1])
        

        W_xh = helper.orthogonal_initializer([x_size, 4 * h_size] , scale = 1.0)

        W_hh = helper.orthogonal_initializer([h_size, 4 * h_size] , scale = 1.0)

        bias = torch.zeros([4*h_size])
        
        print(x.shape, 'shape of x')
        print(h.shape, 'shape of h')
        
        
        W_full = np.concatenate((W_xh,W_hh),axis = 0)
        concat = np.concatenate((x,h), axis = 1)

        concat = torch.mm(concat,W_full) + bias
        concat = helper.layer_norm_all(concat, 4, h_size)

        i,j,f,o = torch.split(tensor = concat, split_size = int(concat.size()[1])//4, dim=1)

        new_c = c * F.sigmoid(f + self.f_bias) + F.sigmoid(i) * F.tanh(j)
        new_h = F.tanh(helper.layer_norm(new_c)) * F.sigmoid(o)

        if self.use_zoneout:
            new_h, new_c = helper.zoneout(new_h, new_c, h, c, self.zoneout_keep_h,
                                          self.zoneout_keep_c, self.is_training)

        return new_h, (new_h, new_c)


    def zero_state(self, batch_size, dtype):
        h = torch.zeros([batch_size, self.num_units]).type(dtype)
        c = torch.zeros([batch_size, self.num_units]).type(dtype)
        return (h, c)

def repackage_hidden(h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(repackage_hidden(v) for v in h)