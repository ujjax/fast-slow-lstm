from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import LNLSTM
import FSRNN

import helper


class PTB_Model(object):
	def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(
            data, batch_size, num_steps, name=name)

        F_size = config.cell_size
        S_size = config.hyper_size

        num_steps = input_.num_steps
        emb_size = config.embed_size

        self.embedding = nn.Embedding(self.input_size, self.hidden_size)

        F_cells = [LNLSTM.LN_LSTMCell(F_size, use_zoneout=True, is_training=is_training,
                                      zoneout_keep_h=config.zoneout_h, zoneout_keep_c=config.zoneout_c)
                   for _ in range(config.fast_layers)]

       	S_cell  = LNLSTM.LN_LSTMCell(S_size, use_zoneout=True, is_training=is_training,
                                     zoneout_keep_h=config.zoneout_h, zoneout_keep_c=config.zoneout_c)


       	FS_cell = FSRNN.FSRNNCell(F_cells, S_cell, config.keep_prob, is_training)

       	self._initial_state = FS_cell.zero_state(batch_size, torch.FloatTensor)

       	state = self._initial_state
       	outputs = []
       	for time_step in range(num_steps):
       		out , state = FS_cell(inputs[:,time_step,:],state)
       		outputs.append(out)

       	output = torch.cat(outputs,dim =1).view([-1,F_size])
       	
       	softmax_w = helper.orthogonal_initializer([F_size, vocab_size])
       	softmax_b = helper.orthogonal_initializer([vocab_size])

       	logits = torch.mm(output , softmax_w) + softmax_b

       	loss = 


       	self._cost = cost = nn.mean(loss) / batch_size

        self._final_state = state

        if not is_training: return




class PTBInput(object):
    """The input data."""

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(
            data, batch_size, num_steps, name=name)

class PTBModel(object):
	"""docstring for PTBModel"""
	def __init__(self, arg):
		super(PTBModel, self).__init__()
		
		self._input = input_

        batch_size = input_.batch_size
        num_steps = input_.num_steps
        emb_size = config.embed_size
        F_size = config.cell_size
        S_size = config.hyper_size
        vocab_size = config.vocab_size



