from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import LNLSTM
import FSRNN

import reader
import config

import time
import numpy as np

criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
args = config.get_config()


class PTB_Model(nn.Module):
    def __init__(self, embedding_dim=args.hidden_size, num_steps=args.num_steps, batch_size=args.batch_size,
                  vocab_size=args.vocab_size, num_layers=args.num_layers, dp_keep_prob=args.keep_prob,name=None):
        super(PTB_Model, self).__init__()
        self.batch_size = batch_size  
        self.num_steps = num_steps 
        self.vocab_size = vocab_size

        self.F_size = args.cell_size
        self.S_size = args.hyper_size

        self.num_steps = num_steps
        self.emb_size = embedding_dim
        self.is_train = False

        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)

        self.F_cells = [LNLSTM.LN_LSTMCell(self.F_size, use_zoneout=True, is_training=self.is_train,
                                           zoneout_keep_h=args.zoneout_h, zoneout_keep_c=args.zoneout_c)
                        for _ in range(args.fast_layers)]

        self.S_cell  = LNLSTM.LN_LSTMCell(self.S_size, use_zoneout=True, is_training=self.is_train,
                                          zoneout_keep_h=args.zoneout_h, zoneout_keep_c=args.zoneout_c)


        self.FS_cell = FSRNN.FSRNNCell(self.F_cells, self.S_cell, args.keep_prob, self.is_train)

        self._initial_state = self.FS_cell.zero_state(batch_size, torch.FloatTensor)

    def forward(self,inputs):
        state = self._initial_state
        outputs = []
        for time_step in range(self.num_steps):
            out , state = self.FS_cell(inputs[:,time_step,:],state)
            outputs.append(out)

        output = torch.cat(outputs,dim =1).view([-1,self.F_size])

        softmax_w = helper.orthogonal_initializer([self.F_size, self.vocab_size])
        softmax_b = helper.orthogonal_initializer([self.vocab_size])

        logits = torch.mm(output , softmax_w) + softmax_b

        return logits.view([self.num_steps,self.batch_size,self.vocab_size]), state


def run_epoch(model, data, is_train=False, lr=1.0):
    """Runs the model on the given data."""
    if is_train:
        model.is_train = True
    else:
        model.eval()
    
    epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps
    start_time = time.time()
    #hidden = model.init_hidden()
    costs = 0.0
    iters = 0.0

    for step, (x, y) in enumerate(reader.ptb_iterator(data, model.batch_size, model.num_steps)):
        inputs = Variable(torch.from_numpy(x.astype(np.int64)).transpose(0, 1).contiguous()).cuda()
        model.zero_grad()
        #hidden = repackage_hidden(hidden)
        outputs, hidden = model(inputs)
        targets = Variable(torch.from_numpy(y.astype(np.int64)).transpose(0, 1).contiguous()).cuda()
        tt = torch.squeeze(targets.view(-1, model.batch_size * model.num_steps))

        loss = criterion(outputs.view(-1, model.vocab_size), tt)
        costs += loss.data[0] * model.num_steps
        iters += model.num_steps

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
            for p in model.parameters():
                p.data.add_(-lr, p.grad.data)
            if step % (epoch_size // 10) == 10:
                print("{} perplexity: {:8.2f} speed: {} wps".format(step * 1.0 / epoch_size, np.exp(costs / iters),
                                  iters * model.batch_size / (time.time() - start_time)))
    return np.exp(costs / iters)

if __name__ == "__main__":
    raw_data = reader.ptb_raw_data(data_path=args.data_path)
    train_data, valid_data, test_data, word_to_id, id_to_word = raw_data
    vocab_size = len(word_to_id)
    print('Vocabluary size: {}'.format(vocab_size))
    model = PTB_Model(embedding_dim=args.hidden_size, num_steps=args.num_steps, batch_size=args.batch_size,
                      vocab_size=vocab_size, num_layers=args.num_layers, dp_keep_prob=args.keep_prob)
    model.cuda()
    lr = args.lr_start
    # decay factor for learning rate
    lr_decay_base = args.lr_decay_rate
    # we will not touch lr for the first m_flat_lr epochs
    m_flat_lr = 14.0

    print("########## Training ##########################")

    for epoch in range(args.max_max_epoch):
        lr_decay = lr_decay_base ** max(epoch - m_flat_lr, 0)
        lr = lr * lr_decay # decay lr if it is time
        train_p = run_epoch(model, train_data, True, lr)
        print('Train perplexity at epoch {}: {:8.2f}'.format(epoch, train_p))
        print('Validation perplexity at epoch {}: {:8.2f}'.format(epoch, run_epoch(model, valid_data)))


    print("########## Testing ##########################")
    model.batch_size = 1 # to make sure we process all the data
    print('Test Perplexity: {:8.2f}'.format(run_epoch(model, test_data)))
    with open(args.save, 'wb') as f:
        torch.save(model, f)
    print("########## Done! ##########################")

