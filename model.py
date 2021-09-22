import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np

import dgl
import dgl.function as fn

class TreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(2 * h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(2 * h_size, 2 * h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        # concatenate h_jl for equation (1), (2), (3), (4)
        h_cat = nodes.mailbox['h'].view(nodes.mailbox['h'].size(0), -1)
        # equation (2)
        f = th.sigmoid(self.U_f(h_cat)).view(*nodes.mailbox['h'].size())
        # second term of equation (5)
        c = th.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_cat), 'c': c}

    def apply_node_func(self, nodes):
        # equation (1), (3), (4)
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        # equation (5)
        c = i * u + nodes.data['c']
        # equation (6)
        h = o * th.tanh(c)
        return {'h' : h, 'c' : c}

class Attention(nn.Module):
        def __init__(self, embed_size, feature_size):
            super(Attention, self).__init__()
            
            self.embed_size = embed_size
            self.feature_size = feature_size
            
            self.linear_in = nn.Linear(feature_size, embed_size, bias=False)
            self.linear_out = nn.Linear(embed_size+feature_size, embed_size)
            
        def forward(self, sent, img, mask):
            # sent: snetence_len * embed_size
            # img: num_region * feature_size
            snetence_len = sent.size(0)
            num_region = img.size(0)
            
            # img_in: num_region * embed_size
            img_in = self.linear_in(img)
            
            atten = th.mm(sent, img_in.transpose(0, 1))
            #atten.data.masked_fill(mask, -1e6)
            atten = F.softmax(atten, dim=1)
            # atten: snetence_len * num_region
            context = th.mm(atten, img)
            # context: snetence_len * feature_size
            output = th.cat((context, sent), dim=1) # output: snetence_len * (feature_size+embed_size)
            output = th.tanh(self.linear_out(output))
            # output: snetence_len * embed_size
            return output

class TreeLSTM(nn.Module):
    def __init__(self,
                 num_vocabs,
                 x_size,
                 h_size,
                 feature_size, 
                 num_classes,
                 dropout,
                 pretrained_emb=None):
        super(TreeLSTM, self).__init__()
        self.x_size = x_size
        self.embedding = nn.Embedding(num_vocabs, x_size)
        self.attention = Attention(x_size, feature_size)
        if pretrained_emb is not None:
            print('Using glove')
            self.embedding.from_pretrained(pretrained_emb)
            self.embedding.weight.requires_grad = True
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(h_size, num_classes)
        self.cell = TreeLSTMCell(x_size, h_size)

    def forward(self, batch, h, c):
        """Compute tree-lstm prediction given a batch.

        Parameters
        ----------
        batch : dgl.data.SSTBatch
            The data batch.
        h : Tensor
            Initial hidden state.
        c : Tensor
            Initial cell state.

        Returns
        -------
        logits : Tensor
            The prediction of each node.
        """
        g = batch.graph
        # to heterogenous graph
        g = dgl.graph(g.edges())
        # feed embedding
        embeds = self.embedding(batch.wordid * batch.mask)
        attn_mask = batch.mask.expand(batch.image.shape[0], batch.wordid.shape[0]).T
        attn_embeds = self.attention(embeds, batch.image, attn_mask)
        g.ndata['iou'] = self.cell.W_iou(self.dropout(attn_embeds)) * batch.mask.float().unsqueeze(-1)
        g.ndata['h'] = h
        g.ndata['c'] = c
        # propagate
        dgl.prop_nodes_topo(g,
                            message_func=self.cell.message_func,
                            reduce_func=self.cell.reduce_func,
                            apply_node_func=self.cell.apply_node_func)
        # compute logits
        h = self.dropout(g.ndata.pop('h'))
        logits = self.linear(h)
        return logits
