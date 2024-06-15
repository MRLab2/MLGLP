

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.nn as gnn
from torch.autograd import Variable

from torch_geometric.data import DataLoader
from tqdm import tqdm

class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_size, latent_dim=[32, 32, 32], with_dropout=False):
        super(Net, self).__init__()
        conv = gnn.GCNConv  # SplineConv  NNConv   GraphConv   SAGEConv
        self.latent_dim = latent_dim
        self.conv_params = nn.ModuleList()
        self.conv_params.append(conv(input_dim, latent_dim[0], cached=False))
        for i in range(1, len(latent_dim)):
            self.conv_params.append(conv(latent_dim[i-1], latent_dim[i], cached=False))
            
        latent_dim = sum(latent_dim)
            
        self.linear1 = nn.Linear(latent_dim*3, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 2)
        
        self.with_dropout = with_dropout
        
        

    def forward(self, data):
        data.to(torch.device("cuda"))
        all_cur_message_layers = []
        batch_size = len(data.y)
        for j in range(3):
            y = data.y
            if(j==0):
               batch = data.batch
               x, edge_index = data.x, data.edge_index
            elif(j==1):
                data1loader = DataLoader(data.sg1, batch_size=batch_size, shuffle=False)
                batch_iterator = iter(data1loader)
                data1 = next(batch_iterator)
                # Access batch attributes
                batch = data1.batch
                x, edge_index = data1.x, data1.edge_index
            else:
                data2loader = DataLoader(data.sg2, batch_size=batch_size, shuffle=False)
                batch_iterator = iter(data2loader)
                data2 = next(batch_iterator)

                # Access batch attributes
                batch = data2.batch
                x, edge_index = data2.x, data2.edge_index
            cur_message_layer = x
            cat_message_layers = []
            lv = 0
            while lv < len(self.latent_dim):
                cur_message_layer = self.conv_params[lv](cur_message_layer, edge_index)  
                cur_message_layer = torch.tanh(cur_message_layer)
                cat_message_layers.append(cur_message_layer)
                lv += 1
            cur_message_layer = torch.cat(cat_message_layers, 1)
            batch_idx = torch.unique(batch)
            idx = []
            for i in batch_idx:
                idx.append((batch==i).nonzero()[0].cpu().numpy()[0])
            cur_message_layer = cur_message_layer[idx,:]
            all_cur_message_layers.append(cur_message_layer)

        concatenated_cur_message_layers = torch.cat(all_cur_message_layers, dim=1)

        hidden = self.linear1(concatenated_cur_message_layers)

        self.feature = hidden
        hidden = F.relu(hidden)
        
        if self.with_dropout:
            hidden = F.dropout(hidden, training=self.training)
            
        logits = self.linear2(hidden)
        logits = F.log_softmax(logits, dim=1)
        
        if y is not None:
            y = Variable(y)
            loss = F.nll_loss(logits, y)

            pred = logits.data.max(1, keepdim=True)[1]
            acc = pred.eq(y.data.view_as(pred)).cpu().sum().item() / float(y.size()[0])
            return logits, loss, acc, self.feature
        else:
            return logits
        