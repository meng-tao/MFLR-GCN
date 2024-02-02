import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# class MultiGraphConvLayer(nn.Module):
#     """ A GCN module operated on dependency graphs. """
#
#     def __init__(self, mem_dim, layers, heads, gcn_dropout):
#         super(MultiGraphConvLayer, self).__init__()
#         self.mem_dim = mem_dim
#         self.layers = layers
#         self.head_dim = self.mem_dim // self.layers
#         self.heads = heads
#         # dcgcn layer
#         self.weight_list = nn.ModuleList()
#         for j in range(self.layers):
#             self.weight_list.append(nn.Linear(self.mem_dim + self.head_dim * j, self.head_dim))
#         self.activation = torch.nn.Tanh()
#         self.dropout = nn.Dropout(p=gcn_dropout)
#
#     def forward(self, adj, gcn_inputs):
#         import pickle
#         import numpy as np
#         with open('/MLFM-GCN/pyrwr/new_tri.pickle', 'rb') as f:
#             new_tri = pickle.load(f)
#         val, r, c = [], [], []
#         _gcn_inputs = gcn_inputs.cpu().detach().numpy()
#         for item in new_tri:
#             r.append(item[0])
#             c.append(item[1])
#             val.append(np.dot(_gcn_inputs[item[0]], _gcn_inputs[item[1]]))
#         row = torch.tensor(r, dtype=torch.long)
#         col = torch.tensor(c, dtype=torch.long)
#         value = torch.tensor(val, dtype=gcn_inputs.dtype, device=gcn_inputs.device)
#         sparse_adj_matrix = torch.sparse_coo_tensor(torch.stack([row, col]),
#                                                     value, (gcn_inputs.shape[0], gcn_inputs.shape[0]),
#                                                     device=gcn_inputs.device)
#         multi_head_list = []
#         gcn_inputs = self.activation(gcn_inputs)
#
#         sparse_adj_matrix = torch.sparse.softmax(sparse_adj_matrix, dim=1)
#
#         for i in range(self.heads):
#             outputs = gcn_inputs
#             output_list = []
#             for l in range(self.layers):
#                 AxW = self.weight_list[l](torch.sparse.mm(sparse_adj_matrix, outputs)) # self loop
#                 AxW = F.softmax(AxW)
#                 outputs = self.dropout(AxW)
#                 if l >= 1:
#                     outputs = torch.cat([outputs, AxW], dim=1)
#                 output_list.append(AxW)
#
#             gcn_ouputs = torch.cat(output_list, dim=1)
#             gcn_ouputs = gcn_ouputs + gcn_inputs
#
#             multi_head_list.append(gcn_ouputs)
#
#         # for i in range(self.heads):
#         #     outputs = gcn_inputs
#         #     output_list = []
#         #     for l in range(self.layers):
#         #         AxW = self.weight_list[l](torch.sparse.mm(sparse_adj_matrix, outputs))  # self loop
#         #         # AxW = F.softmax(AxW)
#         #         outputs = torch.cat([outputs, AxW], dim=1)
#         #         outputs = self.dropout(outputs)
#         #         output_list.append(outputs)
#         #
#         #     gcn_ouputs = torch.cat(output_list, dim=1)
#         #     # gcn_ouputs = gcn_ouputs + gcn_inputs
#         #     multi_head_list.append(gcn_ouputs)
#
#         final_output = torch.mean(torch.stack(multi_head_list), dim=0)
#         out = self.activation(final_output)
#
#         return out


class LSMGCN(nn.Module):
    def __init__(self, hidden_dim, num_layers, gcn_dropout, heads):
        super().__init__()
        self.mem_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.gcn_dropout = gcn_dropout

        # gcn layer
        self.mul_gcn = MultiGraphConvLayer(self.mem_dim, self.num_layers, self.heads, self.gcn_dropout)

    def forward(self, adj, inputs_emb):
        outputs = self.mul_gcn(adj, inputs_emb)
        return outputs



class MultiGraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, mem_dim, layers, heads, gcn_dropout):
        super(MultiGraphConvLayer, self).__init__()
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.heads = heads
        # dcgcn layer
        self.weight_list = nn.ModuleList()
        self.weight_list.append(nn.Linear(self.mem_dim, self.head_dim))
        for j in range(1, self.layers):
            self.weight_list.append(nn.Linear(self.head_dim * j, self.head_dim))
        self.activation = torch.nn.Tanh()
        self.dropout = nn.Dropout(p=gcn_dropout)

    def forward(self, adj, gcn_inputs):

        multi_head_list = []
        # gcn_inputs = self.activation(gcn_inputs)
        gcn_inputs = torch.sparse.mm(adj, gcn_inputs)
        # adj = torch.matmul(gcn_inputs, gcn_inputs.transpose(-1, -2))
        # adj = F.softmax(adj, dim=-1)

        for i in range(self.heads):
            outputs = gcn_inputs
            output_list = []
            for l in range(self.layers):
                AxW = self.weight_list[l](torch.sparse.mm(adj, outputs)) # self loop
                AxW = F.softmax(AxW)
                outputs = self.dropout(AxW)
                if l >= 1:
                    outputs = torch.cat([outputs, AxW], dim=1)
                output_list.append(AxW)

            gcn_ouputs = torch.cat(output_list, dim=1)
            gcn_ouputs = gcn_ouputs + gcn_inputs

            multi_head_list.append(gcn_ouputs)

        final_output = torch.mean(torch.stack(multi_head_list), dim=0)
        out = self.activation(final_output)

        return out
