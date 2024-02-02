import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum


class MR_Graph(nn.Module):
    def __init__(self,
                 node_size,
                 rel_size,
                 triple_size,
                 node_dim,
                 depth=1,
                 attn_heads=1,
                 attn_heads_reduction='concat',
                 use_bias=False):
        super(MR_Graph, self).__init__()

        self.node_size = node_size
        self.rel_size = rel_size
        self.triple_size = triple_size
        self.node_dim = node_dim
        self.attn_heads = attn_heads
        self.attn_heads_reduction = attn_heads_reduction
        self.activation = torch.nn.Tanh()
        self.use_bias = use_bias
        self.depth = depth
        self.attn_kernels = nn.ParameterList()
        self.r_head = 4

        self.dropout = nn.Dropout(p=0.2)


        for l in range(self.depth * self.r_head):
            attn_kernel = torch.nn.Parameter(data=torch.empty(self.node_dim, 1, dtype=torch.float32))
            torch.nn.init.xavier_uniform_(attn_kernel)
            self.attn_kernels.append(attn_kernel)

    def forward(self, inputs):
        outputs = []
        features = inputs[0]
        rel_emb = inputs[1]
        adj = inputs[2]
        r_index = inputs[3]
        r_val = inputs[4]
        # import pickle
        # with open('r_index.pickle', 'wb') as f:
        #     pickle.dump(r_index, f)

        features = self.activation(features)
        outputs.append(features)
        # self.triple_size = r_index.shape[1]
        for l in range(self.depth):
            # attention_kernel = self.attn_kernels[l]
            # matrix shape: [N_tri x N_rel]

            tri_rel = torch.sparse_coo_tensor(indices=r_index, values=r_val,
                                              size=[self.triple_size, self.rel_size], dtype=torch.float32)
            # shape: [N_tri x dim]
            tri_rel = torch.sparse.mm(tri_rel, rel_emb)
            # shape: [N_tri x dim]
            neighs = features[adj[1, :].long()]

            tri_rel = F.normalize(tri_rel, dim=1, p=2)
            neighs = neighs - 2*torch.sum(neighs*tri_rel, dim=1, keepdim=True)*tri_rel

            # att = torch.squeeze(torch.mm(tri_rel, self.attention_kernel), dim=-1)
            att_list = []
            for r_num in range(self.r_head):
                att = torch.squeeze(torch.mm(tri_rel, self.attn_kernels[(l+1) * (r_num+1) - 1]), dim=-1)
                att_list.append(att)
            # 将张量列表转换为 PyTorch 的张量
            tensor_stack = torch.stack(att_list)

            # 计算每个位置上的平均值
            att = torch.mean(tensor_stack, dim=0)
            att = torch.sparse_coo_tensor(indices=adj, values=att, size=[self.node_size, self.node_size])
            att = torch.sparse.softmax(att, dim=1)

            new_features = scatter_sum(src=neighs * torch.unsqueeze(att.coalesce().values(), dim=-1), dim=0,
                                       index=adj[0, :].long())

            features = self.activation(new_features)
            outputs.append(features)

        outputs1 = torch.cat(outputs, dim=-1)

        return outputs1
