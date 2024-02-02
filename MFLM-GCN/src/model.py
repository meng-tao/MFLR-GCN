import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn_layer import MR_Graph
from tabulate import tabulate
import logging
from torch_scatter import scatter_mean
from lsm_gcn import LSMGCN
import dgl
from dgl.nn.pytorch.conv import RelGraphConv
from dgl.nn.pytorch.conv import GATConv, GraphConv

class Encoder_Model(nn.Module):
    def __init__(self, node_hidden, rel_hidden, triple_size, node_size, rel_size, device, adj_matrix, r_index, r_val, rel_matrix, ent_matrix, values,
                 dropout_rate=0.0, ind_dropout_rate=0.0, gamma=3, lr=0.005, depth=2, hidden_dim=300, num_layers=1, heads=4):
        super(Encoder_Model, self).__init__()
        self.node_hidden = node_hidden
        self.node_size = node_size
        self.rel_size = rel_size
        self.triple_size = triple_size
        self.depth = depth
        self.device = device
        self.dropout = nn.Dropout(dropout_rate)
        self.ind_dropout = nn.Dropout(ind_dropout_rate)
        self.gamma = gamma
        self.lr = lr
        self.adj_list = adj_matrix.to(device)
        self.r_index = r_index.to(device)
        self.r_val = r_val.to(device)
        self.rel_adj = rel_matrix.to(device)
        self.ent_adj = ent_matrix.to(device)


        self.ent_embedding = nn.Embedding(node_size, node_hidden)
        self.rel_embedding = nn.Embedding(rel_size, rel_hidden)
        torch.nn.init.xavier_uniform_(self.ent_embedding.weight)
        torch.nn.init.xavier_uniform_(self.rel_embedding.weight)

        self.e_encoder = MR_Graph(node_size=self.node_size,
                                           rel_size=self.rel_size,
                                           triple_size=self.triple_size,
                                           node_dim=self.node_hidden,
                                           depth=self.depth,
                                           use_bias=True
                                           )
        self.r_encoder = MR_Graph(node_size=self.node_size,
                                           rel_size=self.rel_size,
                                           triple_size=self.triple_size,
                                           node_dim=self.node_hidden,
                                           depth=self.depth,
                                           use_bias=True
                                           )
        self.encoder = LSMGCN(hidden_dim, num_layers, dropout_rate, heads)
        self.values = values

    def avg(self, adj, emb, size: int):
        adj = torch.sparse_coo_tensor(indices=adj, values=torch.ones_like(adj[0, :], dtype=torch.float),
                                      size=[self.node_size, size])
        adj = torch.sparse.softmax(adj, dim=1)
        return adj, torch.sparse.mm(adj, emb)

    def gcn_forward(self):
        adj, ent_feature = self.avg(self.ent_adj, self.ent_embedding.weight, self.node_size)
        # Hits@1: 37.2
        # ent_feature = torch.sparse.mm(adj, ent_feature)

        # [epoch 40/300]  epoch loss: 396.73954, time cost: 0.452s
        # Hits@1:  0.4401904761904762   Hits@5:  0.7095238095238096   Hits@10:  0.7978095238095239   MRR:  0.5597874619811612

        # 连通图换成原始图，关系值为1
        # [epoch 173/300]  epoch loss: 452.13971, time cost: 0.311s
        # Hits@1:  0.4119047619047619   Hits@5:  0.6411428571428571   Hits@10:  0.7247619047619047   MRR:  0.5167577509046468


        # ******************* GAT *********************
        # adj = self.adj_list.tolist()
        # g = dgl.graph((adj[0], adj[1])).to(self.device)
        # g = dgl.add_self_loop(g)
        # gatconv = GATConv(self.node_hidden, self.node_hidden, num_heads=5).to(self.device)
        # res = gatconv(g, ent_feature)
        # ent_feature = torch.sum(res, dim=1)

        # ****************** R-GCN ********************
        # adj = self.adj_list.tolist()
        # g = dgl.graph((self.adj_list[0], self.adj_list[1]))
        # values = self.values.to(self.device)
        # conv = RelGraphConv(self.node_hidden, self.node_hidden, self.rel_size, regularizer='basis', num_bases=2).to(self.device)
        # ent_feature = conv(g, ent_feature, values)

        # ******************** LSMR *******************
        # ent_feature = self.encoder(adj, ent_feature)
        opt = [self.rel_embedding.weight, self.adj_list, self.r_index, self.r_val]
        ent_feature = self.e_encoder([ent_feature] + opt)

        out_feature = self.dropout(ent_feature)
        return out_feature

    def forward(self, train_paris:torch.Tensor):
        out_feature = self.gcn_forward()
        loss1 = self.align_loss(train_paris, out_feature)
        return loss1

    def align_loss(self, pairs, emb):
        def squared_dist(A, B):
            row_norms_A = torch.sum(torch.square(A), dim=1)
            row_norms_A = torch.reshape(row_norms_A, [-1, 1])
            row_norms_B = torch.sum(torch.square(B), dim=1)
            row_norms_B = torch.reshape(row_norms_B, [1, -1])
            return row_norms_A + row_norms_B - 2 * torch.matmul(A, B.t())

        l, r = pairs[:, 0].long(), pairs[:, 1].long()
        l_emb, r_emb = emb[l], emb[r]

        pos_dis = torch.sum(torch.square(l_emb - r_emb), dim=-1, keepdim=True)
        l_neg_dis = squared_dist(l_emb, emb)
        r_neg_dis = squared_dist(r_emb, emb)

        del l_emb, r_emb

        l_loss = pos_dis - l_neg_dis + self.gamma
        l_loss = l_loss * (1 - F.one_hot(l, num_classes=self.node_size) - F.one_hot(r, num_classes=self.node_size))
        r_loss = pos_dis - r_neg_dis + self.gamma
        r_loss = r_loss * (1 - F.one_hot(l, num_classes=self.node_size) - F.one_hot(r, num_classes=self.node_size))

        del r_neg_dis, l_neg_dis

        r_loss = (r_loss - torch.mean(r_loss, dim=-1, keepdim=True).detach()) / torch.std(r_loss, dim=-1, unbiased=False, keepdim=True).detach()
        l_loss = (l_loss - torch.mean(l_loss, dim=-1, keepdim=True).detach()) / torch.std(l_loss, dim=-1, unbiased=False, keepdim=True).detach()

        lamb, tau = 30, 10
        l_loss = torch.logsumexp(lamb * l_loss + tau, dim=-1)
        r_loss = torch.logsumexp(lamb * r_loss + tau, dim=-1)
        return torch.mean(l_loss + r_loss)

    def get_embeddings(self, index_a, index_b):
        # forward
        out_feature = self.gcn_forward()
        out_feature = out_feature.cpu()

        # get embeddings
        index_a = torch.Tensor(index_a).long()
        index_b = torch.Tensor(index_b).long()
        Lvec = out_feature[index_a]
        Rvec = out_feature[index_b]
        Lvec = Lvec / (torch.linalg.norm(Lvec, dim=-1, keepdim=True) + 1e-5)
        Rvec = Rvec / (torch.linalg.norm(Rvec, dim=-1, keepdim=True) + 1e-5)
        out_feature = out_feature / (torch.linalg.norm(out_feature, dim=-1, keepdim=True) + 1e-5)
        return Lvec, Rvec, out_feature

    def get_emb(self):
        # forward
        out_feature = self.gcn_forward()
        out_feature = out_feature.cpu()

        # get embeddings
        ill_ent = torch.Tensor(self.ill_ent).long()
        emb = out_feature[ill_ent]

        emb = emb / (torch.linalg.norm(emb, dim=-1, keepdim=True) + 1e-5)

        return emb

    def print_all_model_parameters(self):
        logging.info('\n------------Model Parameters--------------')
        info = []
        head = ["Name", "Element Nums", "Element Bytes", "Total Size (MiB)", "requires_grad"]
        total_size = 0
        total_element_nums = 0
        for name, param in self.named_parameters():
            info.append((name,
                         param.nelement(),
                         param.element_size(),
                         round((param.element_size()*param.nelement())/2**20, 3),
                         param.requires_grad)
                        )
            total_size += (param.element_size()*param.nelement())/2**20
            total_element_nums += param.nelement()
        logging.info(tabulate(info, headers=head, tablefmt="grid"))
        logging.info(f'Total # parameters = {total_element_nums}')
        logging.info(f'Total # size = {round(total_size, 3)} (MiB)')
        logging.info('--------------------------------------------')
        logging.info('')


