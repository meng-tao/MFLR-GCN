#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import iterator
from pyrwr.pyrwr_1 import PyRWR
import pandas as pd
import torch

class RWR(PyRWR):
    def __init__(self):
        super().__init__()

    def compute(self,
                seed,
                c=0.15,
                epsilon=1e-6,
                max_iters=100,
                handles_deadend=True,
                device='cpu'):
        '''
        Compute the RWR score vector w.r.t. the seed node

        inputs
            seed : int
                seed (query) node id
            c : float
                restart probability
            epsilon : float
                error tolerance for power iteration
            max_iters : int
                maximum number of iterations for power iteration
            handles_deadend : bool
                if true, it will handle the deadend issue in power iteration
                otherwise, it won't, i.e., no guarantee for sum of RWR scores
                to be 1 in directed graphs
        outputs
            r : ndarray
                RWR score vector
        '''

        self.normalize()

        # adjust range of seed node id
        seed = seed - self.base

        #  q = np.zeros((self.n, 1))
        q = np.zeros(self.n)
        if seed < 0 or seed >= self.n:
            raise ValueError('Out of range of seed node id')

        q[seed] = 1.0

        r, residuals = iterator.iterate(self.nAT, q, c, epsilon,
                                        max_iters, handles_deadend,
                                        norm_type=1,
                                        device=device)

        return r


def print_result(node_ids, r, dd):
    df = pd.DataFrame()
    df['Node'] = node_ids
    df['Score'] = r
    df = df.sort_values(by=['Score'], ascending=False)
    df = df.reset_index(drop=True)
    df.index = df.index + 1
    return list(df[0:dd + 10]["Node"]), list(df[0:dd + 10]["Score"])


def get_seeds(seeds):
    if type(seeds) is str:
        _seeds = []
        with open(seeds, 'r') as f:
            _seeds = [int(seed) for seed in f]
    elif type(seeds) is list:
        _seeds = [int(seed) for seed in seeds]
    else:
        raise TypeError('Seeds for PPR should be list or file path')

    return _seeds


def write_vector(output_path, node_ids, r):
    data = np.vstack((node_ids, r)).transpose()
    np.savetxt(output_path, data, fmt='%d %e')



if __name__ == '__main__':

    # bath_path = '/home/shaohongen/ContEA-main-LSM/datasets/zh_en/'
    # entity_list = []
    # with open(bath_path + 'ent_ids_1', "r", encoding="utf-8") as f:
    #     for line in f.readlines():
    #         ll = line.split('\t')
    #         entity_list.append(ll[0])
    import pickle

    with open('/home/shaohongen/ContEA-main-LSM/id_neib_en1.pickle', 'rb') as f:
        id_neib = pickle.load(f)

    device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
    rwr = RWR()
    rwr.read_graph('/home/shaohongen/ContEA-main-LSM/datasets/zh_en/sample2.tsv', 'undirected')
    new_node = []
    for item in id_neib:
        # print(item, id_neib[item])
        if int(item) != 13618:
            continue
        r = rwr.compute(int(item), c=0.15, epsilon=1e-9, max_iters=100, handles_deadend=True, device='gpu')
        node_ids = rwr.node_ids

        # write_vector('output/scores.tsv', node_ids, r)
        ne_new, score = print_result(node_ids, r, len(id_neib[item]))
        print(item, ne_new)
        print(score)
        ne_new.extend(id_neib[item])
        print(id_neib[item])
        for ii in ne_new:
            new_node.append([item, ii])
        # print(new_node)
        break
    # with open('new_tri.pickle', 'wb') as f:
    #     pickle.dump(new_node, f)
    # [2569, 23163, 1386, 10013, 991, 23637]