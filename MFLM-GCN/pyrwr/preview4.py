import copy
import csv
import multiprocess
import os
import random
from multiprocess import freeze_support

import numpy as np
from numpy import ndarray
import ExtensionWays.adjustGraph
import FindWays.findRealButterflyCommunity
import networkx as nx
from pyrwrmaster.pyrwr.rwr import RWR
import Model.RWR


# 读图
def readGraph(file_graph='dblp_graph_0602.tsv',
              file_left='dblp_left_0602.tsv',
              file_right='dblp_right_0602.tsv',
              file_butterfly='dblp_butterfly_0602.tsv',
              file_label='dblp_label.tsv',
              all_graph='com-dblp.ungraph.txt'):
    # 生产对应位置的文件绝对位置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_graph = os.path.join(current_dir, file_graph)
    file_left = os.path.join(current_dir, file_left)
    file_right = os.path.join(current_dir, file_right)
    file_butterfly = os.path.join(current_dir, file_butterfly)
    file_label = os.path.join(current_dir, file_label)
    all_graph = os.path.join(current_dir, all_graph)

    # 读取文件
    pool = multiprocessing.Pool(processes=5)
    # file_paths = [file_graph, file_left, file_right, file_butterfly, all_graph]
    file_paths = [file_left, file_right]
    result = pool.map(read_file, file_paths)
    pool.close()
    pool.join()

    # 建图
    # graph = result[0]
    # left = result[1]
    # right = result[2]
    # butterfly = result[3]
    # all_graph = result[4]
    left = result[0]
    right = result[1]

    # 读取全图和蝴蝶的标签
    return left, right


# 读取单个文件的函数
def read_file(file_path):
    graph = nx.Graph()
    with open(file_path, 'r', encoding='utf-8', newline='') as f:
        for line in f:
            if line.startswith('#'):
                continue
            line = line.strip('\n')
            line = line.strip().split('\t')
            graph.add_edge(
                int(line[0]), int(line[1])
            )
    return graph


# 循环计算RWR分数
# GPU加速
def rwrPreviewGpu(graph=nx.Graph(), alpha=0.5, eps=1e-8, max_iter=500, label='A', num_chunks=4):
    node_index = {n: i for i, n in enumerate(graph.nodes())}
    nodes = list(graph.nodes())
    current_dir = os.path.dirname(os.path.abspath(__file__))

    if label == 'A':
        rwrFile = 'rwr_A_max.tsv'
        rwrFile = os.path.join(current_dir, rwrFile)
        with open(rwrFile, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            for edge in graph.edges():
                writer.writerow(
                    [
                        node_index[edge[0]],
                        node_index[edge[1]]
                    ]
                )
    else:
        rwrFile = 'rwr_B_max.tsv'
        rwrFile = os.path.join(current_dir, rwrFile)
        with open(rwrFile, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            for edge in graph.edges():
                writer.writerow(
                    [
                        node_index[edge[0]],
                        node_index[edge[1]]
                    ]
                )

    # Split the nodes into multiple chunks
    # num_chunks = multiprocessing.cpu_count()
    # num_chunks = 6
    chunks = np.array_split(nodes, num_chunks)

    # Create a process pool
    pool = multiprocessing.Pool(num_chunks)

    # Calculate the RWR scores in parallel
    results = []
    for chunk in chunks:
        result = pool.apply_async(
            rwrGitHub, (rwrFile, len(graph.nodes()), node_index, chunk))
        results.append(result)

    # Collect the results
    all_scores = np.zeros(len(graph.nodes()))
    for result in results:
        all_scores += result.get()

    # Write the scores to a file
    all_scores_file = 'dblp_all_{}_scores_max.tsv'.format(label)
    all_scores_file = os.path.join(current_dir, all_scores_file)
    length = len(nodes)
    with open(all_scores_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        for v in nodes:
            writer.writerow([
                v, all_scores[node_index[v]] / len(nodes)
            ])


# RWR计算得分
def rwrGitHub(rwrFile='', nodes_length=None, id_map=None, v_list=None):
    # 引用RWR
    rwr = RWR()
    rwr.read_graph(input_path=rwrFile, graph_type='undirected')
    scores_array = np.zeros(nodes_length)

    for v in v_list:
        # 要计算单个节点对其他所有节点的RWR分数
        # 映射ID
        data = rwr.compute(seed=id_map[v], device='gpu', epsilon=1e-8, max_iters=200)
        scores_array = data + scores_array

    return scores_array


# RWR计算得分
def rwrGitHub_single(rwrFile='', alpha=0.15, relative_file='', nodes=[], node_index={}, label='left'):
    # 引用RWR
    rwr = RWR()
    rwr.read_graph(input_path=rwrFile, graph_type='undirected')

    # 要计算单个节点对其他所有节点的RWR分数
    for v in nodes:
        write_file = f'dblp_{label}_{v}.tsv'
        write_file = os.path.join(relative_file, write_file)
        data = rwr.compute(seed=node_index[v], device='gpu', epsilon=1e-6, max_iters=200, c=alpha)
        with open(write_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            for v in nodes:
                writer.writerow(
                    [v, data[node_index[v]]]
                )


def read0624():
    left = nx.Graph()
    right = nx.Graph()
    butterfly = nx.Graph()
    graph = nx.Graph()
    file_left = 'file_another_graph_500.tsv'
    file_right = 'file_graph_500.tsv'
    file_butterfly = 'dblp_connect_butterfly_0624.tsv'
    with open(file_left, 'r', encoding='utf-8', newline='') as f:
        for line in f:
            if line.startswith('#'):
                continue
            line = line.strip('\n')
            line = line.strip().split('\t')
            left.add_edge(
                int(line[0]), int(line[1])
            )
            graph.add_edge(
                int(line[0]), int(line[1])
            )
    with open(file_right, 'r', encoding='utf-8', newline='') as f:
        for line in f:
            if line.startswith('#'):
                continue
            line = line.strip('\n')
            line = line.strip().split('\t')
            right.add_edge(
                int(line[0]), int(line[1])
            )
            graph.add_edge(
                int(line[0]), int(line[1])
            )
    with open(file_butterfly, 'r', encoding='utf-8', newline='') as f:
        for line in f:
            if line.startswith('#'):
                continue
            line = line.strip('\n')
            line = line.strip().split('\t')
            butterfly.add_edge(
                int(line[0]), int(line[1])
            )
            graph.add_edge(
                int(line[0]), int(line[1])
            )

    return left, right, butterfly, graph


def read0625():
    left = nx.Graph()
    right = nx.Graph()
    butterfly = nx.Graph()
    graph = nx.Graph()
    file_left = 'file_another_graph_101.tsv'
    file_right = 'file_graph_101.tsv'
    file_butterfly = 'dblp_connect_butterfly_101.tsv'
    with open(file_left, 'r', encoding='utf-8', newline='') as f:
        for line in f:
            if line.startswith('#'):
                continue
            line = line.strip('\n')
            line = line.strip().split('\t')
            left.add_edge(
                int(line[0]), int(line[1])
            )
            graph.add_edge(
                int(line[0]), int(line[1])
            )
    with open(file_right, 'r', encoding='utf-8', newline='') as f:
        for line in f:
            if line.startswith('#'):
                continue
            line = line.strip('\n')
            line = line.strip().split('\t')
            right.add_edge(
                int(line[0]), int(line[1])
            )
            graph.add_edge(
                int(line[0]), int(line[1])
            )

    all_graph_file = 'com-dblp.ungraph.txt'
    all_graph = nx.Graph()
    with open(all_graph_file, 'r', encoding='utf-8', newline='') as f:
        for line in f:
            if line.startswith('#'):
                continue
            line = line.strip('\n')
            line = line.strip().split('\t')
            all_graph.add_edge(
                int(line[0]), int(line[1])
            )

            if left.has_node(line[0]) or right.has_node(line[1]):
                butterfly.add_edge(
                    int(line[0]), int(line[1])
                )

            if left.has_node(line[1]) or right.has_node(line[0]):
                butterfly.add_edge(
                    int(line[0]), int(line[1])
                )

    return left, right, graph, butterfly, all_graph


if __name__ == '__main__':
    freeze_support()
    multiprocessing.set_start_method('spawn')
    print('---Read Graph---')
    left, right, graph, butterfly, all_graph = read0625()
    # rwrPreviewGpu(graph=right, label='B', num_chunks=4)
    # rwrPreviewGpu(graph=left, label='A', num_chunks=4)

    # node_index = {n: i for i, n in enumerate(left.nodes())}
    # nodes = list(left.nodes())
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # rwrFile = 'dblp_rwr_A_0706.tsv'
    # rwrFile = os.path.join(current_dir, rwrFile)
    # with open(rwrFile, 'w', encoding='utf-8', newline='') as f:
    #     writer = csv.writer(f, delimiter='\t')
    #     for edge in left.edges():
    #         writer.writerow(
    #             [
    #                 node_index[edge[0]],
    #                 node_index[edge[1]]
    #             ]
    #         )
    # relative_file = 'dblp_single_rwr_scores'
    # relative_file = os.path.join(relative_file)
    # # rwrGitHub_single(rwrFile=rwrFile, node_index=node_index, nodes=nodes, label='left', relative_file=relative_file)
    #
    # node_index = {n: i for i, n in enumerate(right.nodes())}
    # nodes = list(right.nodes())
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # rwrFile = 'dblp_rwr_B_0706.tsv'
    # rwrFile = os.path.join(current_dir, rwrFile)
    # with open(rwrFile, 'w', encoding='utf-8', newline='') as f:
    #     writer = csv.writer(f, delimiter='\t')
    #     for edge in right.edges():
    #         writer.writerow(
    #             [
    #                 node_index[edge[0]],
    #                 node_index[edge[1]]
    #             ]
    #         )
    # rwrGitHub_single(rwrFile=rwrFile, node_index=node_index, nodes=nodes, label='right', relative_file=relative_file)
