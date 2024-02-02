# from gensim.models import KeyedVectors
# import pickle
#
#
# def cixiangliang(kg_dict, pickle_name):
#     temp_kg_dict = {}
#
#     for item in kg_dict:
#         word = kg_dict[item]
#         if word in word_vectors:
#             vector = word_vectors[word]
#             temp_kg_dict[item] = vector
#             print(f"单词'{word}'的词向量维度:", len(vector), )
#             # print("词向量:", vector)
#         else:
#             print(f"单词 '{word}' 不在词向量模型中。")
#             temp_kg_dict[item] = word
#
#     with open(pickle_name, 'wb') as f:
#         pickle.dump(temp_kg_dict, f)
#
# # 加载预训练的Word2Vec词向量模型
#
# bath_path = '/home/shaohongen/ContEA-main-LSM/datasets/DBP15K/base/'
#
# kg_1_dict = {}
# with open(bath_path+'ent_ids_1', "r", encoding="utf-8") as f:
#     for line in f.readlines():
#         id = int(line.split('\t')[0])
#         name = line.split('\t')[1].replace('http://zh.dbpedia.org/resource/', '').replace('\n', '')
#         kg_1_dict[id] = name
#
# # print(kg_1_dict)
# kg_2_dict = {}
# with open(bath_path+'ent_ids_2', "r", encoding="utf-8") as f:
#     for line in f.readlines():
#         id = int(line.split('\t')[0])
#         name = line.split('\t')[1].replace('http://dbpedia.org/resource/', '').replace('\n', '')
#         kg_2_dict[id] = name
#
#
# kgr_1_dict = {}
# with open(bath_path+'rel_ids_1', "r", encoding="utf-8") as f:
#     for line in f.readlines():
#         id = int(line.split('\t')[0])
#         name = line.split('\t')[1].replace('http://zh.dbpedia.org/property/', '').replace('\n', '')
#         kgr_1_dict[id] = name
#
# # print(kg_1_dict)
# kgr_2_dict = {}
# with open(bath_path+'rel_ids_2', "r", encoding="utf-8") as f:
#     for line in f.readlines():
#         id = int(line.split('\t')[0])
#         name = line.split('\t')[1].replace('http://dbpedia.org/property/', '').replace('\n', '')
#         kgr_2_dict[id] = name
#
# word_vectors = KeyedVectors.load_word2vec_format('/home/shaohongen/ContEA-main-LSM/model/GoogleNews-vectors-negative300.bin.gz', binary=True)
#
# cixiangliang(kg_1_dict, 'kg_1_dict.pickle')
# cixiangliang(kg_2_dict, 'kg_2_dict.pickle')
# cixiangliang(kgr_1_dict, 'kgr_1_dict.pickle')
# cixiangliang(kgr_2_dict, 'kgr_2_dict.pickle')


bath_path = '/home/shaohongen/ContEA-main-LSM/datasets/zh_en/'
# ss = ''
# with open(bath_path + 'triples_2', "r", encoding="utf-8") as f:
#     for line in f.readlines():
#         ll = line.split('\t')
#         ss += ll[0] + '\t' + ll[2]
#
# with open(bath_path + 'sample2.tsv', 'w', encoding='utf-8') as f:
#     f.write(ss)



# ********************************************
# id_neib = {}
# with open(bath_path + 'triples_2', "r", encoding="utf-8") as f:
#     for line in f.readlines():
#         ll = line.replace('\n', '').split('\t')
#         if int(ll[0]) == 13618 or int(ll[2]) == 13618:
#             print(ll)
#         if int(ll[0]) not in id_neib:
#             id_neib[int(ll[0])] = [int(ll[2])]
#         else:
#             if int(ll[2]) not in id_neib[int(ll[0])]:
#                 id_neib[int(ll[0])].append(int(ll[2]))
#
#         if int(ll[2]) not in id_neib:
#             id_neib[int(ll[2])] = [int(ll[0])]
#         else:
#             if int(ll[0]) not in id_neib[int(ll[2])]:
#                 id_neib[int(ll[2])].append(int(ll[0]))

# print(id_neib[3118])

# len_dict = {}
# for item in id_neib:
#     length = len(id_neib[item])
#     if length in len_dict:
#         len_dict[length] += 1
#     else:
#         len_dict[length] = 1
#
# import pickle
# with open('id_neib_en1.pickle', 'wb') as f:
#     pickle.dump(id_neib, f)
# print(len_dict)

# ************************************************


import pickle
import numpy as np

# with open('/home/shaohongen/ContEA-main-LSM/src/r_index.pickle', 'rb') as f:
#     r_index = pickle.load(f).cpu().numpy().tolist()
#
# with open('/home/shaohongen/ContEA-main-LSM/src/r_val.pickle', 'rb') as f:
#     r_val = pickle.load(f).cpu().numpy().tolist()
#
# with open('/home/shaohongen/ContEA-main-LSM/src/features.pickle', 'rb') as f:
#     features = pickle.load(f).cpu().detach().numpy()
#
# with open('/home/shaohongen/ContEA-main-LSM/pyrwr/new_tri.pickle', 'rb') as f:
#     new_tri = pickle.load(f)
#
# # print(features)
# large_array = np.full((38960, 38960), False, dtype=bool)
#
#
# for item in new_tri:
#     i, j = item
#     large_array[i, j] = True
#
# with open('large_array.pickle', 'wb') as f:
#     pickle.dump(large_array, f)

# ************************************

# for item in new_tri:
#     if item in r_index:
#         continue
#     r_index[0].append(item[0])
#     r_index[1].append(item[1])
#     val = np.dot(features[item[0]], features[item[1]])
#     r_val.append(val)
#     print(item, val)
# print(r_index, r_val)

# with open('new_r_index.pickle', 'wb') as f:
#     pickle.dump(r_index, f)
#
# with open('new_r_val.pickle', 'wb') as f:
#     pickle.dump(r_val, f)

# import numpy as np
#
#
# def attention_score(query_vector, key_vector):
#     # 计算两个向量的点积
#     dot_product = np.dot(query_vector, key_vector)
#
#     # 返回注意力分数
#     return dot_product
#
#
# # 示例
# query_vector = np.array([0.1, 0.2, 0.3])
# key_vector = np.array([0.4, 0.5, 0.6])
#
# score = attention_score(query_vector, key_vector)
# print(f"注意力分数: {score}")



# 统计某个函数运行时在GPU显存占用情况的代码

# import GPUtil
# import time
#
# # 记录模块运行前的GPU显存占用
# gpus_before = GPUtil.getGPUs()
# mem_used_before = [gpu.memoryUsed for gpu in gpus_before]
# print("GPU Memory Usage Before:", mem_used_before)
#
# # 要运行的函数放在此处 #
#
# time.sleep(10)  # 等待一段时间确保GPU状态更新
#
# # 记录模块运行后的GPU显存占用
# gpus_after = GPUtil.getGPUs()
# mem_used_after = [gpu.memoryUsed for gpu in gpus_after]
# print("GPU Memory Usage After:", mem_used_after)
#
# # 计算显存占用的变化
# mem_used_change = [after - before for before, after in zip(mem_used_before, mem_used_after)]
# print("Change in GPU Memory Usage:", mem_used_change)




from dgl.nn.pytorch.conv import RelGraphConv
import dgl
import torch

# 定义示例参数
num_nodes = 10  # 节点数量
in_feat = 5     # 输入特征的维度
num_rels = 3    # 关系类型的数量
out_feat = 2

# 创建一个示例图
g = dgl.rand_graph(num_nodes, 20)  # 随机图
x = torch.rand(num_nodes, in_feat) # 节点特征
rel_type = torch.randint(0, num_rels, (g.number_of_edges(),)) # 边的关系类型


RGCN = RelGraphConv(in_feat, out_feat, num_rels, regularizer='basis', num_bases=4)

out = RGCN(g, x, rel_type)

print(out)

import dgl
import torch

# 假设 self.adj_list 是一个二维张量，形状为 (2, E)
# 例如：self.adj_list = torch.tensor([[0, 1, 2], [1, 2, 3]])
adj_list = torch.tensor([[0, 1, 2], [1, 2, 3]])

# 直接使用二维张量创建 DGL 图
g = dgl.graph((adj_list[0], adj_list[1]))

print(g)
