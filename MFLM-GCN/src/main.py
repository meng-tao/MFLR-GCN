import torch
from model import Encoder_Model
import warnings
from eval import Evaluate
import argparse
import time
from utils import *
from evals import evaluate

warnings.filterwarnings('ignore')

from data_loader import KGs
import numpy as np
import logging
torch.cuda.device_count()

def dual_evaluate():
    #from evals import evaluate

    Lvec, Rvec,out_feature = model.get_embeddings(test_pair[:, 0], test_pair[:, 1])
    evals = evaluate(dev_pair=test_pair)
    results = evals.CSLS_cal(Lvec, Rvec)

    def cal(results):
        hits1, hits5, hits10, mrr = 0, 0, 0, 0
        for x in results[:, 1]:
            if x < 1:
                hits1 += 1
            if x < 5:
                hits5 += 1
            if x < 10:
                hits10 += 1
            mrr += 1 / (x + 1)
        return hits1, hits5, hits10, mrr

    hits1, hits5, hits10, mrr = cal(results)
    print("Hits@1: ", hits1 / len(Lvec), " ", "Hits@5: ", hits5 / len(Lvec), " ", "Hits@10: ", hits10 / len(Lvec)," ", "MRR: ", mrr / len(Lvec))
    return out_feature


def train_base(args, train_pair, model:Encoder_Model):
    total_train_time = 0.0
    for epoch in range(args.epoch):
        time1 = time.time()
        total_loss = 0
        batch_num = len(train_pair) // args.batch_size + 1
        model.train()
        for b in range(batch_num):
            pairs = train_pair[b * args.batch_size:(b + 1) * args.batch_size]
            if len(pairs) == 0:
                continue
            pairs = torch.from_numpy(pairs).to(device)
            optimizer.zero_grad()
            loss = model(pairs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        time2 = time.time()
        total_train_time += time2 - time1

        print(f'[epoch {epoch + 1}/{args.epoch}]  epoch loss: {(total_loss):.5f}, time cost: {(time2 - time1):.3f}s')

        if True:
            logging.info("---------Validation---------")
            model.eval()
            with torch.no_grad():
                dual_evaluate()


    print(f"Total training time: {total_train_time:.3f}s")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='alignment model')
    parser.add_argument('--log_path', default='../logs', type=str)
    parser.add_argument('--dataset', default='zh_en', type=str)
    parser.add_argument('--batch', default='base', type=str)
    parser.add_argument('--gpu', default=0, type=int)

    # training and finetuning hyper-parameters
    parser.add_argument('--ent_hidden', default=300, type=int)
    parser.add_argument('--rel_hidden', default=300, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--beta', default=0.1, type=float)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--ind_dropout_rate', default=0.3, type=float)

    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--depth', default=3, type=int)
    parser.add_argument('--gamma', default=2.0, type=float)

    # hyper-parameters for test
    parser.add_argument('--eval_batch_size', default=512, type=int)
    parser.add_argument('--dev_interval', default=2, type=int)
    parser.add_argument('--stop_step', default=3, type=int)
    parser.add_argument('--sim_threshold', default=0.0, type=float)
    parser.add_argument('--topk', default=1, type=int)
    parser.add_argument('--M', default=500, type=int)

    # LSM
    parser.add_argument('--hidden_dim', type=int, default=300, help='RNN hidden state size.')
    parser.add_argument('--num_layers', type=int, default=4, help='Num of AGGCN blocks.')
    parser.add_argument('--heads', type=int, default=4, help='Num of heads in multi-head attention.')

    args = parser.parse_args()

    # set gpu
    device = set_device(args)

    # load data
    kgs = KGs()
    train_pair, valid_pair, test_pair, ent_adj, r_index, r_val, \
        ent_adj_with_loop, ent_rel_adj, entity1, entity2, ill_ent, values = kgs.load_data(args)

    ent_adj = torch.from_numpy(np.transpose(ent_adj))
    ent_rel_adj = torch.from_numpy(np.transpose(ent_rel_adj))
    ent_adj_with_loop = torch.from_numpy(np.transpose(ent_adj_with_loop))
    r_index = torch.from_numpy(np.transpose(r_index))
    r_val = torch.from_numpy(r_val)

    # define model
    model = Encoder_Model(node_hidden=args.ent_hidden,
                          rel_hidden=args.rel_hidden,
                          node_size=kgs.old_ent_num,
                          rel_size=kgs.total_rel_num,
                          triple_size=kgs.triple_num,
                          device=device,
                          adj_matrix=ent_adj,
                          r_index=r_index,
                          r_val=r_val,
                          rel_matrix=ent_rel_adj,
                          ent_matrix=ent_adj_with_loop,
                          dropout_rate=args.dropout_rate,
                          ind_dropout_rate=args.ind_dropout_rate,
                          gamma=args.gamma,
                          lr=args.lr,
                          depth=args.depth,
                          values=torch.tensor(values)
                          ).to(device)

    evaluator = evaluate(dev_pair=test_pair)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)

    # print model parameters
    model.print_all_model_parameters()

    if 'base' in args.batch:
        train_base(args, train_pair, model)

