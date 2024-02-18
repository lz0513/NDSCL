from __future__ import division
from __future__ import print_function
import random
import time
import argparse
import numpy as np
from copy import deepcopy as dcp
import os.path as osp
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import accuracy, normalize, drop_feature, adj_nor, plot_tsne
from model import GCN
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Amazon
import torch_geometric.transforms as T

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--data_aug', type=int, default=1,
                    help='do data augmentation.')
parser.add_argument('--top_k', type=int, default=1,
                    help='y_pre select top_k')
parser.add_argument('--sample_size', type=float, default=0.6,
                    help='sample size')
parser.add_argument('--neg_type', type=float, default=0,
                    help='0,selection;1 not selection')
parser.add_argument('--encoder_type', type=int, default=3,
                    help='do data augmentation.')
parser.add_argument('--debias', type=float, default=0.098,
                    help='debias rate.')
parser.add_argument('--seed', type=int, default=2023, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--tau', type=float, default=0.4,
                    help='tau rate .')
parser.add_argument('--dataset', type=str, default='Cora',
                    help='Cora/CiteSeer/PubMed/')
parser.add_argument('--use_gdc', action='store_true', default=True,
                    help='Use GDC preprocessing.')

args = parser.parse_args()
times = 10


# load dataset
def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'Pubmed', 'DBLP', 'WikiCS', 'Amazon-Photo']
    name = 'dblp' if name == 'DBLP' else name
    print(name)
    if name == 'dblp':
        return CitationFull(path, name, transform=T.NormalizeFeatures())
    elif name == 'WikiCS':
        return WikiCS(path, transform=T.NormalizeFeatures(), is_undirected=True)
    elif name == 'Amazon-Photo':
        return Amazon(path, name='photo', transform=T.NormalizeFeatures())
    else:
        return (CitationFull if name == 'dblp' else Planetoid)(path, name, transform=T.NormalizeFeatures())


if args.dataset == 'Cora' or args.dataset == 'CiteSeer' or args.dataset == 'PubMed':
    path = osp.join(osp.expanduser('~'), 'datasets', args.dataset)
    print(path)
else:
    path = osp.expanduser('~/datasets')
    path = osp.join(path, args.dataset)

dataset = get_dataset(path, args.dataset)
data = dataset[0]
data1 = dataset[0]

if args.use_gdc:
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                # diffusion_kwargs=dict(method='heat', t=5),
                sparsification_kwargs=dict(method='topk', k=128, dim=0),
                exact=True)
    data_dif = gdc(data1)

#######数据处理
idx_train = data.train_mask
idx_val = data.val_mask

idx_test = data.test_mask
features = data.x
features = normalize(features)
features = torch.from_numpy(features)
labels = data.y
adj = torch.eye(data.x.shape[0])
for i in range(data.edge_index.shape[1]):
    adj[data.edge_index[0][i]][data.edge_index[1][i]] = 1
adj = adj.float()
adj = adj_nor(adj)

features_dif = data_dif.x
features_dif = normalize(features_dif)
features_dif = torch.from_numpy(features_dif)
adj_dif = torch.eye(data_dif.x.shape[0])
for i in range(data_dif.edge_index.shape[1]):
    adj_dif[data_dif.edge_index[0][i]][data_dif.edge_index[1][i]] = data_dif.edge_attr[i]
adj_dif = adj_dif.float()
adj_dif = adj_nor(adj_dif)

best_model = None
best_val_acc = 0.0

def train(model, optimizer, epoch, features, adj, idx_train, idx_val, labels, data_aug, encoder_type, debias, top_k,
          sample_size, neg_type, feature_dif, adj_dif, p_hat):
    global best_model
    global best_val_acc
    t = time.time()
    model.train()
    optimizer.zero_grad()

    # semi-supervised CE loss
    y_pre, output = model(features, adj, encoder_type)
    loss_train = F.cross_entropy(y_pre[idx_train], labels[idx_train])
    acc_train = accuracy(y_pre[idx_train], labels[idx_train])

    # sample nodes
    node_mask = torch.empty(features.shape[0], dtype=torch.float32).uniform_(0, 1).cuda()
    node_mask = node_mask < sample_size

    # negative selection, neg_mask
    if neg_type == 0:
        y_pre = y_pre.detach()
        y_pre = y_pre[node_mask]

        _, y_poslabel = torch.topk(y_pre, top_k)
        y_pl = torch.zeros(y_pre.shape).cuda().scatter_(1, y_poslabel, 1)
        neg_mask = torch.mm(y_pl, y_pl.T) <= 0
        neg_mask = neg_mask.cuda()

        del y_pl, y_poslabel
        torch.cuda.empty_cache()
    else:
        neg_mask = (1 - torch.eye(node_mask.sum())).cuda()

    threshold = 0.6
    m_tau = 1.2
    momentum = 0.999

    if data_aug == 1:
        y_pre_s, output_dif = model(feature_dif, adj_dif, encoder_type)

        features = drop_feature(features, 0.1)
        y_pre_week, out_week = model(features, adj, encoder_type)

        y_pre_week = F.softmax(y_pre_week - m_tau * torch.log(p_hat).cuda(), dim=1)
        max_probs, pseudo_label = torch.max(y_pre_week, dim=-1)
        mask = max_probs.ge(threshold).float()

        p_hat = momentum * p_hat.cuda() + (1 - momentum) * y_pre_week.detach().mean(dim=0)
        delta_logits = torch.log(p_hat)

        y_pre_s = y_pre_s + m_tau * delta_logits

        del feature_dif
        torch.cuda.empty_cache()

        loss_u = (F.cross_entropy(y_pre_s[node_mask], pseudo_label[node_mask], reduction='none') * mask[
            node_mask]).mean()
        loss_cl = model.cl_lossaug(output, output_dif, node_mask, neg_mask, debias)
    else:
        pass

    if neg_type == 0:
        if epoch <= 50:
            loss = loss_train + 0.0001 * loss_cl + 0.1 * loss_u
        else:
            loss = loss_train + 0.8 * loss_cl + 1.2 * loss_u
    else:
        loss = loss_train + args.weight * loss_cl
    loss.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        y_pre, _ = model(features, adj, encoder_type)

    loss_val = F.cross_entropy(y_pre[idx_val], labels[idx_val])
    acc_val = accuracy(y_pre[idx_val], labels[idx_val])
    if acc_val > best_val_acc:
        best_val_acc = acc_val
        best_model = dcp(model)

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'loss_cl: {:.4f}'.format(loss_cl.item()),
          'loss_u: {:.4f}'.format(loss_u.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
def test(model, features, adj, labels, idx_test, encoder_type):
    model.eval()
    y_pre, out = model(features, adj, encoder_type)
    y_pre = F.log_softmax(y_pre, dim=1)
    # plot_tsne(y_pre.detach().cpu().numpy(), labels.cpu(). numpy())
    loss_test = F.nll_loss(y_pre[idx_test], labels[idx_test])
    acc_test = accuracy(y_pre[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test


# main
features = features.cuda()
features_dif = features_dif.cuda()
adj = adj.cuda()
adj_dif = adj_dif.cuda()
labels = labels.cuda()
idx_train = idx_train.cuda()
idx_val = idx_val.cuda()
idx_test = idx_test.cuda()
data.edge_index = data.edge_index.cuda()

test_acc = torch.zeros(times)
test_acc = test_acc.cuda()

# seed = args.seed
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)

for i in range(times):
    best_model = None
    best_val_acc = 0.0
    # Model and optimizer
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout,
                tau=args.tau).cuda()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    # Train model
    t_total = time.time()
    nclass = labels.max().item() + 1
    p_hat = torch.ones([1, nclass]) / nclass
    for epoch in range(args.epochs):
        train(model, optimizer, epoch, features, adj, idx_train, idx_val, labels, args.data_aug, args.encoder_type,
              args.debias, args.top_k, args.sample_size, args.neg_type, features_dif, adj_dif, p_hat)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    # Testing
    test_acc[i] = test(best_model, features, adj, labels, idx_test, args.encoder_type)

print("=== FINISH ===")
print(torch.max(test_acc))
print(torch.min(test_acc))
print("10次平均", torch.mean(test_acc))
print("10次标准差", test_acc.std())
print(test_acc)