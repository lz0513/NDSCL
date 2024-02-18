import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import copy
import random
import pdb

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

# adj normalization
def adj_nor(edge):
    degree = torch.sum(edge, dim=1)
    degree = 1 / torch.sqrt(degree)
    degree = torch.diag(degree)
    adj = torch.mm(torch.mm(degree, edge), degree)
    return adj

# data augmentation
# node feature
def aug_random_mask(input_feature, drop_percent=0.2):
    node_num = input_feature.shape[0]
    mask_num = int(node_num * drop_percent)
    node_idx = [i for i in range(node_num)]
    mask_idx = random.sample(node_idx, mask_num)
    aug_feature = copy.deepcopy(input_feature)
    zeros = torch.zeros_like(aug_feature[0])
    for j in mask_idx:
        aug_feature[j] = zeros
    return aug_feature

# dimension feature
def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1),),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x

# adj
def aug_random_edge(input_adj, drop_percent=0.2):
    percent = drop_percent / 2
    row_idx, col_idx = input_adj.nonzero()

    index_list = []
    for i in range(len(row_idx)):
        index_list.append((row_idx[i], col_idx[i]))

    single_index_list = []
    for i in list(index_list):
        single_index_list.append(i)
        index_list.remove((i[1], i[0]))

    edge_num = int(len(row_idx) / 2)  # 9228 / 2
    add_drop_num = int(edge_num * percent / 2)
    aug_adj = copy.deepcopy(input_adj.todense().tolist())

    edge_idx = [i for i in range(edge_num)]
    drop_idx = random.sample(edge_idx, add_drop_num)

    for i in drop_idx:
        aug_adj[single_index_list[i][0]][single_index_list[i][1]] = 0
        aug_adj[single_index_list[i][1]][single_index_list[i][0]] = 0

    '''
    above finish drop edges
    '''
    node_num = input_adj.shape[0]
    l = [(i, j) for i in range(node_num) for j in range(i)]
    add_list = random.sample(l, add_drop_num)

    for i in add_list:
        aug_adj[i[0]][i[1]] = 1
        aug_adj[i[1]][i[0]] = 1

    aug_adj = np.matrix(aug_adj)
    aug_adj = sp.csr_matrix(aug_adj)
    return aug_adj

def label_propagation(adj, labels, idx, K, alpha):
    y0 = torch.zeros(size=(labels.shape[0], labels.max().item() + 1))
    for i in idx:
        y0[i][labels[i]] = 1.0

    y = y0
    for i in range(K):
        y = torch.matmul(adj, y)
        for i in idx:
            y[i] = F.one_hot(torch.tensor(labels[i].cpu().numpy().astype(np.int64)), labels.max().item() + 1)
        y = (1 - alpha) * y + alpha * y0
    return y

def plot_tsne(features, labels):
    import pandas as pd
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    import seaborn as sns
    class_num = len(np.unique(labels))  # 要分类的种类个数  eg:[0, 1, 2, 3]这个就是为4
    tsne_features = tsne.fit_transform(features)  # 将特征使用PCA降维至2维
    print('tsne_features的shape:', tsne_features.shape)
    plt.scatter(tsne_features[:, 0], tsne_features[:, 1])  # 将对降维的特征进行可视化
    df = pd.DataFrame()
    df["y"] = labels
    df["comp1"] = tsne_features[:, 0]
    df["comp2"] = tsne_features[:, 1]
    sns.scatterplot(x="comp1", y="comp2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", class_num),
                    data=df)
    plt.xlabel('t-SNE Dim-1', font={'family': 'Times New Roman', 'size': 15})
    plt.ylabel('t-SNE Dim-2', font={'family': 'Times New Roman', 'size': 15})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
