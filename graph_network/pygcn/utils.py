import os
import pandas as pd
import numpy as np
import random
from collections import OrderedDict
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

import torch



def load_data(path, dataset, sub_feat=None):
    """Load multimorbidity network and disease features"""
    disease_order = []
    with open(path + 'data_split4graphNet/' + dataset + '/disease_index.txt', 'r') as f:
        for line in f:
            disease_order.append(line.strip().split()[0])
        f.close()

    # phenotype data
    df = pd.read_csv(path + 'data/disease_phenotype_score_data_processed.csv', index_col=0)
    df.index = [index[:3] for index in df.index]
    df = df.reindex(disease_order)

    features = np.concatenate([df[df > 0].fillna(0).values, -df[df < 0].fillna(0).values], axis=-1)
    features = pd.DataFrame(features)
    new_columns = [col + '*pos' for col in df.columns] + [col + '*neg' for col in df.columns]
    features.columns = new_columns

    if sub_feat is not None:
        features = features.loc[:, features.columns.isin(sub_feat)]

    features = features.values
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    features = torch.FloatTensor(features)

    data_dict = OrderedDict()
    for i in range(0, 100):
        with open(path + 'data_split4graphNet/' + dataset + '/data' + str(i) + '.npy', 'rb') as f:
            train_edge = np.load(f)
            val_edge = np.load(f)
            val_edge_false = np.load(f)
            test_edge = np.load(f)
            test_edge_false = np.load(f)
        f.close()

        adj = np.zeros((len(disease_order), len(disease_order)))
        adj[train_edge[:, 0], train_edge[:, 1]] = 1
        adj = adj + adj.T + np.eye(adj.shape[0])
        adj_norm = normalize(adj)

        data_dict[i] = {'adj': torch.FloatTensor(adj),
                        'adj_norm': torch.FloatTensor(adj_norm),
                        'val_edge': torch.LongTensor(val_edge),
                        'val_edge_false': torch.LongTensor(val_edge_false),
                        'test_edge': torch.LongTensor(test_edge),
                        'test_edge_false': torch.LongTensor(test_edge_false)}

    return features, data_dict




def normalize(adj):
    degrees = adj.sum(1)
    degrees_matrix_inv_sqrt = np.diag(np.power(degrees, -0.5))
    return np.dot(np.dot(degrees_matrix_inv_sqrt, adj), degrees_matrix_inv_sqrt)



def evaluate_embeddings(embeddings, pos_edges, neg_edges, loss_func):
    innerProd = []
    for edge in pos_edges:
        innerProd.append(embeddings[edge[0]].dot(embeddings[edge[1]]).detach().cpu().item())
    for edge in neg_edges:
        innerProd.append(embeddings[edge[0]].dot(embeddings[edge[1]]).detach().cpu().item())
    labels = [1] * pos_edges.shape[0] + [0] * neg_edges.shape[0]
    preds = torch.sigmoid(torch.FloatTensor(innerProd))

    auroc = roc_auc_score(labels, preds)
    precision, recall, thresholds = precision_recall_curve(labels, preds)
    auprc = auc(recall, precision)

    loss = loss_func(torch.FloatTensor(innerProd), torch.FloatTensor(labels))

    return auroc, auprc, loss





def get_selected_phenotypes(path, lmd=0.0001, z_threshold=0.001):
    file_list = os.listdir(path)
    file_list = [f for f in file_list if f.startswith('lmd' + str(lmd))]
    df = pd.DataFrame()
    for f in file_list:
        df1 = pd.read_csv(path + f, index_col=0)
        df = pd.concat([df, df1], axis=1)
    s = df.max(axis=1)
    all_phenotypes = list(s.index)
    selected_phenotypes = list(s[s > z_threshold].index)

    return all_phenotypes, selected_phenotypes





def pred_load_data(path, dataset, sub_feat=None):
    # phenotype data
    phenotype_df = pd.read_csv(path + 'data/disease_phenotype_score_data_processed.csv', index_col=0)
    phenotype_df.index = [index[:3] for index in phenotype_df.index]
    features = np.concatenate([phenotype_df[phenotype_df > 0].fillna(0).values, -phenotype_df[phenotype_df < 0].fillna(0).values], axis=-1)
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    phenotype_df = pd.DataFrame(features, index=phenotype_df.index,
                columns=[col + '*pos' for col in phenotype_df.columns] + [col + '*neg' for col in phenotype_df.columns])

    if sub_feat is not None:
        phenotype_df = phenotype_df.loc[:, phenotype_df.columns.isin(sub_feat)]

    # multimorbidity dataset
    if dataset == 'UKB':
        multimorbidity_df = pd.read_csv(path + 'data/ukb_multimorbidity.csv')
    elif dataset == 'HuDiNe':
        multimorbidity_df = pd.read_csv(path + 'data/hudine_multimorbidity.csv')

    disease_intersection = set(phenotype_df.index) & (set(multimorbidity_df['code1']) | set(multimorbidity_df['code2']))

    multimorbidity_df = multimorbidity_df[multimorbidity_df['code1'].isin(disease_intersection)
                                    & multimorbidity_df['code2'].isin(disease_intersection)]

    disease_order = list(set(multimorbidity_df['code1']) | set(multimorbidity_df['code2']))
    random.shuffle(disease_order)
    phenotype_df = phenotype_df.reindex(disease_order)

    return phenotype_df, multimorbidity_df