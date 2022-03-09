from __future__ import division
from __future__ import print_function
import sys
sys.path.append('graph_network/')

import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
from itertools import combinations

import torch
import torch.optim as optim

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import StratifiedKFold

from pygcn.utils import load_data, evaluate_embeddings, get_selected_phenotypes, pred_load_data, normalize
from pygcn.models import GCN


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')




# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.')
parser.add_argument('--embedding', type=int, default=32, help='Dimensions of embeddings.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')

parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--dataset', type=str, default='UKB', help='multimorbidity dataset')

args = parser.parse_args()



# data
for run in range(1000):
    all_phenotypes, selected_phenotypes = get_selected_phenotypes(
        path='feature_selection_result/' + args.dataset + '/', lmd=0.0001, threshold=0.001)

    phenotype_df, multimorbidity_df = pred_load_data(path="./", dataset=args.dataset, sub_feat=selected_phenotypes)

    disease_order = list(phenotype_df.index)
    disease_index = {v: i for i, v in enumerate(disease_order)}
    index_disease = {i: v for i, v in enumerate(disease_order)}

    multimorbid_diseasePairs = set(zip(multimorbidity_df['code1'], multimorbidity_df['code2']))

    diseasePair_label = []
    for code1, code2 in combinations(disease_order, 2):
        if ((code1, code2) in multimorbid_diseasePairs) | ((code2, code1) in multimorbid_diseasePairs):
            diseasePair_label.append([disease_index[code1], disease_index[code2], 1])
        else:
            diseasePair_label.append([disease_index[code1], disease_index[code2], 0])

    diseasePair_label = np.array(diseasePair_label)
    for i in range(10):
        np.random.shuffle(diseasePair_label)

    diseasePairs = diseasePair_label[:, :2]
    labels = diseasePair_label[:, -1]

    diseasePair_ComorProbability = []

    skf = StratifiedKFold(n_splits=10)
    for train_idx, test_idx in skf.split(diseasePairs, labels):
        diseasePair_train, diseasePair_test = diseasePairs[train_idx], diseasePairs[test_idx]
        label_train, label_test = labels[train_idx], labels[test_idx]

        adj = sp.coo_matrix((label_train, (diseasePair_train[:, 0], diseasePair_train[:, 1])),
                            dtype=float, shape=(len(disease_order), len(disease_order))).toarray()
        adj = adj + adj.T + np.eye(adj.shape[0])
        adj_norm = normalize(adj)

        train_mask = sp.coo_matrix(
            (np.ones(diseasePair_test.shape[0]), (diseasePair_test[:, 0], diseasePair_test[:, 1])),
            dtype=float, shape=(len(disease_order), len(disease_order))).toarray()
        train_mask = train_mask + train_mask.T + np.eye(train_mask.shape[0])
        train_mask_numpy = np.where(train_mask == 0, train_mask + 1, train_mask - 1).flatten()
        train_mask_tensor = torch.from_numpy(train_mask).to(device)

        # Model and optimizer
        model = GCN(nfeat=phenotype_df.shape[1], nhid=args.hidden, n_emb=args.embedding, dropout=args.dropout)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

        model.to(device)
        features = torch.FloatTensor(phenotype_df.values).to(device)
        adj = torch.FloatTensor(adj).to(device)
        adj_norm = torch.FloatTensor(adj_norm).to(device)


        def train(epoch):
            model.train()
            optimizer.zero_grad()
            innerProd, _ = model(features, adj_norm)
            loss_train = bce_loss(innerProd.view(-1), adj.view(-1))
            loss_train.backward()
            optimizer.step()

            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            innerProd, embedding = model(features, adj_norm)
            adj_pred = torch.sigmoid(innerProd)

            auroc_train = roc_auc_score(adj.cpu().numpy().flatten() * train_mask_numpy,
                                        adj_pred.cpu().detach().numpy().flatten() * train_mask_numpy)
            precision, recall, thresholds = precision_recall_curve(adj.cpu().numpy().flatten() * train_mask_numpy,
                                                                   adj_pred.detach().cpu().numpy().flatten() * train_mask_numpy)
            auprc_train = auc(recall, precision)

            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'auroc_train: {:.4f}'.format(auroc_train.item()),
                  'auprc_train: {:.4f}'.format(auprc_train.item()))
            return embedding, auroc_train, auprc_train


        def test(embedding):
            pred_test = torch.sigmoid(torch.mm(embedding, embedding.T))[diseasePair_test[:, 0], diseasePair_test[:, 1]]

            auroc_test = roc_auc_score(label_test, pred_test.cpu().detach().numpy())
            precision, recall, thresholds = precision_recall_curve(label_test, pred_test.cpu().detach().numpy())
            auprc_test = auc(recall, precision)
            print('test', auroc_test, auprc_test)
            return pred_test


        # Train model
        best_embedding = None
        best_auroc_train = 0
        best_auprc_train = 0
        for epoch in range(args.epochs):
            embedding, auroc_train, auprc_train = train(epoch)

            if auroc_train > best_auroc_train:
                best_embedding = embedding
                best_auroc_train = auroc_train
                best_auprc_train = auprc_train
                patience = 0
            else:
                patience += 1
                if patience == 10:
                    print('early stopping ...')
                    break

        # Testing
        pred_test = test(best_embedding)
        for (idx1, idx2), label, prob in zip(diseasePair_test, label_test, pred_test):
            disease1 = index_disease[idx1]
            disease2 = index_disease[idx2]
            diseasePair_ComorProbability.append([disease1, disease2, str(label), str(prob.item())])

    df = pd.DataFrame(diseasePair_ComorProbability, columns=['disease1', 'disease2', 'label', 'probability'])
    df.to_csv('multimorbidity_prediction_result/' + args.dataset + '/pred' + str(run) + '.csv', index=False)