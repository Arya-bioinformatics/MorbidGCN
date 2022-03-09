from __future__ import division
from __future__ import print_function

import sys
sys.path.append('graph_network/')
import argparse
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from pygcn.utils import load_data, evaluate_embeddings, get_selected_phenotypes
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
parser.add_argument('--dataset', type=str, default='UKB', help='Multimorbidity dataset')
parser.add_argument('--lmd', type=float, default=0.0001, help='lambda of L1 regularization')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--outfile', type=str, default='gn_result/ukb_selectedFeature.csv')

args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)


# data
all_phenotypes, selected_phenotypes = get_selected_phenotypes(path='feature_selection_result/' + args.dataset, lmd=args.lmd, z_threshold=0.001)
# selected_phenotypes = all_phenotypes

features, data_dict = load_data(path="./", dataset=args.dataset, sub_feat=selected_phenotypes)
# features = torch.eye(features.shape[0])


result = []
for i in data_dict:
    data = data_dict[i]
    adj = data['adj']
    adj_norm = data['adj_norm']
    val_edge = data['val_edge']
    val_edge_false = data['val_edge_false']
    test_edge = data['test_edge']
    test_edge_false = data['test_edge_false']

    train_mask_edge = torch.cat([val_edge, val_edge_false, test_edge, test_edge_false], dim=0)
    train_mask = torch.sparse_coo_tensor(train_mask_edge.T, torch.ones(train_mask_edge.shape[0]), adj.shape).to_dense()
    train_mask = torch.maximum(train_mask, train_mask.T) + torch.eye(adj.shape[0])
    train_mask_tensor = torch.where(train_mask == 0, train_mask + 1, train_mask - 1)
    train_mask_numpy = train_mask_tensor.numpy().flatten()


    # Model and optimizer
    model = GCN(nfeat=features.shape[1], nhid=args.hidden, n_emb=args.embedding, dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean', weight=train_mask_tensor.to(device).view(-1))
    bce_loss_eval = torch.nn.BCEWithLogitsLoss(reduction='mean')

    model.to(device)
    features = features.to(device)
    adj = adj.to(device)
    adj_norm = adj_norm.to(device)
    val_edge = val_edge.to(device)
    val_edge_false = val_edge_false.to(device)
    test_edge = test_edge.to(device)
    test_edge_false = test_edge_false.to(device)


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
                                    adj_pred.detach().cpu().numpy().flatten() * train_mask_numpy)
        precision, recall, thresholds = precision_recall_curve(adj.cpu().numpy().flatten() * train_mask_numpy,
                                                               adj_pred.detach().cpu().numpy().flatten() * train_mask_numpy)
        auprc_train = auc(recall, precision)

        auroc_val, auprc_val, loss_val = evaluate_embeddings(embedding, val_edge, val_edge_false, bce_loss_eval)
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'auroc_train: {:.4f}'.format(auroc_train.item()),
              'auprc_train: {:.4f}'.format(auprc_train.item()),
              'auroc_val: {:.4f}'.format(auroc_val.item()),
              'auprc_val: {:.4f}'.format(auprc_val.item()),
              'loss_val: {:.4f}'.format(loss_val.item()))
        return embedding, auroc_val, auprc_val


    def test(embedding):
        auroc_test, auprc_test, loss_test = evaluate_embeddings(embedding, test_edge, test_edge_false, bce_loss_eval)
        print('Test set results:',
              'auroc= {:.4f}'.format(auroc_test.item()),
              'auprc= {:.4f}'.format(auprc_test.item()),
              'loss= {:.4f}'.format(loss_test.item()))
        return auroc_test, auprc_test


    # Train model
    best_embedding = None
    best_auroc_val = 0
    best_auprc_val = 0
    for epoch in range(args.epochs):
        embedding, auroc_val, auprc_val = train(epoch)

        if auroc_val > best_auroc_val:
            best_embedding = embedding
            best_auroc_val = auroc_val
            best_auprc_val = auprc_val
            patience = 0
        else:
            patience += 1
            if patience == 10:
                print('early stopping ...')
                break

    # Testing
    auroc_test, auprc_test = test(best_embedding)
    result.append([i, best_auroc_val, best_auprc_val, auroc_test, auprc_test])

df = pd.DataFrame(result, columns=['data', 'auroc_val', 'auprc_val', 'auroc_test', 'auprc_test'])
df.to_csv(args.outfile, index=False)

print('ok')