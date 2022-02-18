import pandas as pd
import numpy as np
from itertools import combinations
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
import torch.nn as nn
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')



class Weighted_InnerProd(torch.nn.Module):
    def __init__(self, in_features):
        super(Weighted_InnerProd, self).__init__()
        self.ww_ = nn.Parameter(torch.rand(in_features))

    def forward(self, x):
        inner_product = x @ torch.diag(self.ww_) @ x.T
        return inner_product, self.ww_


def read_data(dataset_path='data/ukb_multimorbidity.csv'):
    # phenotype data
    phenotype_df = pd.read_csv('data/disease_phenotype_score_data_processed.csv', index_col=0)
    phenotype_df.index = [index[:3] for index in phenotype_df.index]

    # UKB multimorbidity data
    multimorbidity_df = pd.read_csv(dataset_path)[['code1', 'code2']]
    disease_intersection = set(phenotype_df.index) & (set(multimorbidity_df['code1']) | set(multimorbidity_df['code2']))
    multimorbidity_df = multimorbidity_df[multimorbidity_df['code1'].isin(disease_intersection) & multimorbidity_df['code2'].isin(disease_intersection)]

    disease_order = list(set(multimorbidity_df['code1']) | set(multimorbidity_df['code2']))
    phenotype_df = phenotype_df.reindex(disease_order)

    return phenotype_df, multimorbidity_df


def train_test_split_(phenotype_df, multimorbidity_df, test_size=0.1):
    disease_order = list(phenotype_df.index)
    disease_index = {v: i for i, v in enumerate(disease_order)}

    multimorbid_diseasePair = set(zip(multimorbidity_df['code1'], multimorbidity_df['code2']))

    diseasePair_label = []
    for code1, code2 in combinations(disease_order, 2):
        if ((code1, code2) in multimorbid_diseasePair) | ((code2, code1) in multimorbid_diseasePair):
            diseasePair_label.append([disease_index[code1], disease_index[code2], 1])
        else:
            diseasePair_label.append([disease_index[code1], disease_index[code2], 0])

    diseasePair_label = np.array(diseasePair_label)

    while True:
        diseasePair_train, diseasePair_test, label_train, label_test = \
            train_test_split(diseasePair_label[:, :2], diseasePair_label[:, -1], test_size=test_size, stratify=diseasePair_label[:, -1])
        if len(set(diseasePair_train[:, 0]) | set(diseasePair_train[:, 1])) == len(disease_index):
            break

    return diseasePair_train, diseasePair_test, label_train, label_test


def get_pred_prob(inner_prod, edge_idx):
    sigmoid_inner_prod = torch.sigmoid(inner_prod)
    label_prob = sigmoid_inner_prod[edge_idx[:, 0], edge_idx[:, 1]]
    return label_prob


def train(model, optimizer, loss_func, epochs, lmd, features, diseasePair_train_cv, label_train_cv, diseasePair_validation, label_validation):
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        inner_prod, ww_ = model(features)
        label_train_pred = get_pred_prob(inner_prod, diseasePair_train_cv)

        loss = loss_func(label_train_pred, label_train_cv) + lmd * torch.norm(ww_, 1)
        loss.backward()
        optimizer.step()
        model.ww_.data[model.ww_.data < 0] = 0

        train_auroc = roc_auc_score(label_train_cv.cpu().numpy(), label_train_pred.detach().cpu().numpy())
        precision, recall, thresholds = precision_recall_curve(label_train_cv.cpu().numpy(), label_train_pred.detach().cpu().numpy())
        train_auprc = auc(recall, precision)

        label_validation_pred = get_pred_prob(inner_prod, diseasePair_validation)
        val_auroc = roc_auc_score(label_validation.cpu().numpy(), label_validation_pred.detach().cpu().numpy())
        precision, recall, thresholds = precision_recall_curve(label_validation.cpu().numpy(), label_validation_pred.detach().cpu().numpy())
        val_auprc = auc(recall, precision)

        log = 'Epoch: {:03d} Train_Loss: {:.4f} Train_AUROC: {:.4f} Train_AUPRC: {:.4f} Val_AUROC: {:.4f} Val_AUPRC: {:.4f}'
        print(log.format(epoch, loss.item(), train_auroc, train_auprc, val_auroc, val_auprc))

    return ww_, val_auroc, val_auprc, train_auroc, train_auprc





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lmd', type=float, default=1e-4, help='L1 regularization coefficient')
    parser.add_argument('--dataset', type=str, default='UKB', help='multimorbidity dataset')
    args = parser.parse_args()

    # parameter
    lr = 0.01
    weight_decay = 5e-4
    epochs = 500
    test_size = 0.1

    for run in range(100):
        # data
        if args.dataset == 'UKB':
            phenotype_df, multimorbidity_df = read_data(dataset_path='data/ukb_multimorbidity.csv')
        elif args.dataset == 'HuDiNe':
            phenotype_df, multimorbidity_df = read_data(dataset_path='data/hudine_multimorbidity.csv')
        diseasePair_train, diseasePair_test, label_train, label_test = train_test_split_(phenotype_df, multimorbidity_df, test_size=test_size)

        # save test disease-pairs and labels
        index_disease = {i: v for i, v in enumerate(phenotype_df.index)}
        list1 = []
        for row, label in zip(diseasePair_test, label_test):
            code1, code2 = index_disease[row[0]], index_disease[row[1]]
            list1.append([code1, code2, label])
        test_df = pd.DataFrame(list1, columns=['code1', 'code2', 'label'])
        test_df.to_csv('feature_selection_result/' + args.dataset + '/testset_lmd' + str(args.lmd) + '_run' + str(run) + '.csv', index=False)

        phenotype_positive = phenotype_df[phenotype_df > 0].fillna(0).values
        phenotype_negative = phenotype_df[phenotype_df < 0].fillna(0).values
        phenotype = np.concatenate([phenotype_positive, -phenotype_negative], axis=-1)
        scaler = MinMaxScaler()
        features = scaler.fit_transform(phenotype)
        features = torch.Tensor(features).to(device)


        # 10-fold cross validation
        skf = StratifiedKFold(n_splits=10)
        val_auroc_ls = []
        val_auprc_ls = []
        train_auroc_ls = []
        train_auprc_ls = []
        max_ww_ = torch.zeros(features.shape[1]).to(device)

        for train_idx, validation_idx in skf.split(diseasePair_train, label_train):
            diseasePair_train_cv, diseasePair_validation = diseasePair_train[train_idx], diseasePair_train[validation_idx]
            label_train_cv, label_validation = label_train[train_idx], label_train[validation_idx]

            # convert to tensor
            diseasePair_train_cv = torch.LongTensor(diseasePair_train_cv).to(device)
            diseasePair_validation = torch.LongTensor(diseasePair_validation).to(device)
            label_train_cv = torch.FloatTensor(label_train_cv).to(device)
            label_validation = torch.FloatTensor(label_validation).to(device)

            model = Weighted_InnerProd(features.shape[1]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            bce_loss = torch.nn.BCELoss(reduction='mean')

            ww_, val_auroc, val_auprc, train_auroc, train_auprc = train(model, optimizer, bce_loss, epochs, args.lmd, features,
                                        diseasePair_train_cv, label_train_cv, diseasePair_validation, label_validation)
            val_auroc_ls.append(val_auroc)
            val_auprc_ls.append(val_auprc)
            train_auroc_ls.append(train_auroc)
            train_auprc_ls.append(train_auprc)
            max_ww_ = torch.max(max_ww_, ww_)

        val_auroc_mean = np.mean(val_auroc_ls)
        val_auprc_mean = np.mean(val_auprc_ls)
        train_auroc_mean = np.mean(train_auroc_ls)
        train_auprc_mean = np.mean(train_auprc_ls)
        print('10-fold cross validation mean:', train_auroc_mean, train_auprc_mean, val_auroc_mean, val_auprc_mean)

        indicator_df = pd.DataFrame()
        indicator_df['phenotype'] = [col + '*pos' for col in phenotype_df.columns] + [col + '*neg' for col in phenotype_df.columns]
        indicator_df['score'] = max_ww_.detach().cpu().numpy()

        outfile = '~'.join(['lmd' + str(args.lmd), 'run' + str(run), 'val_auroc' + str(round(val_auroc_mean, 4)),
                            'val_auprc' + str(round(val_auprc_mean, 4)), 'train_auroc' + str(round(train_auroc_mean, 4)),
                            'train_auprc' + str(round(train_auprc_mean, 4))]) + '.csv'

        indicator_df.to_csv('feature_selection_result/' + args.dataset + '/' + outfile, index=False)

    print('ok')