import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
plt.rc('font', family='Helvetica')
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.preprocessing import MinMaxScaler



def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


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


_, ukb_selected_phenotypes = get_selected_phenotypes(path='feature_selection_result/UKB/', lmd=0.0001, z_threshold=0.001)
_, hudine_selected_phenotypes = get_selected_phenotypes(path='feature_selection_result/HuDiNe/', lmd=1e-05, z_threshold=0.001)


def get_auroc_auprc_corrCoef(dataset, valueType, phenotype_path='data/disease_phenotype_score_data_processed.csv'):
    phenotype_df = pd.read_csv(phenotype_path, index_col=0)
    phenotype_df.index = [index[:3] for index in phenotype_df.index]
    temp = pd.DataFrame()
    for each in valueType:
        temp1 = phenotype_df.loc[:, phenotype_df.columns.str.contains(each)]
        temp = pd.concat([temp, temp1], axis=1)
    phenotype_df = temp

    if dataset == 'UKB':
        multimorbidity_df = pd.read_csv('data/ukb_multimorbidity.csv')
        selected_phenotypes = ukb_selected_phenotypes
    else:
        multimorbidity_df = pd.read_csv('data/hudine_multimorbidity.csv')
        selected_phenotypes = hudine_selected_phenotypes

    disease_intersection = set(phenotype_df.index) & (set(multimorbidity_df['code1']) | set(multimorbidity_df['code2']))

    multimorbidity_df = multimorbidity_df[multimorbidity_df['code1'].isin(disease_intersection)
                                    & multimorbidity_df['code2'].isin(disease_intersection)]

    disease_order = list(set(multimorbidity_df['code1']) | set(multimorbidity_df['code2']))
    disease_order.sort()
    disease_index = {v: i for i, v in enumerate(disease_order)}

    multimorbidity_df['code1'] = multimorbidity_df['code1'].replace(disease_index)
    multimorbidity_df['code2'] = multimorbidity_df['code2'].replace(disease_index)
    adj = sp.coo_matrix((np.ones(multimorbidity_df.shape[0]), (multimorbidity_df['code1'], multimorbidity_df['code2'])),
                        shape=(len(disease_index), len(disease_index)), dtype=float).toarray()
    adj = adj + adj.T + np.eye(len(disease_index))

    phenotype = phenotype_df.reindex(disease_order)
    phenotype_positive = phenotype[phenotype > 0].fillna(0).values
    phenotype_negative = phenotype[phenotype < 0].fillna(0).values
    phenotype = np.concatenate([phenotype_positive, -phenotype_negative], axis=-1)
    phenotype = pd.DataFrame(phenotype)
    phenotype.columns = [col + '*pos' for col in phenotype_df.columns] + [col + '*neg' for col in phenotype_df.columns]
    phenotype = phenotype.loc[:, phenotype.columns.isin(selected_phenotypes)]
    scaler = MinMaxScaler()
    phenotype = scaler.fit_transform(phenotype)

    adj_pred = sigmoid(phenotype.dot(phenotype.T))

    fpr, tpr, threshold = roc_curve(adj.flatten(), adj_pred.flatten())
    auroc = auc(fpr, tpr)
    precision, recall, threshold = precision_recall_curve(adj.flatten(), adj_pred.flatten())
    auprc = auc(recall, precision)

    return auroc, auprc



def get_auroc_auprc_otherStatistics(dataset, valueType, phenotype_path='data/other_statistics/disease_phenotype_score_data_processed.csv'):
    phenotype_df = pd.read_csv(phenotype_path, index_col=0)
    phenotype_df.index = [index[:3] for index in phenotype_df.index]
    temp = pd.DataFrame()
    for each in valueType:
        temp1 = phenotype_df.loc[:, phenotype_df.columns.str.contains(each)]
        temp = pd.concat([temp, temp1], axis=1)
    phenotype_df = temp

    if dataset == 'UKB':
        multimorbidity_df = pd.read_csv('data/ukb_multimorbidity.csv')
        selected_phenotype = set([each.split('*')[0] for each in ukb_selected_phenotypes])
    else:
        multimorbidity_df = pd.read_csv('data/hudine_multimorbidity.csv')
        selected_phenotype = set([each.split('*')[0] for each in hudine_selected_phenotypes])

    disease_intersection = set(phenotype_df.index) & (set(multimorbidity_df['code1']) | set(multimorbidity_df['code2']))

    multimorbidity_df = multimorbidity_df[multimorbidity_df['code1'].isin(disease_intersection)
                                    & multimorbidity_df['code2'].isin(disease_intersection)]

    disease_order = list(set(multimorbidity_df['code1']) | set(multimorbidity_df['code2']))
    disease_order.sort()
    disease_index = {v: i for i, v in enumerate(disease_order)}

    multimorbidity_df['code1'] = multimorbidity_df['code1'].replace(disease_index)
    multimorbidity_df['code2'] = multimorbidity_df['code2'].replace(disease_index)
    adj = sp.coo_matrix((np.ones(multimorbidity_df.shape[0]), (multimorbidity_df['code1'], multimorbidity_df['code2'])),
                        shape=(len(disease_index), len(disease_index)), dtype=float).toarray()
    adj = adj + adj.T + np.eye(len(disease_index))

    phenotype_df.columns = [col.split('*')[0] for col in phenotype_df.columns]
    phenotype_df = phenotype_df.loc[:, phenotype_df.columns.isin(selected_phenotype)]
    phenotype = phenotype_df.reindex(disease_order).values
    scaler = MinMaxScaler()
    phenotype = scaler.fit_transform(phenotype)

    adj_pred = phenotype.dot(phenotype.T)
    adj_pred = (adj_pred - np.min(adj_pred)) / (np.max(adj_pred) - np.min(adj_pred))
    fpr, tpr, threshold = roc_curve(adj.flatten(), adj_pred.flatten())
    auroc = auc(fpr, tpr)
    precision, recall, threshold = precision_recall_curve(adj.flatten(), adj_pred.flatten())
    auprc = auc(recall, precision)

    return auroc, auprc



def get_auroc_auprc_random(dataset, phenotype_path='data/other_statistics/disease_phenotype_score_data_processed.csv'):
    phenotype_df = pd.read_csv(phenotype_path, index_col=0)
    phenotype_df.index = [index[:3] for index in phenotype_df.index]

    if dataset == 'UKB':
        multimorbidity_df = pd.read_csv('data/ukb_multimorbidity.csv')
        selected_phenotypes = ukb_selected_phenotypes
    else:
        multimorbidity_df = pd.read_csv('data/hudine_multimorbidity.csv')
        selected_phenotypes = hudine_selected_phenotypes

    disease_intersection = set(phenotype_df.index) & (set(multimorbidity_df['code1']) | set(multimorbidity_df['code2']))

    multimorbidity_df = multimorbidity_df[multimorbidity_df['code1'].isin(disease_intersection)
                                    & multimorbidity_df['code2'].isin(disease_intersection)]

    disease_order = list(set(multimorbidity_df['code1']) | set(multimorbidity_df['code2']))
    disease_order.sort()
    disease_index = {v: i for i, v in enumerate(disease_order)}

    multimorbidity_df['code1'] = multimorbidity_df['code1'].replace(disease_index)
    multimorbidity_df['code2'] = multimorbidity_df['code2'].replace(disease_index)
    adj = sp.coo_matrix((np.ones(multimorbidity_df.shape[0]), (multimorbidity_df['code1'], multimorbidity_df['code2'])),
                        shape=(len(disease_index), len(disease_index)), dtype=float).toarray()
    adj = adj + adj.T + np.eye(len(disease_index))

    phenotype_df = phenotype_df.reindex(disease_order)
    phenotype = np.random.randn(phenotype_df.shape[0], len(selected_phenotypes))
    phenotype = pd.DataFrame(phenotype)
    phenotype_positive = phenotype[phenotype > 0].fillna(0).values
    phenotype_negative = phenotype[phenotype < 0].fillna(0).values
    phenotype = np.concatenate([phenotype_positive, -phenotype_negative], axis=-1)
    scaler = MinMaxScaler()
    phenotype = scaler.fit_transform(phenotype)

    adj_pred = phenotype.dot(phenotype.T)
    adj_pred = (adj_pred - np.min(adj_pred)) / (np.max(adj_pred) - np.min(adj_pred))
    fpr, tpr, threshold = roc_curve(adj.flatten(), adj_pred.flatten())
    auroc = auc(fpr, tpr)
    precision, recall, threshold = precision_recall_curve(adj.flatten(), adj_pred.flatten())
    auprc = auc(recall, precision)

    return auroc, auprc




if __name__ == '__main__':

    auroc_ls = []
    auprc_ls = []
    valueType_ls = []
    dataset_ls = []
    method_ls = []

    auroc, auprc = get_auroc_auprc_corrCoef('UKB', phenotype_path='data/disease_phenotype_score_data_processed.csv',
                                            valueType=['continuous~', 'integer~'])
    auroc_ls.append(auroc)
    auprc_ls.append(auprc)
    valueType_ls.append('Continuous')
    dataset_ls.append('UKB')
    method_ls.append('Correlation coefficient')

    auroc, auprc = get_auroc_auprc_corrCoef('UKB', phenotype_path='data/disease_phenotype_score_data_processed.csv',
                                            valueType=['single_category_binary~', 'single_category_unordered~', 'multiple_category~'])
    auroc_ls.append(auroc)
    auprc_ls.append(auprc)
    valueType_ls.append('Binary')
    dataset_ls.append('UKB')
    method_ls.append('Correlation coefficient')

    auroc, auprc = get_auroc_auprc_corrCoef('UKB', phenotype_path='data/disease_phenotype_score_data_processed.csv',
                                            valueType=['single_category_ordered~'])
    auroc_ls.append(auroc)
    auprc_ls.append(auprc)
    valueType_ls.append('Ordered categorical')
    dataset_ls.append('UKB')
    method_ls.append('Correlation coefficient')

    auroc, auprc = get_auroc_auprc_corrCoef('HuDiNe', phenotype_path='data/disease_phenotype_score_data_processed.csv',
                                            valueType=['continuous~', 'integer~'])
    auroc_ls.append(auroc)
    auprc_ls.append(auprc)
    valueType_ls.append('Continuous')
    dataset_ls.append('HuDiNe')
    method_ls.append('Correlation coefficient')

    auroc, auprc = get_auroc_auprc_corrCoef('HuDiNe', phenotype_path='data/disease_phenotype_score_data_processed.csv',
                                            valueType=['single_category_binary~', 'single_category_unordered~', 'multiple_category~'])
    auroc_ls.append(auroc)
    auprc_ls.append(auprc)
    valueType_ls.append('Binary')
    dataset_ls.append('HuDiNe')
    method_ls.append('Correlation coefficient')

    auroc, auprc = get_auroc_auprc_corrCoef('HuDiNe', phenotype_path='data/disease_phenotype_score_data_processed.csv',
                                            valueType=['single_category_ordered~'])
    auroc_ls.append(auroc)
    auprc_ls.append(auprc)
    valueType_ls.append('Ordered categorical')
    dataset_ls.append('HuDiNe')
    method_ls.append('Correlation coefficient')

    auroc, auprc = get_auroc_auprc_otherStatistics('UKB', phenotype_path='data/other_statistics/disease_phenotype_score_data_processed.csv',
                                            valueType=['continuous~', 'integer~'])
    auroc_ls.append(auroc)
    auprc_ls.append(auprc)
    valueType_ls.append('Continuous')
    dataset_ls.append('UKB')
    method_ls.append('Other Statistics')

    auroc, auprc = get_auroc_auprc_otherStatistics('UKB', phenotype_path='data/other_statistics/disease_phenotype_score_data_processed.csv',
                                            valueType=['single_category_binary~', 'single_category_unordered~',
                                                       'multiple_category~'])
    auroc_ls.append(auroc)
    auprc_ls.append(auprc)
    valueType_ls.append('Binary')
    dataset_ls.append('UKB')
    method_ls.append('Other Statistics')

    auroc, auprc = get_auroc_auprc_otherStatistics('UKB', phenotype_path='data/other_statistics/disease_phenotype_score_data_processed.csv',
                                            valueType=['single_category_ordered~'])
    auroc_ls.append(auroc)
    auprc_ls.append(auprc)
    valueType_ls.append('Ordered categorical')
    dataset_ls.append('UKB')
    method_ls.append('Other Statistics')

    auroc, auprc = get_auroc_auprc_otherStatistics('HuDiNe', phenotype_path='data/other_statistics/disease_phenotype_score_data_processed.csv',
                                            valueType=['continuous~', 'integer~'])
    auroc_ls.append(auroc)
    auprc_ls.append(auprc)
    valueType_ls.append('Continuous')
    dataset_ls.append('HuDiNe')
    method_ls.append('Other Statistics')

    auroc, auprc = get_auroc_auprc_otherStatistics('HuDiNe', phenotype_path='data/other_statistics/disease_phenotype_score_data_processed.csv',
                                            valueType=['single_category_binary~', 'single_category_unordered~',
                                                       'multiple_category~'])
    auroc_ls.append(auroc)
    auprc_ls.append(auprc)
    valueType_ls.append('Binary')
    dataset_ls.append('HuDiNe')
    method_ls.append('Other Statistics')

    auroc, auprc = get_auroc_auprc_otherStatistics('HuDiNe', phenotype_path='data/other_statistics/disease_phenotype_score_data_processed.csv',
                                            valueType=['single_category_ordered~'])
    auroc_ls.append(auroc)
    auprc_ls.append(auprc)
    valueType_ls.append('Ordered categorical')
    dataset_ls.append('HuDiNe')
    method_ls.append('Other Statistics')

    df = pd.DataFrame()
    df['Method'] = method_ls
    df['Dataset'] = dataset_ls
    df['ValueType'] = valueType_ls
    df['AUPRC'] = auprc_ls
    df['AUROC'] = auroc_ls

    random_auroc_ukb, random_auprc_ukb = get_auroc_auprc_random('UKB')
    random_auroc_hudine, random_auprc_hudine = get_auroc_auprc_random('HuDiNe')
    print('random ukb:', random_auroc_ukb, random_auprc_ukb)
    print('random hudine:', random_auroc_hudine, random_auprc_hudine)

    # ----------- plot
    df1 = df[(df['Dataset'] == 'UKB')][['ValueType', 'Method', 'AUROC']]
    df1.columns = ['ValueType', 'Method', 'Score']

    df2 = df[(df['Dataset'] == 'HuDiNe')][['ValueType', 'Method', 'AUROC']]
    df2.columns = ['ValueType', 'Method', 'Score']

    df3 = df[(df['Dataset'] == 'UKB')][['ValueType', 'Method', 'AUPRC']]
    df3.columns = ['ValueType', 'Method', 'Score']

    df4 = df[(df['Dataset'] == 'HuDiNe')][['ValueType', 'Method', 'AUPRC']]
    df4.columns = ['ValueType', 'Method', 'Score']

    data_ls = [df1, df2, df3, df4]

    fig, axes = plt.subplots(nrows=2, ncols=2, sharey=False, sharex=True, figsize=(6, 4))
    i = 0
    for ax, data in zip(axes.flatten(), data_ls):
        sns.barplot(data=data, x='ValueType', y='Score', hue='Method', ax=ax, palette=['#6495ed', '#ffa500'])
        ax.get_legend().remove()
        ax.set_xlabel(None)
        ax.tick_params(axis='x', labelrotation=30)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        xlabels = ax.get_xticklabels()
        ax.set_xticklabels(xlabels, rotation=30, ha='right')
        if i < 2:
            ax.set_ylabel('AUROC')
        if i >= 2:
            ax.set_ylabel('AUPRC')
        i += 1

    for score, ax in zip([random_auroc_ukb, random_auroc_hudine, random_auprc_ukb, random_auprc_hudine], axes.flat):
        ax.axhline(y=score, xmin=0, xmax=1, linewidth=0.5, color='r', linestyle='-.')

    fig.savefig('s2.pdf', bbox_inches='tight')
    plt.show()

    print('ok')