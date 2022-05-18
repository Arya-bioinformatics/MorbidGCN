import pandas as pd
import numpy as np
import random
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
plt.rc('font', family='Helvetica')
import seaborn as sns
import re
from itertools import combinations
import rbo





def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s



def get_selection_result(dataset, z_threshold=0.001, outfile='a.csv'):
    result = []
    file_list = os.listdir('feature_selection_result/' + dataset)

    for i, f in enumerate(file_list):
        if 'testset' in f:
            continue
        lmd, run, val_auroc, val_auprc, train_auroc, train_auprc = f.replace('.csv', '').split('~')
        lmd = lmd.replace('lmd', '')
        run = run.replace('run', '')
        val_auroc = val_auroc.replace('val_auroc', '')
        val_auprc = val_auprc.replace('val_auprc', '')
        train_auroc = train_auroc.replace('train_auroc', '')
        train_auprc = train_auprc.replace('train_auprc', '')

        df = pd.read_csv('feature_selection_result/' + dataset + '/' + f)
        selected_phenotypes = df[df['score'] > z_threshold]['phenotype'].tolist()
        result.append([lmd, run, val_auroc, val_auprc, train_auroc, train_auprc, len(selected_phenotypes)])

    result_df = pd.DataFrame(result, columns=['lmd', 'run', 'val_auroc', 'val_auprc', 'train_auroc', 'train_auprc', 'phenotype'])
    result_df.to_csv(outfile, index=False)


def plot_UKB_selected_pheNum(outpath):
    # ------- plot the number of the selected features at different z threshold
    df1 = pd.read_csv('a0.1.csv')
    df2 = pd.read_csv('a0.01.csv')
    df3 = pd.read_csv('a0.001.csv')
    df4 = pd.read_csv('a0.0001.csv')
    df5 = pd.read_csv('a0.00001.csv')
    df6 = pd.read_csv('a0.csv')

    list1 = [df1, df2, df3, df4, df5, df6]
    list2 = [0.1, 0.01, 0.001, 1e-4, 1e-5, 0]

    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(5, 4))
    i = 0
    for df, ax, threshold in zip(list1, axes.flat, list2):
        df_mean = df.groupby('lmd').mean()
        df_min = df.groupby('lmd').min()
        df_max = df.groupby('lmd').max()

        # plot the number of selected phenotypes
        x = np.log10(df_mean.index)
        y_phenotype = df_mean['phenotype']
        y_phenotype_min = df_min['phenotype']
        y_phenotype_max = df_max['phenotype']

        ax.errorbar(x, y_phenotype, yerr=[y_phenotype - y_phenotype_min, y_phenotype_max - y_phenotype], fmt='o:', capsize=2)
        ax.set_xticks([-7, -6, -5, -4, -3, -2])
        ax.set_xticklabels([-7, -6, -5, -4, -3, -2], fontsize=12)

        if i >= 3:
            ax.set_xlabel('Log10(' + r"$\lambda$" + ')', fontsize=12)
            ax.tick_params(axis="x", labelsize=12)
        ax.text(-7, 1400, 'Z > ' + str(threshold))

        if i % 3 == 0:
            ax.set_ylabel('Number of \nselected phenotypes', fontsize=12)
        i += 1

    plt.savefig(outpath, bbox_inches='tight')
    plt.show()


def plot_HuDiNe_selected_pheNum(outpath):
    # ------- plot the number of the selected features at different z threshold
    df1 = pd.read_csv('b0.1.csv')
    df2 = pd.read_csv('b0.01.csv')
    df3 = pd.read_csv('b0.001.csv')
    df4 = pd.read_csv('b0.0001.csv')
    df5 = pd.read_csv('b0.00001.csv')
    df6 = pd.read_csv('b0.csv')

    list1 = [df1, df2, df3, df4, df5, df6]
    list2 = [0.1, 0.01, 0.001, 1e-4, 1e-5, 0]

    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(5, 4))
    i = 0
    for df, ax, threshold in zip(list1, axes.flat, list2):

        df_mean = df.groupby('lmd').mean()
        df_min = df.groupby('lmd').min()
        df_max = df.groupby('lmd').max()

        # plot the number of selected phenotypes
        x = np.log10(df_mean.index)
        y_phenotype = df_mean['phenotype']
        y_phenotype_min = df_min['phenotype']
        y_phenotype_max = df_max['phenotype']

        ax.errorbar(x, y_phenotype, yerr=[y_phenotype - y_phenotype_min, y_phenotype_max - y_phenotype], fmt='o:', capsize=2)
        ax.set_xticks([-7, -6, -5, -4, -3, -2])
        ax.set_xticklabels([-7, -6, -5, -4, -3, -2], fontsize=12)
        if i >= 3:
            ax.set_xlabel('Log10(' + r"$\lambda$" + ')', fontsize=12)
            ax.tick_params(axis="x", labelsize=12)
        ax.text(-7, 450, 'Z > ' + str(threshold))

        if i % 3 == 0:
            ax.set_ylabel('Number of \nselected phenotypes', fontsize=12)
        i += 1

    plt.savefig(outpath, bbox_inches='tight')
    plt.show()


def plot_UKB_performance(outpath):
    # ------- plot auroc, auprc for 100 runs at different lmd
    result_df = pd.read_csv('a0.001.csv')
    df_mean = result_df.groupby('lmd').mean()
    df_min = result_df.groupby('lmd').min()
    df_max = result_df.groupby('lmd').max()

    x = np.log10(df_mean.index)

    # plot val auroc and auprc
    y_val_auroc = df_mean['val_auroc']
    y_val_auroc_min = df_min['val_auroc']
    y_val_auroc_max = df_max['val_auroc']

    y_val_auprc = df_mean['val_auprc']
    y_val_auprc_min = df_min['val_auprc']
    y_val_auprc_max = df_max['val_auprc']

    # plot train auroc and auprc
    y_train_auroc = df_mean['train_auroc']
    y_train_auroc_min = df_min['train_auroc']
    y_train_auroc_max = df_max['train_auroc']

    y_train_auprc = df_mean['train_auprc']
    y_train_auprc_min = df_min['train_auprc']
    y_train_auprc_max = df_max['train_auprc']

    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(4, 5))

    ax0.errorbar(x+0.1, y_train_auroc, yerr=[y_train_auroc - y_train_auroc_min, y_train_auroc_max - y_train_auroc],
             fmt='o:', capsize=2, label='Train')
    ax0.errorbar(x-0.1, y_val_auroc, yerr=[y_val_auroc - y_val_auroc_min, y_val_auroc_max - y_val_auroc],
             fmt='o:', capsize=2, label='Validation')
    ax1.errorbar(x+0.1, y_train_auprc, yerr=[y_train_auprc - y_train_auprc_min, y_train_auprc_max - y_train_auprc],
             fmt='o:', capsize=2, label='Train')
    ax1.errorbar(x-0.1, y_val_auprc, yerr=[y_val_auprc - y_val_auprc_min, y_val_auprc_max - y_val_auprc],
             fmt='o:', capsize=2, label='Validation')

    ax0.set_yticks([0.86, 0.88, 0.90, 0.92])
    ax0.set_yticklabels([0.86, 0.88, 0.90, 0.92], fontsize=12)
    ax0.set_ylabel('AUROC', fontsize=12)
    ax0.legend()

    ax1.set_yticks([0.78, 0.80, 0.82, 0.84, 0.86])
    ax1.set_yticklabels([0.78, 0.80, 0.82, 0.84, 0.86], fontsize=12)
    ax1.set_ylabel('AUPRC', fontsize=12)
    ax1.legend()

    ax1.set_xticks([-7, -6, -5, -4, -3, -2])
    ax1.set_xticklabels([-7, -6, -5, -4, -3, -2], fontsize=12)
    ax1.set_xlabel('Log10(' + r"$\lambda$" + ')', fontsize=12)

    plt.savefig(outpath, bbox_inches='tight')
    plt.show()


def plot_HuDiNe_performance(outpath):
    # ------- plot auroc, auprc for 100 runs at different lmd
    result_df = pd.read_csv('b0.001.csv')
    df_mean = result_df.groupby('lmd').mean()
    df_min = result_df.groupby('lmd').min()
    df_max = result_df.groupby('lmd').max()

    x = np.log10(df_mean.index)

    # plot val auroc and auprc
    y_val_auroc = df_mean['val_auroc']
    y_val_auroc_min = df_min['val_auroc']
    y_val_auroc_max = df_max['val_auroc']

    y_val_auprc = df_mean['val_auprc']
    y_val_auprc_min = df_min['val_auprc']
    y_val_auprc_max = df_max['val_auprc']

    # plot train auroc and auprc
    y_train_auroc = df_mean['train_auroc']
    y_train_auroc_min = df_min['train_auroc']
    y_train_auroc_max = df_max['train_auroc']

    y_train_auprc = df_mean['train_auprc']
    y_train_auprc_min = df_min['train_auprc']
    y_train_auprc_max = df_max['train_auprc']


    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=False, figsize=(4, 5))

    ax0.errorbar(x+0.1, y_train_auroc, yerr=[y_train_auroc - y_train_auroc_min, y_train_auroc_max - y_train_auroc],
             fmt='o:', capsize=2, label='Train')
    ax0.errorbar(x-0.1, y_val_auroc, yerr=[y_val_auroc - y_val_auroc_min, y_val_auroc_max - y_val_auroc],
             fmt='o:', capsize=2, label='Validation')
    ax1.errorbar(x+0.1, y_train_auprc, yerr=[y_train_auprc - y_train_auprc_min, y_train_auprc_max - y_train_auprc],
             fmt='o:', capsize=2, label='Train')
    ax1.errorbar(x-0.1, y_val_auprc, yerr=[y_val_auprc - y_val_auprc_min, y_val_auprc_max - y_val_auprc],
             fmt='o:', capsize=2, label='Validation')

    ax0.set_yticks([0.71, 0.72, 0.73, 0.74])
    ax0.set_yticklabels([0.71, 0.72, 0.73, 0.74], fontsize=12)
    ax0.set_ylabel('AUROC', fontsize=12)
    ax0.legend()

    ax1.set_yticks([0.56, 0.57, 0.58, 0.59])
    ax1.set_yticklabels([0.56, 0.57, 0.58, 0.59], fontsize=12)
    ax1.set_ylabel('AUPRC', fontsize=12)
    ax1.legend()

    ax1.set_xticks([-7, -6, -5, -4, -3, -2])
    ax1.set_xticklabels([-7, -6, -5, -4, -3, -2], fontsize=12)
    ax1.set_xlabel('Log10(' + r"$\lambda$" + ')', fontsize=12)

    plt.savefig(outpath, bbox_inches='tight')
    plt.show()



def plot_pheSelectedTimes(dataset, lmd, z_threshold, outpath):
    # ------- repeatability of selected phenotypes for different runs (train/test split)
    result = []
    file_list = os.listdir('feature_selection_result/' + dataset)
    file_list = [f for f in file_list if 'lmd' + str(lmd) + '~' in f]
    for f in file_list:
        df = pd.read_csv('feature_selection_result/' + dataset + '/' + f)
        selected_phenotypes = df[df['score'] > z_threshold]['phenotype'].tolist()
        result += selected_phenotypes

    phenotype_times = []
    for each in set(result):
        phenotype_times.append([each, result.count(each)])
    df = pd.DataFrame(phenotype_times, columns=['Phenotype', 'Times'])

    # plot
    plt.figure(figsize=(6, 6))
    labels = ['100', '90~99', '50~99', '0~49']
    sizes = []
    for each in labels:
        if each == '100':
            s = df[df['Times'] == 100].shape[0] / df.shape[0]
            sizes.append(s)
        else:
            min_time, max_time = [int(time) for time in each.split('~')]
            s = df[(df['Times'] >= min_time) & (df['Times'] <= max_time)].shape[0] / df.shape[0]
            sizes.append(s)
    colors = ['#6495ed', '#ffa500', '#7b68ee', '#ff7f50']
    explode = [0.05, 0., 0., 0.]
    patches, text1, text2 = plt.pie(sizes, explode, labels, colors, autopct='%3.2f%%', shadow=False, startangle=90,
                                pctdistance=0.8)
    for each in text1:
        each.set_fontsize(15)
    for each in text2:
        each.set_fontsize(15)
    plt.axis('equal')

    plt.savefig(outpath, bbox_inches='tight')
    plt.show()



def plot_rank_similarity(dataset, lmd, z_threshold, outpath):
    # -------- rank consistency of selected phenotypes for different runs (train/test split)
    selected_phenotypes_rank_dict = dict()
    file_list = os.listdir('feature_selection_result/' + dataset)
    file_list = [f for f in file_list if 'lmd' + str(lmd) + '~' in f]
    for f in file_list:
        df = pd.read_csv('feature_selection_result/' + dataset + '/' + f)
        df = df.sort_values(by='score', ascending=False)
        selected_phenotypes = df[df['score'] > z_threshold]['phenotype'].tolist()
        run = re.findall(r'~run(.*?)~', f)[0]
        selected_phenotypes_rank_dict[run] = selected_phenotypes

    result = []
    for run1, run2 in combinations(selected_phenotypes_rank_dict.keys(), 2):
        list1 = selected_phenotypes_rank_dict[run1]
        list2 = selected_phenotypes_rank_dict[run2]
        s = rbo.RankingSimilarity(list1, list2).rbo()
        result.append([run1, run2, s])

    df = pd.DataFrame(result, columns=['run1', 'run2', 'rbo_similarity'])

    # plot
    plt.figure(figsize=(3, 2))
    ax = sns.distplot(df['rbo_similarity'], color="#1e90ff")
    ax.set_xlabel('RBO Score')
    ax.set_ylabel('Feature density')
    for loc, spine in ax.spines.items():
        if (loc == 'top') | (loc == 'right'):
            spine.set_color('none')

    plt.savefig(outpath, bbox_inches='tight')
    plt.show()


def plot_selectedPhe_performance(dataset, lmd, z_threshold, outpath):
    # ------- compare prediction result of selected phenotypes, all phenotypes, randomly selected phenotypes
    phenotype_df = pd.read_csv('data/disease_phenotype_score_data_processed.csv', index_col=0)
    phenotype_df.index = [index[:3] for index in phenotype_df.index]

    features = np.concatenate([phenotype_df[phenotype_df > 0].fillna(0).values,
                               -phenotype_df[phenotype_df < 0].fillna(0).values], axis=-1)
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    features = pd.DataFrame(features)
    features.columns = [col + '*pos' for col in phenotype_df.columns] + [col + '*neg' for col in phenotype_df.columns]
    features.index = phenotype_df.index
    disease_index = {v: i for i, v in enumerate(phenotype_df.index)}

    result = []
    file_list = os.listdir('feature_selection_result/' + dataset)
    file_list = [f for f in file_list if 'lmd' + str(lmd) + '~' in f]
    for i, f in enumerate(file_list):
        df = pd.read_csv('feature_selection_result/' + dataset + '/' + f)
        # selected phenotypes and randomly selected phenotypes
        selected_phenotypes = df[df['score'] > z_threshold]['phenotype'].tolist()
        all_phenotypes = df['phenotype'].tolist()
        random_phenotypes_fromAll = random.sample(all_phenotypes, len(selected_phenotypes))
        random_phenotypes_fromRemain = random.sample(set(all_phenotypes) - set(selected_phenotypes), len(selected_phenotypes))

        df1 = pd.read_csv('feature_selection_result/' + dataset + '/testset_' + '_'.join(f.split('~')[:2]) + '.csv')
        df1['code1'] = df1['code1'].replace(disease_index)
        df1['code2'] = df1['code2'].replace(disease_index)

        # prediction of selected phenotypes
        data = features.loc[:, features.columns.isin(selected_phenotypes)].values
        adj_pred = sigmoid(data.dot(data.T))
        label_pred = []
        label_true = []
        for code1_index, code2_index, label in df1.values.tolist():
            label_pred.append(adj_pred[code1_index, code2_index])
            label_true.append(label)
        auroc1 = roc_auc_score(label_true, label_pred)
        precision, recall, threshold = precision_recall_curve(label_true, label_pred)
        auprc1 = auc(recall, precision)

        # prediction of randomly selected phenotypes from all
        data = features.loc[:, features.columns.isin(random_phenotypes_fromAll)].values
        adj_pred = sigmoid(data.dot(data.T))
        label_pred = []
        label_true = []
        for code1_index, code2_index, label in df1.values.tolist():
            label_pred.append(adj_pred[code1_index, code2_index])
            label_true.append(label)
        auroc2 = roc_auc_score(label_true, label_pred)
        precision, recall, threshold = precision_recall_curve(label_true, label_pred)
        auprc2 = auc(recall, precision)

        # prediction of randomly selected phenotypes from remain
        data = features.loc[:, features.columns.isin(random_phenotypes_fromRemain)].values
        adj_pred = sigmoid(data.dot(data.T))
        label_pred = []
        label_true = []
        for code1_index, code2_index, label in df1.values.tolist():
            label_pred.append(adj_pred[code1_index, code2_index])
            label_true.append(label)
        auroc3 = roc_auc_score(label_true, label_pred)
        precision, recall, threshold = precision_recall_curve(label_true, label_pred)
        auprc3 = auc(recall, precision)

        # prediction of all phenotypes
        data = features.loc[:, :].values
        adj_pred = sigmoid(data.dot(data.T))
        label_pred = []
        label_true = []
        for code1_index, code2_index, label in df1.values.tolist():
            label_pred.append(adj_pred[code1_index, code2_index])
            label_true.append(label)
        auroc4 = roc_auc_score(label_true, label_pred)
        precision, recall, threshold = precision_recall_curve(label_true, label_pred)
        auprc4 = auc(recall, precision)

        result.append(['selected', 'AUROC', auroc1])
        result.append(['selected', 'AUPRC', auprc1])
        result.append(['random_all', 'AUROC', auroc2])
        result.append(['random_all', 'AUPRC', auprc2])
        result.append(['random_remain', 'AUROC', auroc3])
        result.append(['random_remain', 'AUPRC', auprc3])
        result.append(['all', 'AUROC', auroc4])
        result.append(['all', 'AUPRC', auprc4])

    df = pd.DataFrame(result, columns=['Group', 'Measure', 'Score'])

    # statistics
    df1 = df[df['Measure'] == 'AUROC']
    df2 = df[df['Measure'] == 'AUPRC']
    dict1 = df1.groupby('Group')['Score'].mean().to_dict()
    dict2 = df2.groupby('Group')['Score'].mean().to_dict()

    print('selected feature, auroc increase:', dict1['selected'] - dict1['all'])
    print('selected feature, auprc increase:', dict2['selected'] - dict2['all'])
    print('random from remained feature, auroc increase:', dict1['all'] - dict1['random_remain'])
    print('random from remained feature, auprc increase:', dict2['all'] - dict2['random_remain'])
    print('random from all feature, auroc increase:', dict1['all'] - dict1['random_all'])
    print('random from all feature, auprc increase:', dict2['all'] - dict2['random_all'])

    #  plot
    df1 = df[df['Measure'] == 'AUROC']
    df2 = df[df['Measure'] == 'AUPRC']

    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(3, 5))
    sns.boxplot(x='Group', y='Score', data=df1, order=['selected', 'all', 'random_all', 'random_remain'], width=0.4,
                linewidth=0.8, fliersize=0.2, saturation=1., ax=ax0, palette=['#3cb371', '#da70d6', '#ff7f50', '#f15c80'])
    sns.boxplot(x='Group', y='Score', data=df2, order=['selected', 'all', 'random_all', 'random_remain'], width=0.4,
                linewidth=0.8, fliersize=0.2, saturation=1., ax=ax1, palette=['#3cb371', '#da70d6', '#ff7f50', '#f15c80'])
    ax0.set_xlabel(None)
    ax0.set_ylabel('AUROC')

    ax1.tick_params(axis='x', labelrotation=30)
    ax1.set_xlabel(None)
    ax1.set_ylabel('AUPRC')
    xlabels = ['Selected', 'All', 'Randomly selected\nfrom all', 'Randomly selected\nfrom remain']
    ax1.set_xticklabels(xlabels, rotation=30, ha='right')

    plt.savefig(outpath, bbox_inches='tight')
    plt.show()




def plot_selectedPhe_weights(dataset, lmd, z_threshold, outpath):
    # -------- plot weight coefficients and groups of the selected phenotypes
    file_list = os.listdir('feature_selection_result/' + dataset)
    file_list = [f for f in file_list if f.startswith('lmd' + str(lmd))]
    df = pd.DataFrame()
    for i, f in enumerate(file_list):
        df1 = pd.read_csv('feature_selection_result/' + dataset + '/' + f, index_col=0)
        df = pd.concat([df, df1], axis=1)
    s = df.max(axis=1)
    s = s[s > z_threshold]

    # plot
    df = pd.DataFrame(s, index=s.index, columns='weights')
    list1 = []
    for index in df.index:
        score = df.loc[index].values[0]
        group = index.split('~')[1]
        list1.append([index, group, score])

    df = pd.DataFrame(list1, columns=['Phenotype', 'Group', 'Score'])
    df = df.sort_values(by='Group')

    plt.figure(figsize=(8, 4))
    ax = sns.scatterplot(x='Phenotype', y='Score', hue='Group', data=df, palette='tab10')
    ax.xaxis.set_ticks_position('none')
    ax.set_xticklabels([])

    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8], fontsize=15)
    ax.set_xlabel('Phenotypes', fontsize=15)
    ax.set_ylabel('Weights', fontsize=15)

    ax.legend(bbox_to_anchor=(1, 1), ncol=1, frameon=False, fontsize=10)

    plt.savefig(outpath, bbox_inches='tight')
    plt.show()










if __name__ == '__main__':

    dataset = 'UKB'
    get_selection_result(z_threshold=0.1, outfile='a0.1.csv')
    get_selection_result(z_threshold=0.01, outfile='a0.01.csv')
    get_selection_result(z_threshold=0.001, outfile='a0.001.csv')
    get_selection_result(z_threshold=0.0001, outfile='a0.0001.csv')
    get_selection_result(z_threshold=0.00001, outfile='a0.00001.csv')
    get_selection_result(z_threshold=0, outfile='a0.csv')

    dataset = 'HuDiNe'
    get_selection_result(z_threshold=0.1, outfile='b0.1.csv')
    get_selection_result(z_threshold=0.01, outfile='b0.01.csv')
    get_selection_result(z_threshold=0.001, outfile='b0.001.csv')
    get_selection_result(z_threshold=0.0001, outfile='b0.0001.csv')
    get_selection_result(z_threshold=0.00001, outfile='b0.00001.csv')
    get_selection_result(z_threshold=0, outfile='b0.csv')

    # supplemetary fig. 1
    plot_UKB_selected_pheNum('s1_a.pdf')
    plot_HuDiNe_selected_pheNum('s1_b.pdf')
    plot_UKB_performance('s1_c.pdf')
    plot_HuDiNe_performance('s1_d.pdf')

    # fig. 2
    plot_pheSelectedTimes(dataset='UKB', lmd=0.0001, z_threshold=0.001, outpath='fig2_a.pdf')
    plot_pheSelectedTimes(dataset='HuDiNe', lmd=1e-05, z_threshold=0.001, outpath='fig2_b.pdf')
    plot_rank_similarity(dataset='UKB', lmd=0.0001, z_threshold=0.001, outpath='fig2_c.pdf')
    plot_rank_similarity(dataset='HuDiNe', lmd=1e-05, z_threshold=0.001, outpath='fig2_d.pdf')
    plot_selectedPhe_performance(dataset='UKB', lmd=0.0001, z_threshold=0.001, outpath='fig2_e.pdf')
    plot_selectedPhe_performance(dataset='HuDiNe', lmd=1e-05, z_threshold=0.001, outpath='fig2_f.pdf')
    plot_selectedPhe_weights(dataset='UKB', lmd=0.0001, z_threshold=0.001, outpath='fig2_g.pdf')
    plot_selectedPhe_weights(dataset='HuDiNe', lmd=1e-05, z_threshold=0.001, outpath='fig2_h.pdf')



    print('ok')