import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
from matplotlib import pyplot as plt
plt.rc('font', family='Helvetica')
from itertools import combinations
import scipy.sparse as sp
from scipy.stats import ttest_ind




def plot_differential_ratio():
    file_list = os.listdir('patient_phenotype_test/')

    total = 0
    for each in file_list:
        df = pd.read_csv('patient_phenotype_test/' + each)
        total += df.shape[0]
    bf_threshold = 0.05 / total

    list1 = []
    for each in file_list:
        df = pd.read_csv('patient_phenotype_test/' + each)
        phenotype = set(df['phenotype']).pop()

        x1 = df[df['comorbidity_flag'] == 'comorbidity'].shape[0]
        x2 = df[df['comorbidity_flag'] == 'comorbidity_novel'].shape[0]
        x3 = df[df['comorbidity_flag'] == 'noncomorbidity'].shape[0]

        df = df[(df['p1'] < bf_threshold) & (df['p2'] < bf_threshold) & (df['t1'] * df['t2'] > 0) & (
                    df['code1_coeff*code2_coeff'] > 0)]
        y1 = df[df['comorbidity_flag'] == 'comorbidity'].shape[0]
        y2 = df[df['comorbidity_flag'] == 'comorbidity_novel'].shape[0]
        y3 = df[df['comorbidity_flag'] == 'noncomorbidity'].shape[0]
        list1.append([phenotype, 'Multimorbidity', y1 / x1])
        list1.append([phenotype, 'Novel_multimorbidity', y2 / x2])
        list1.append([phenotype, 'Non-multimorbidity', y3 / x3])

    df = pd.DataFrame(list1, columns=['phenotype', 'group', 'proportion'])

    # .......plot
    list1 = df[df['group'] == 'Multimorbidity']['proportion'].tolist()
    list2 = df[df['group'] == 'Novel_multimorbidity']['proportion'].tolist()
    list3 = df[df['group'] == 'Non-multimorbidity']['proportion'].tolist()
    print('Multimorbidity VS. Novel_multimorbidity', ttest_ind(list1, list2))
    print('Multimorbidity VS. Non-multimorbidity', ttest_ind(list1, list3))
    print('Novel_multimorbidity VS. Non-multimorbidity', ttest_ind(list2, list3))
    df['proportion'] = df['proportion'] * 100

    plt.figure(figsize=(2, 3))
    ax = sns.barplot(x='group', y='proportion', data=df, palette=['#1e90ff', '#ff7f50', '#91e8e1'], errwidth=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(range(3), ['Multimorbidity', 'Novel_multimorbidity', 'Non-multimorbidity'], rotation=30)
    plt.xlabel(None)
    plt.ylabel('Differential proportion')

    x1, x2, y, h = 0, 1, 1.45, 0.05
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, c='k')
    ax.text((x1 + x2) * .5, y + h + 0.01, 'P=3.2e-15', ha='center', va='bottom', color='k')

    x1, x2, y, h = 1, 2, 0.3, 0.05
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, c='k')
    ax.text((x1 + x2) * .5, y + h + 0.01, 'P=3.9e-4', ha='center', va='bottom', color='k')

    x1, x2, y, h = 0, 2, 1.8, 0.05
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, c='k')
    ax.text((x1 + x2) * .5, y + h + 0.01, 'P=5e-18', ha='center', va='bottom', color='k')

    plt.savefig('fig5_a.pdf', bbox_inches='tight')
    plt.show()



def plot_ccg_differential_count():
    # ............ category-category-group differential counts
    disease_category = dict()
    df = pd.read_excel('data/disease_category.xlsx')
    for each in df.values.tolist():
        disease_category[each[0][:3]] = each[1]

    file_list = os.listdir('patient_phenotype_test/')

    total = 0
    for each in file_list:
        df = pd.read_csv('patient_phenotype_test/' + each)
        total += df.shape[0]
    bf_threshold = 0.05 / total

    result = pd.DataFrame()
    for i, each in enumerate(file_list):
        print(i, each)
        df = pd.read_csv('patient_phenotype_test/' + each)
        df = df[df['comorbidity_flag'].isin(['comorbidity', 'comorbidity_novel'])]
        df = df[(df['p1'] < bf_threshold) & (df['p2'] < bf_threshold) & (df['t1'] * df['t2'] > 0) & (
                    df['code1_coeff*code2_coeff'] > 0)]
        df['category1'] = df['code1'].replace(disease_category)
        df['category2'] = df['code2'].replace(disease_category)
        if result.shape[0] == 0:
            result = df
        else:
            result = pd.concat([result, df], axis=0)

    result.to_csv('phenotype_significant_multimorbidity.csv', index=False)

    # ........... plot
    df = pd.read_csv('phenotype_significant_multimorbidity.csv')

    list1 = []
    for each in df['phenotype'].tolist():
        list1.append(each.split('~')[1])
    df['group'] = list1

    group_ls = list(set(df['group']))
    group_ls.sort()

    category_ls = list(set(df['category1']) | set(df['category2']))
    category_ls.sort()
    category_index = {v: i for i, v in enumerate(category_ls)}
    category_combination = list(combinations(category_ls, 2))
    for each in category_ls:
        category_combination.append((each, each))

    list1 = []
    group_phenotype_count = dict()
    for group in group_ls:
        df1 = df[df['group'] == group]
        group_phenotype_count[group] = len(set(df1['phenotype']))
        for c1, c2 in category_combination:
            df2 = df1[((df1['category1'] == c1) & (df1['category2'] == c2)) | (
                        (df1['category1'] == c2) & (df1['category2'] == c1))][['code1', 'code2']]
            df2 = df2.drop_duplicates()
            if df2.shape[0] != 0:
                list1.append([group, c1, c2, df2.shape[0]])

    df = pd.DataFrame(list1, columns=['group', 'category1', 'category2', 'count'])

    fig, axes = plt.subplots(5, 3, sharex=False, sharey=True, figsize=(20, 20))
    cbar_ax = fig.add_axes([1, .3, .015, .4])
    for i, ax in enumerate(axes.flat):
        if i >= 14:
            fig.delaxes(ax)
            continue
        group = group_ls[i]
        df1 = df[df['group'] == group][['category1', 'category2', 'count']]

        df1['category1'] = df1['category1'].replace(category_index)
        df1['category2'] = df1['category2'].replace(category_index)
        data = sp.coo_matrix((df1['count'], (df1['category1'], df1['category2'])),
                             shape=(len(category_index), len(category_index)), dtype=float).toarray()

        data = np.maximum(data, data.transpose())
        data = np.tril(data, k=0)
        data[np.triu_indices(data.shape[0], 1)] = np.nan

        x = pd.DataFrame(data, index=category_ls, columns=category_ls, dtype=float)

        ax.text(x=0.5, y=0.90, s=group + f' ({group_phenotype_count[group]})', fontsize=12, alpha=1, ha='center',
                va='bottom', transform=ax.transAxes)

        sns.heatmap(x, ax=ax, cbar=i == 0, cmap="Blues", vmin=0, vmax=100, cbar_ax=None if i else cbar_ax,
                    annot=True, fmt='.0f', annot_kws={"fontsize": 12})
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, fontsize=8, ha='right')
        if i < 11:
            for l in ax.get_xticklabels():
                l.set_visible(False)

    plt.xticks(rotation=30)
    fig.tight_layout(rect=[0, 0, 0.99, 1])
    plt.savefig('s3.pdf', bbox_inches='tight')
    plt.show()




# ........... plot cases
def plot_cases():
    # the following is an example
    phenotype = '26527'
    df = pd.read_csv('phenotype_significant_multimorbidity.csv')
    df = df[df['phenotype'].str.contains(phenotype)]
    df1 = df[(df['code1'] == 'E78') & (df['code2'] == 'I10')][['code1', 'code2', 'code1_patient_phenotype',
                                                               'code2_patient_phenotype',
                                                               'bothDisease_patient_phenotype',
                                                               'p1', 'p2', 'p3']]
    code1, code2, code1_score, code2_score, both_score, p1, p2, p3 = df1.values.tolist()[0]
    list1 = []
    list1.append([code1, code1_score])
    list1.append([code2, code2_score])
    list1.append(['Both', both_score])
    df = pd.DataFrame(list1, columns=['group', 'score'])

    plt.figure(figsize=(2, 3))
    ax = sns.barplot(x='group', y='score', data=df, palette=['#1e90ff', '#ff7f50', '#91e8e1'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    x1, x2, y, h = 0, 1, 1185, 2
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, c='k')
    ax.text((x1 + x2) * .5, y + h + 0.01, 'P=' + str('%.1e' % p3), ha='center', va='bottom', color='k')

    x1, x2, y, h = 1, 2, 1228, 2
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, c='k')
    ax.text((x1 + x2) * .5, y + h + 0.01, 'P=' + str('%.1e' % p2), ha='center', va='bottom', color='k')

    x1, x2, y, h = 0, 2, 1238, 2
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, c='k')
    ax.text((x1 + x2) * .5, y + h + 0.01, 'P=' + str('%.1e' % p1), ha='center', va='bottom', color='k')

    ax.set_ylim(1150, 1250)
    plt.ylabel('Volume of CSF (whole brain)')
    plt.xlabel(None)
    plt.savefig('s4_1.pdf', bbox_inches='tight')
    plt.show()


plot_differential_ratio()
plot_ccg_differential_count()
plot_cases()

print('ok')