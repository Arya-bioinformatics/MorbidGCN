import pandas as pd
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from itertools import combinations, product
import pickle
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
plt.rc('font', family='Helvetica')
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import os




def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s



def plot(fpr_tpr_dict, p_r_dict, key_order, save_name='a.pdf', title='none'):
    color_order = ['#1e90ff', '#ff7f50', '#1e90ff', '#ff7f50']
    linestyle_order = ['solid', 'solid', 'dotted', 'dotted']

    fig = plt.figure(figsize=(5, 9))

    ax1 = fig.add_subplot(211)
    if title != 'none':
        title = str.capitalize(title)
        ax1.set_title(title.replace('_', ' '), fontsize=15, y=1.01)

    for i, each in enumerate(key_order):
        fpr, tpr, roc_auc = fpr_tpr_dict[each]
        ax1.plot(fpr, tpr, color=color_order[i], linestyle=linestyle_order[i], lw=2, alpha=1., label='%s (%s, AUROC=%0.2f)' % (each[0], each[1], roc_auc))

    ax1.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r', alpha=1.)

    ax1.axis(xmin=-0.05, xmax=1.05, ymin=-0.05, ymax=1.05)
    ax1.set_xlabel('False Positive Rate', fontsize=15)
    ax1.set_ylabel('True Positive Rate', fontsize=15)
    ax1.legend(loc="lower right", fontsize=12)

    # style
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)

    ax2 = fig.add_subplot(212)
    for i, each in enumerate(key_order):
        ps, rs, auprc = p_r_dict[each]
        ax2.plot(rs, ps, color=color_order[i], linestyle=linestyle_order[i], lw=2, alpha=1., label='%s (%s, AUPRC=%0.2f)' % (each[0], each[1], auprc))

    f_scores = np.linspace(0.2, 0.8, num=7)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        ax2.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        ax2.annotate('f1={0:0.1f}'.format(f_score), xy=(0.90, y[45] + 0.02), fontsize=12)

    ax2.axis(xmin=0., xmax=1., ymin=0., ymax=1.)
    ax2.set_xlabel('Recall', fontsize=15)
    ax2.set_ylabel('Precision', fontsize=15)
    ax2.legend(loc="lower left", fontsize=12)

    # style
    ax2.tick_params(axis='x', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)

    fig.savefig(save_name, bbox_inches='tight')
    plt.show()


def get_selected_phenotypes(path, lmd, z_threshold):
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


mesh_icd10_map = defaultdict(set)
with open('data/mesh_icd10_map.txt', 'r') as infile:
    for i, line in enumerate(infile):
        if i < 1:
            continue
        str1 = line.strip('\r\n').split('\t')
        set1 = set([each for each in str1[3].split('|') if '-' not in each])
        for each in str1[1].split('|'):
            mesh_icd10_map[each] |= set1
infile.close()


# mesh disease
mesh_disease_id = defaultdict(set)
mesh_disease_id['adrenal adenoma'].add('D000310')
with open('../../UMLS/MeSH/2021AA/META/MRCONSO.RRF', 'r') as infile:
    for line in infile:
        str1 = line.strip('\r\n')
        str2 = str1.split('|')
        meshid, meshname = str2[10], str.lower(str2[14])
        if len(meshid) != 7:
            continue
        if meshid[0] != 'D':
            continue
        mesh_disease_id[meshname].add(meshid)
infile.close()


# -------------------------------------------------------------------------------------------------------------------- #
#                                            Sab, by gene network overlap
# -------------------------------------------------------------------------------------------------------------------- #
diseasePair_Sab = []
with open('baseline_method/sAB/DataS4_disease_pairs.tsv', 'r') as infile:
    for line in infile:
        if line.startswith('#'):
            continue
        str1 = line.strip('\r\n')
        str2 = str1.split('\t')
        d1, d2, Sab = str.lower(str2[0]), str.lower(str2[1]), float(str2[2])
        id_set1, id_set2 = mesh_disease_id[d1], mesh_disease_id[d2]
        icd10_set1 = set()
        icd10_set2 = set()
        for id in id_set1:
            if id in mesh_icd10_map:
                icd10_set1 |= mesh_icd10_map[id]
        for id in id_set2:
            if id in mesh_icd10_map:
                icd10_set2 |= mesh_icd10_map[id]
        for code1, code2 in product(icd10_set1, icd10_set2):
            if (code1[0] > 'N') | (code2[0] > 'N'):
                continue
            if code1 < code2:
                diseasePair_Sab.append([code1, code2, Sab])
            elif code1 > code2:
                diseasePair_Sab.append([code2, code1, Sab])
infile.close()

diseasePair_Sab = pd.DataFrame(diseasePair_Sab, columns=['code1', 'code2', 'Sab'])
diseasePair_Sab.sort_values('Sab', ascending=False).drop_duplicates(['code1', 'code2']).sort_index()

# phenotype data
phenotype_df = pd.read_csv('data/disease_phenotype_score_data_processed.csv', index_col=0)
phenotype_df.index = [index[:3] for index in phenotype_df.index]

fpr_tpr_dict = {}
p_r_dict = {}

for dataset in ['UKB', 'HuDiNe']:
    if dataset == 'UKB':
        multimorbidity_df = pd.read_csv('data/ukb_multimorbidity.csv')
        selected_phenotypes = ukb_selected_phenotypes
    else:
        multimorbidity_df = pd.read_csv('data/hudine_multimorbidity.csv')
        selected_phenotypes = hudine_selected_phenotypes

    disease_intersection = (set(diseasePair_Sab['code1']) | set(diseasePair_Sab['code2'])) \
                           & (set(multimorbidity_df['code1']) | set(multimorbidity_df['code2'])) & set(phenotype_df.index)

    df = multimorbidity_df[multimorbidity_df['code1'].isin(disease_intersection) & multimorbidity_df['code2'].isin(disease_intersection)]

    disease_order = list(set(df['code1']) | set(df['code2']))
    disease_order.sort()
    disease_index = {v: i for i, v in enumerate(disease_order)}

    df['code1'] = df['code1'].replace(disease_index)
    df['code2'] = df['code2'].replace(disease_index)
    adj = sp.coo_matrix((np.ones(df.shape[0]), (df['code1'], df['code2'])),
                        shape=(len(disease_index), len(disease_index)), dtype=float).toarray()
    adj = adj + adj.T + np.eye(adj.shape[0])

    # phenotype prediction
    phenotype = phenotype_df.reindex(disease_order)
    phenotype_positive = phenotype[phenotype > 0].fillna(0).values
    phenotype_negative = phenotype[phenotype < 0].fillna(0).values
    phenotype = np.concatenate([phenotype_positive, -phenotype_negative], axis=-1)
    phenotype = pd.DataFrame(phenotype)
    phenotype.columns = [col + '*pos' for col in phenotype_df.columns] + [col + '*neg' for col in phenotype_df.columns]
    phenotype = phenotype.loc[:, phenotype.columns.isin(selected_phenotypes)]
    phenotype = phenotype.values

    scaler = MinMaxScaler()
    phenotype = scaler.fit_transform(phenotype)

    adj_pred = sigmoid(phenotype.dot(phenotype.T))
    fpr, tpr, threshold = roc_curve(adj.flatten(), adj_pred.flatten())
    roc_auc = auc(fpr, tpr)
    fpr_tpr_dict[('Phenotype', dataset)] = (fpr, tpr, roc_auc)
    precision, recall, threshold = precision_recall_curve(adj.flatten(), adj_pred.flatten())
    auprc = auc(recall, precision)
    p_r_dict[('Phenotype', dataset)] = (precision, recall, auprc)

    # reference data prediction
    df = diseasePair_Sab[diseasePair_Sab['code1'].isin(disease_order) & diseasePair_Sab['code2'].isin(disease_order)]
    df['code1'] = df['code1'].replace(disease_index)
    df['code2'] = df['code2'].replace(disease_index)

    temp = scaler.fit_transform(np.array(-df['Sab']).reshape(-1,1)).reshape(-1)
    adj_pred = sp.coo_matrix((temp, (df['code1'], df['code2'])),
                             shape=(len(disease_index), len(disease_index)), dtype=float).toarray()
    adj_pred = adj_pred + adj_pred.T + np.eye(adj_pred.shape[0])
    fpr, tpr, threshold = roc_curve(adj.flatten(), adj_pred.flatten())
    roc_auc = auc(fpr, tpr)
    fpr_tpr_dict[('Sab', dataset)] = (fpr, tpr, roc_auc)
    precision, recall, threshold = precision_recall_curve(adj.flatten(), adj_pred.flatten())
    auprc = auc(recall, precision)
    p_r_dict[('Sab', dataset)] = (precision, recall, auprc)

key_order = [('Phenotype', 'UKB'), ('Sab', 'UKB'), ('Phenotype', 'HuDiNe'), ('Sab', 'HuDiNe')]
plot(fpr_tpr_dict, p_r_dict, key_order, 'fig3_a.pdf')



# ------------------------------------------------------------------------------------------------------------------ #
#                                             gene network reconstruction
# ------------------------------------------------------------------------------------------------------------------ #
with open('baseline_method/GeneNetRR/disease_embedding.pkl', 'rb') as f:
    fpkl = pickle.load(f)

# use the disease in supplementary table S1, random1.xlsx, random2.xlsx
umls_id = {'C0752109', 'C0007193', 'C0024117', 'C0162674', 'C0011849', 'C0011881', 'C0014175', 'C0014544', 'C0020179',
           'C0020474', 'C0021364', 'C0348494', 'C0024530', 'C0011570', 'C0026769', 'C0027051', 'C0022680', 'C0032460',
           'C0003873', 'C0036202', 'C0028754', 'C0036341', 'C0005586', 'C0010674', 'C0002871'} # in supplementary Table S1

df = pd.read_excel('baseline_method/GeneNetRR/random1.xlsx')
umls_id = umls_id | (set(df['D1']) | set(df['D2']))

df = pd.read_excel('baseline_method/GeneNetRR/random2.xlsx')
umls_id = umls_id | (set(df['D1']) | set(df['D2']))

umls_icd10 = defaultdict(set)
with open('../../UMLS/ICD10/2021AA/META/MRCONSO.RRF', 'r') as f:
    for line in f:
        str1 = line.strip('\r\n').split('|')
        umls, icd10 = str1[0], str1[13]
        umls_icd10[umls].add(icd10[:3])
    f.close()

disease_GeneEmbedding = dict()
for each in umls_id:
    if each not in umls_icd10:
        print(each)
        continue
    if each not in fpkl:
        print(each)
        continue
    set1 = umls_icd10[each]
    for each1 in set1:
        if each1[0] > 'N':
            continue
        if each1 in disease_GeneEmbedding:
            continue
        disease_GeneEmbedding[each1] = list(fpkl[each])

geneEmbedding_df = pd.DataFrame.from_dict(disease_GeneEmbedding, orient='index')

# phenotype
phenotype_df = pd.read_csv('data/disease_phenotype_score_data_processed.csv', index_col=0)
phenotype_df.index = [index[:3] for index in phenotype_df.index]

fpr_tpr_dict = {}
p_r_dict = {}

for dataset in ['UKB', 'HuDiNe']:
    if dataset == 'UKB':
        multimorbidity_df = pd.read_csv('data/ukb_multimorbidity.csv')
        selected_phenotypes = ukb_selected_phenotypes
    else:
        multimorbidity_df = pd.read_csv('data/hudine_multimorbidity.csv')
        selected_phenotypes = hudine_selected_phenotypes

    disease_intersection = (set(geneEmbedding_df.index)) & (set(multimorbidity_df['code1']) | set(multimorbidity_df['code2'])) \
                            & set(phenotype_df.index)

    df = multimorbidity_df[multimorbidity_df['code1'].isin(disease_intersection) & multimorbidity_df['code2'].isin(disease_intersection)]

    disease_order = list(set(df['code1']) | set(df['code2']))
    disease_order.sort()
    disease_index = {v: i for i, v in enumerate(disease_order)}

    df['code1'] = df['code1'].replace(disease_index)
    df['code2'] = df['code2'].replace(disease_index)
    adj = sp.coo_matrix((np.ones(df.shape[0]), (df['code1'], df['code2'])), shape=(len(disease_index), len(disease_index)),
                        dtype=float).toarray()
    adj = adj + adj.T + np.eye(adj.shape[0])

    # phenotype prediction
    phenotype = phenotype_df.reindex(disease_order)
    phenotype_positive = phenotype[phenotype > 0].fillna(0).values
    phenotype_negative = phenotype[phenotype < 0].fillna(0).values
    phenotype = np.concatenate([phenotype_positive, -phenotype_negative], axis=-1)
    phenotype = pd.DataFrame(phenotype)
    phenotype.columns = [col + '*pos' for col in phenotype_df.columns] + [col + '*neg' for col in phenotype_df.columns]
    phenotype = phenotype.loc[:, phenotype.columns.isin(selected_phenotypes)]
    phenotype = phenotype.values


    scaler = MinMaxScaler()
    phenotype = scaler.fit_transform(phenotype)

    # adj_pred = phenotype.dot(phenotype.T)
    adj_pred = phenotype.dot(phenotype.T)
    fpr, tpr, threshold = roc_curve(adj.flatten(), adj_pred.flatten())
    roc_auc = auc(fpr, tpr)
    fpr_tpr_dict[('Phenotype', dataset)] = (fpr, tpr, roc_auc)
    precision, recall, threshold = precision_recall_curve(adj.flatten(), adj_pred.flatten())
    auprc = auc(recall, precision)
    p_r_dict[('Phenotype', dataset)] = (precision, recall, auprc)

    # reference data prediction
    df = geneEmbedding_df.reindex(disease_order)
    data = df.values
    adj_pred = sigmoid(data.dot(data.T))

    fpr, tpr, threshold = roc_curve(adj.flatten(), adj_pred.flatten())
    roc_auc = auc(fpr, tpr)
    fpr_tpr_dict[('GeneNetRR', dataset)] = (fpr, tpr, roc_auc)
    precision, recall, threshold = precision_recall_curve(adj.flatten(), adj_pred.flatten())
    auprc = auc(recall, precision)
    p_r_dict[('GeneNetRR', dataset)] = (precision, recall, auprc)

key_order = [('Phenotype', 'UKB'), ('GeneNetRR', 'UKB'), ('Phenotype', 'HuDiNe'), ('GeneNetRR', 'HuDiNe')]
plot(fpr_tpr_dict, p_r_dict, key_order, 'fig3_b.pdf')



# ------------------------------------------------------------------------------------------------------------------ #
#                                             human symptom network
# ------------------------------------------------------------------------------------------------------------------ #
# ...... disease symptom similarity
list1 = []
with open('baseline_method/HSDN/41467_2014_BFncomms5212_MOESM1045_ESM.txt', 'r') as infile:
    for i, line in enumerate(infile):
        if i < 1:
            continue
        str1 = line.strip('\r\n').split('\t')
        list1.append([str1[0], str1[1], float(str1[3])])
    infile.close()

df = pd.DataFrame(list1, columns=['symptom', 'disease', 'score'])

symptom_order = list(set(df['symptom']))
disease_order = list(set(df['disease']))
symptom_order.sort()
disease_order.sort()
symptom_index = {v: i for i, v in enumerate(symptom_order)}
disease_index = {v: i for i, v in enumerate(disease_order)}

df['symptom'] = df['symptom'].replace(symptom_index)
df['disease'] = df['disease'].replace(disease_index)

data = sp.coo_matrix((df['score'], (df['disease'], df['symptom'])), shape=(len(disease_index), len(symptom_index)), dtype=float).toarray()
x = cosine_similarity(data)

df = pd.DataFrame(x, index=disease_order, columns=disease_order)

list1 = []
for d1, d2 in combinations(disease_order, 2):
    list1.append([d1, d2, str(df.loc[d1, d2])])

with open('diseasePair_SympSim.txt', 'w+') as outfile:
    outfile.write('disease1\tdisease2\tsimilarity\n')
    for each in list1:
        outfile.write('\t'.join(each) + '\n')
    outfile.close()

# ....... compare
diseasePair_SympSim = []
with open('diseasePair_SympSim.txt', 'r') as infile:
    for i, line in enumerate(infile):
        if i < 1:
            continue
        str1 = line.strip('\r\n').split('\t')
        d1, d2, SympSim = str.lower(str1[0]), str.lower(str1[1]), float(str1[2])
        id_set1, id_set2 = mesh_disease_id[d1], mesh_disease_id[d2]
        icd10_set1 = set()
        icd10_set2 = set()
        for id in id_set1:
            if id in mesh_icd10_map:
                icd10_set1 |= mesh_icd10_map[id]
        for id in id_set2:
            if id in mesh_icd10_map:
                icd10_set2 |= mesh_icd10_map[id]
        for code1, code2 in product(icd10_set1, icd10_set2):
            if (code1[0] > 'N') | (code2[0] > 'N'):
                continue
            if code1 < code2:
                diseasePair_SympSim.append([code1, code2, SympSim])
            elif code1 > code2:
                diseasePair_SympSim.append([code2, code1, SympSim])
infile.close()

diseasePair_SympSim = pd.DataFrame(diseasePair_SympSim, columns=['code1', 'code2', 'SympSim'])
diseasePair_SympSim.sort_values('SympSim', ascending=False).drop_duplicates(['code1', 'code2']).sort_index()

# phenotype data
phenotype_df = pd.read_csv('data/disease_phenotype_score_data_processed.csv', index_col=0)
phenotype_df.index = [index[:3] for index in phenotype_df.index]

fpr_tpr_dict = {}
p_r_dict = {}

for dataset in ['UKB', 'HuDiNe']:
    if dataset == 'UKB':
        multimorbidity_df = pd.read_csv('data/ukb_multimorbidity.csv')
        selected_phenotypes = ukb_selected_phenotypes
    else:
        multimorbidity_df = pd.read_csv('data/hudine_multimorbidity.csv')
        selected_phenotypes = hudine_selected_phenotypes

    disease_intersection = (set(diseasePair_SympSim['code1']) | set(diseasePair_SympSim['code2'])) \
                           & (set(multimorbidity_df['code1']) | set(multimorbidity_df['code2'])) & set(phenotype_df.index)

    df = multimorbidity_df[multimorbidity_df['code1'].isin(disease_intersection) & multimorbidity_df['code2'].isin(disease_intersection)]

    disease_order = list(set(df['code1']) | set(df['code2']))
    disease_order.sort()
    disease_index = {v: i for i, v in enumerate(disease_order)}

    df['code1'] = df['code1'].replace(disease_index)
    df['code2'] = df['code2'].replace(disease_index)
    adj = sp.coo_matrix((np.ones(df.shape[0]), (df['code1'], df['code2'])),
                        shape=(len(disease_index), len(disease_index)), dtype=float).toarray()
    adj = adj + adj.T + np.eye(adj.shape[0])

    # phenotype prediction
    phenotype = phenotype_df.reindex(disease_order)
    phenotype_positive = phenotype[phenotype > 0].fillna(0).values
    phenotype_negative = phenotype[phenotype < 0].fillna(0).values
    phenotype = np.concatenate([phenotype_positive, -phenotype_negative], axis=-1)
    phenotype = pd.DataFrame(phenotype)
    phenotype.columns = [col + '*pos' for col in phenotype_df.columns] + [col + '*neg' for col in phenotype_df.columns]
    phenotype = phenotype.loc[:, phenotype.columns.isin(selected_phenotypes)]
    phenotype = phenotype.values

    scaler = MinMaxScaler()
    phenotype = scaler.fit_transform(phenotype)

    adj_pred = phenotype.dot(phenotype.T)
    fpr, tpr, threshold = roc_curve(adj.flatten(), adj_pred.flatten())
    roc_auc = auc(fpr, tpr)
    fpr_tpr_dict[('Phenotype', dataset)] = (fpr, tpr, roc_auc)
    precision, recall, threshold = precision_recall_curve(adj.flatten(), adj_pred.flatten())
    auprc = auc(recall, precision)
    p_r_dict[('Phenotype', dataset)] = (precision, recall, auprc)

    # reference data prediction
    df = diseasePair_SympSim[diseasePair_SympSim['code1'].isin(disease_order) & diseasePair_SympSim['code2'].isin(disease_order)]
    df['code1'] = df['code1'].replace(disease_index)
    df['code2'] = df['code2'].replace(disease_index)

    adj_pred = sp.coo_matrix((df['SympSim'], (df['code1'], df['code2'])),
                             shape=(len(disease_index), len(disease_index)), dtype=float).toarray()
    adj_pred = adj_pred + adj_pred.T + np.eye(adj_pred.shape[0])
    fpr, tpr, threshold = roc_curve(adj.flatten(), adj_pred.flatten())
    roc_auc = auc(fpr, tpr)
    fpr_tpr_dict[('HSDN', dataset)] = (fpr, tpr, roc_auc)
    precision, recall, threshold = precision_recall_curve(adj.flatten(), adj_pred.flatten())
    auprc = auc(recall, precision)
    p_r_dict[('HSDN', dataset)] = (precision, recall, auprc)

key_order = [('Phenotype', 'UKB'), ('HSDN', 'UKB'), ('Phenotype', 'HuDiNe'), ('HSDN', 'HuDiNe')]
plot(fpr_tpr_dict, p_r_dict, key_order, 'fig3_c.pdf')



# -------------------------------------------------------------------------------------------------------------------- #
#                   SimilarityFusion, by six different types of biological data (ontological,
#          phenotypic, literature co-occurrence, genetic association, gene expression and drug indication data)
# -------------------------------------------------------------------------------------------------------------------- #
df = pd.read_csv('baseline_method/FusedSim/diseaseNameMappingToICD.csv', dtype=str)
mesh_icd9_map = dict(zip(df['Transcriptomic space'].map(str.lower), df['ICD9']))

icd9_icd10_map = dict()
with open('data/icd9cm_icd10_map.txt', 'r') as infile:
    for i, line in enumerate(infile):
        if i < 1:
            continue
        str1 = line.strip('\r\n').split('\t')
        set1 = set([each for each in str1[1].split('|') if '-' not in each])
        set2 = set([each for each in str1[-1].split('|') if '-' not in each])
        if (len(set1) == 0) | (len(set2) == 0):
            continue
        for each in set1:
            icd9_icd10_map[each] = set2
    infile.close()

# fused similarity
df = pd.read_csv('baseline_method/FusedSim/fusedSimilarity.csv', index_col=0)

diseasePair_fusedSimilarity = []
for d1, d2 in product(df.index, df.columns):
    if d1 == d2:
        continue
    id1 = mesh_icd9_map[str.lower(d1)]
    id2 = mesh_icd9_map[str.lower(d2)]
    id1 = ''.join(['0']*(3-len(id1))) + id1
    id2 = ''.join(['0'] * (3 - len(id1))) + id2
    if (id1 not in icd9_icd10_map) | (id2 not in icd9_icd10_map):
        continue
    icd10_set1 = icd9_icd10_map[id1]
    icd10_set2 = icd9_icd10_map[id2]
    for code1, code2 in product(icd10_set1, icd10_set2):
        if (code1[0] > 'N') | (code2[0] > 'N'):
            continue
        if code1 == code2:
            continue
        if code1 < code2:
            diseasePair_fusedSimilarity.append([code1, code2, df.loc[d1, d2]])
        else:
            diseasePair_fusedSimilarity.append([code2, code1, df.loc[d1, d2]])

diseasePair_fusedSimilarity = pd.DataFrame(diseasePair_fusedSimilarity, columns=['code1', 'code2', 'FusedSim'])
diseasePair_fusedSimilarity.sort_values('FusedSim', ascending=False).drop_duplicates(['code1', 'code2']).sort_index()

# phenotype data
phenotype_df = pd.read_csv('data/disease_phenotype_score_data_processed.csv', index_col=0)
phenotype_df.index = [index[:3] for index in phenotype_df.index]

fpr_tpr_dict = {}
p_r_dict = {}

for dataset in ['UKB', 'HuDiNe']:
    if dataset == 'UKB':
        multimorbidity_df = pd.read_csv('data/ukb_multimorbidity.csv')
        selected_phenotypes = ukb_selected_phenotypes
    else:
        multimorbidity_df = pd.read_csv('data/hudine_multimorbidity.csv')
        selected_phenotypes = hudine_selected_phenotypes

    disease_intersection = (set(diseasePair_fusedSimilarity['code1']) | set(diseasePair_fusedSimilarity['code2'])) \
                           & (set(multimorbidity_df['code1']) | set(multimorbidity_df['code2'])) & set(phenotype_df.index)

    df = multimorbidity_df[multimorbidity_df['code1'].isin(disease_intersection) & multimorbidity_df['code2'].isin(disease_intersection)]

    disease_order = list(set(df['code1']) | set(df['code2']))
    disease_order.sort()
    disease_index = {v: i for i, v in enumerate(disease_order)}

    df['code1'] = df['code1'].replace(disease_index)
    df['code2'] = df['code2'].replace(disease_index)
    adj = sp.coo_matrix((np.ones(df.shape[0]), (df['code1'], df['code2'])),
                        shape=(len(disease_index), len(disease_index)), dtype=float).toarray()
    adj = adj + adj.T + np.eye(adj.shape[0])

    # phenotype prediction
    phenotype = phenotype_df.reindex(disease_order)
    phenotype_positive = phenotype[phenotype > 0].fillna(0).values
    phenotype_negative = phenotype[phenotype < 0].fillna(0).values
    phenotype = np.concatenate([phenotype_positive, -phenotype_negative], axis=-1)
    phenotype = pd.DataFrame(phenotype)
    phenotype.columns = [col + '*pos' for col in phenotype_df.columns] + [col + '*neg' for col in phenotype_df.columns]
    phenotype = phenotype.loc[:, phenotype.columns.isin(selected_phenotypes)]
    phenotype = phenotype.values

    scaler = MinMaxScaler()
    phenotype = scaler.fit_transform(phenotype)

    adj_pred = phenotype.dot(phenotype.T)
    fpr, tpr, threshold = roc_curve(adj.flatten(), adj_pred.flatten())
    roc_auc = auc(fpr, tpr)
    fpr_tpr_dict[('Phenotype', dataset)] = (fpr, tpr, roc_auc)
    precision, recall, threshold = precision_recall_curve(adj.flatten(), adj_pred.flatten())
    auprc = auc(recall, precision)
    p_r_dict[('Phenotype', dataset)] = (precision, recall, auprc)

    # reference data prediction
    df = diseasePair_fusedSimilarity[diseasePair_fusedSimilarity['code1'].isin(disease_order)
                                     & diseasePair_fusedSimilarity['code2'].isin(disease_order)]
    df['code1'] = df['code1'].replace(disease_index)
    df['code2'] = df['code2'].replace(disease_index)

    adj_pred = sp.coo_matrix((df['FusedSim'], (df['code1'], df['code2'])),
                                 shape=(len(disease_index), len(disease_index)), dtype=float).toarray()
    adj_pred = adj_pred + adj_pred.T + np.eye(adj_pred.shape[0])
    fpr, tpr, threshold = roc_curve(adj.flatten(), adj_pred.flatten())
    roc_auc = auc(fpr, tpr)
    fpr_tpr_dict[('FusedSim', dataset)] = (fpr, tpr, roc_auc)
    precision, recall, threshold = precision_recall_curve(adj.flatten(), adj_pred.flatten())
    auprc = auc(recall, precision)
    p_r_dict[('FusedSim', dataset)] = (precision, recall, auprc)

key_order = [('Phenotype', 'UKB'), ('FusedSim', 'UKB'), ('Phenotype', 'HuDiNe'), ('FusedSim', 'HuDiNe')]
plot(fpr_tpr_dict, p_r_dict, key_order, 'fig3_d.pdf')



print('ok')