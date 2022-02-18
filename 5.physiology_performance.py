import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations, product
import pickle
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
plt.rc('font', family='Helvetica')
import seaborn as sns
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
import os





# disease category
disease_category = dict()
df = pd.read_excel('data/disease_category.xlsx')
for disease, category in df.values.tolist():
    disease_category[disease[:3]] = category

# phenotype data
phenotype_df = pd.read_csv('data/disease_phenotype_score_data_processed.csv', index_col=0)
phenotype_df.index = [index[:3] for index in phenotype_df.index]

# multimorbidity data
UKB_multimorbidity_df = pd.read_csv('data/ukb_multimorbidity.csv')
hudine_multimorbidity_df = pd.read_csv('data/hudine_multimorbidity.csv')

# icd9-icd10 map
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

# mesh-icd10 map
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

# mesh disease name - id map
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

# umls - icd10 map
umls_icd10 = defaultdict(set)
with open('../../UMLS/ICD10/2021AA/META/MRCONSO.RRF', 'r') as f:
    for line in f:
        str1 = line.strip('\r\n').split('|')
        umls, icd10 = str1[0], str1[13]
        umls_icd10[umls].add(icd10[:3])
    f.close()

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


# -------------------------------------------------------------------------------------------------------------------- #
#                                            Sab, by gene network overlap
# -------------------------------------------------------------------------------------------------------------------- #
def SabSim(dataset):
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

    if dataset == 'UKB':
        multimorbidity_df = UKB_multimorbidity_df.copy()
        selected_phenotypes = ukb_selected_phenotypes
    elif dataset == 'HuDiNe':
        multimorbidity_df = hudine_multimorbidity_df.copy()
        selected_phenotypes = hudine_selected_phenotypes

    disease_intersection = set(phenotype_df.index) & (set(multimorbidity_df['code1']) | set(multimorbidity_df['code2'])) \
                           & (set(diseasePair_Sab['code1']) | set(diseasePair_Sab['code2']))

    multimorbidity_df = multimorbidity_df[multimorbidity_df['code1'].isin(disease_intersection)
                                    & multimorbidity_df['code2'].isin(disease_intersection)]

    # multimorbid disease-pairs
    multimorbid_diseasePair = set(zip(multimorbidity_df['code1'], multimorbidity_df['code2']))

    # disease-pairs labels
    disease_order = list(set(multimorbidity_df['code1']) | set(multimorbidity_df['code2']))
    disease_order.sort()
    disease_index = {v: i for i, v in enumerate(disease_order)}

    samePhysiology_diseasePairs = []
    diffPhysiology_diseasePairs = []
    samePhysiology_label = []
    diffPhysiology_label = []
    for code1, code2 in combinations(disease_order, 2):
        if disease_category[code1] == disease_category[code2]:
            samePhysiology_diseasePairs.append((code1, code2))
            if ((code1, code2) in multimorbid_diseasePair) | ((code2, code1) in multimorbid_diseasePair):
                samePhysiology_label.append(1)
            else:
                samePhysiology_label.append(0)
        else:
            diffPhysiology_diseasePairs.append((code1, code2))
            if ((code1, code2) in multimorbid_diseasePair) | ((code2, code1) in multimorbid_diseasePair):
                diffPhysiology_label.append(1)
            else:
                diffPhysiology_label.append(0)

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

    samePhysiology_label_phenotypePred = []
    diffPhysiology_label_phenotypePred = []
    for code1, code2 in samePhysiology_diseasePairs:
        samePhysiology_label_phenotypePred.append(sigmoid(phenotype[disease_index[code1]].dot(phenotype[disease_index[code2]])))
    for code1, code2 in diffPhysiology_diseasePairs:
        diffPhysiology_label_phenotypePred.append(sigmoid(phenotype[disease_index[code1]].dot(phenotype[disease_index[code2]])))

    # Sab prediction
    diseasePair_Sab = diseasePair_Sab[diseasePair_Sab['code1'].isin(disease_order)
                                                    & diseasePair_Sab['code2'].isin(disease_order)]
    diseasePair_Sab['Sab'] = scaler.fit_transform(np.array(-diseasePair_Sab['Sab']).reshape(-1,1)).reshape(-1)

    Sab_dict = {(code1, code2): sim for code1, code2, sim in diseasePair_Sab.values.tolist()}

    samePhysiology_label_SabPred = []
    diffPhysiology_label_SabPred = []
    for code1, code2 in samePhysiology_diseasePairs:
        if (code1, code2) in Sab_dict:
            samePhysiology_label_SabPred.append(Sab_dict[(code1, code2)])
        elif (code2, code1) in Sab_dict:
            samePhysiology_label_SabPred.append(Sab_dict[(code2, code1)])
        else:
            samePhysiology_label_SabPred.append(0)
            print('Sab.......', code1, code2)

    for code1, code2 in diffPhysiology_diseasePairs:
        if (code1, code2) in Sab_dict:
            diffPhysiology_label_SabPred.append(Sab_dict[(code1, code2)])
        elif (code2, code1) in Sab_dict:
            diffPhysiology_label_SabPred.append(Sab_dict[(code2, code1)])
        else:
            diffPhysiology_label_SabPred.append(0)
            print('Sab.......', code1, code2)

    fpr, tpr, threshold = roc_curve(samePhysiology_label, samePhysiology_label_phenotypePred)
    auroc_samePhysiologyPhenotype = auc(fpr, tpr)
    precision, recall, threshold = precision_recall_curve(samePhysiology_label, samePhysiology_label_phenotypePred)
    auprc_samePhysiologyPhenotype = auc(recall, precision)
    print(dataset, 'phenotype, same Physiology:', auroc_samePhysiologyPhenotype, auprc_samePhysiologyPhenotype)

    fpr, tpr, threshold = roc_curve(diffPhysiology_label, diffPhysiology_label_phenotypePred)
    auroc_diffPhysiologyPhenotype = auc(fpr, tpr)
    precision, recall, threshold = precision_recall_curve(diffPhysiology_label, diffPhysiology_label_phenotypePred)
    auprc_diffPhysiologyPhenotype = auc(recall, precision)
    print(dataset, 'phenotype, diff Physiology:', auroc_diffPhysiologyPhenotype, auprc_diffPhysiologyPhenotype)

    fpr, tpr, threshold = roc_curve(samePhysiology_label, samePhysiology_label_SabPred)
    auroc_samePhysiologySab = auc(fpr, tpr)
    precision, recall, threshold = precision_recall_curve(samePhysiology_label, samePhysiology_label_SabPred)
    auprc_samePhysiologySab = auc(recall, precision)
    print(dataset, 'Sab, same Physiology:', auroc_samePhysiologySab, auprc_samePhysiologySab)

    fpr, tpr, threshold = roc_curve(diffPhysiology_label, diffPhysiology_label_SabPred)
    auroc_diffPhysiologySab = auc(fpr, tpr)
    precision, recall, threshold = precision_recall_curve(diffPhysiology_label, diffPhysiology_label_SabPred)
    auprc_diffPhysiologySab = auc(recall, precision)
    print(dataset, 'Sab, diff Physiology:', auroc_diffPhysiologySab, auprc_diffPhysiologySab)

    return auroc_samePhysiologyPhenotype, auroc_samePhysiologySab, auroc_diffPhysiologyPhenotype, auroc_diffPhysiologySab, \
           auprc_samePhysiologyPhenotype, auprc_samePhysiologySab, auprc_diffPhysiologyPhenotype, auprc_diffPhysiologySab



# ------------------------------------------------------------------------------------------------------------------ #
#                                             gene network reconstruction
# ------------------------------------------------------------------------------------------------------------------ #
def GeneNetRR(dataset):
    with open('baseline_method/GeneNetRR/disease_embedding.pkl', 'rb') as f:
        fpkl = pickle.load(f)

    # use the disease in supplementary table S1, random1.xlsx, random2.xlsx
    umls_id = {'C0752109', 'C0007193', 'C0024117', 'C0162674', 'C0011849', 'C0011881', 'C0014175', 'C0014544',
               'C0020179', 'C0020474', 'C0021364', 'C0348494', 'C0024530', 'C0011570', 'C0026769', 'C0027051',
               'C0022680', 'C0032460', 'C0003873', 'C0036202', 'C0028754', 'C0036341', 'C0005586', 'C0010674',
               'C0002871'}  # in supplementary Table S1

    df = pd.read_excel('baseline_method/GeneNetRR/random1.xlsx')
    umls_id = umls_id | (set(df['D1']) | set(df['D2']))

    df = pd.read_excel('baseline_method/GeneNetRR/random2.xlsx')
    umls_id = umls_id | (set(df['D1']) | set(df['D2']))

    disease_GeneEmbedding = dict()
    for each in umls_id:
        if each not in umls_icd10:
            # print(each)
            continue
        if each not in fpkl:
            # print(each)
            continue
        set1 = umls_icd10[each]
        for each1 in set1:
            if each1[0] > 'N':
                continue
            if each1 in disease_GeneEmbedding:
                continue
            disease_GeneEmbedding[each1] = list(fpkl[each])

    geneEmbedding_df = pd.DataFrame.from_dict(disease_GeneEmbedding, orient='index')

    if dataset == 'UKB':
        multimorbidity_df = UKB_multimorbidity_df.copy()
        selected_phenotypes = ukb_selected_phenotypes
    elif dataset == 'HuDiNe':
        multimorbidity_df = hudine_multimorbidity_df.copy()
        selected_phenotypes = hudine_selected_phenotypes

    disease_intersection = set(phenotype_df.index) & (set(multimorbidity_df['code1']) | set(multimorbidity_df['code2'])) \
                           & set(geneEmbedding_df.index)

    multimorbidity_df = multimorbidity_df[multimorbidity_df['code1'].isin(disease_intersection)
                                    & multimorbidity_df['code2'].isin(disease_intersection)]

    # multimorbid disease-pairs
    multimorbid_diseasePair = set(zip(multimorbidity_df['code1'], multimorbidity_df['code2']))

    # disease-pairs labels
    disease_order = list(set(multimorbidity_df['code1']) | set(multimorbidity_df['code2']))
    disease_order.sort()
    disease_index = {v: i for i, v in enumerate(disease_order)}

    samePhysiology_diseasePairs = []
    diffPhysiology_diseasePairs = []
    samePhysiology_label = []
    diffPhysiology_label = []
    for code1, code2 in combinations(disease_order, 2):
        if disease_category[code1] == disease_category[code2]:
            samePhysiology_diseasePairs.append((code1, code2))
            if ((code1, code2) in multimorbid_diseasePair) | ((code2, code1) in multimorbid_diseasePair):
                samePhysiology_label.append(1)
            else:
                samePhysiology_label.append(0)
        else:
            diffPhysiology_diseasePairs.append((code1, code2))
            if ((code1, code2) in multimorbid_diseasePair) | ((code2, code1) in multimorbid_diseasePair):
                diffPhysiology_label.append(1)
            else:
                diffPhysiology_label.append(0)

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

    samePhysiology_label_phenotypePred = []
    diffPhysiology_label_phenotypePred = []
    for code1, code2 in samePhysiology_diseasePairs:
        samePhysiology_label_phenotypePred.append(sigmoid(phenotype[disease_index[code1]].dot(phenotype[disease_index[code2]])))
    for code1, code2 in diffPhysiology_diseasePairs:
        diffPhysiology_label_phenotypePred.append(sigmoid(phenotype[disease_index[code1]].dot(phenotype[disease_index[code2]])))

    # gene network reconstruction and representation prediction
    samePhysiology_label_geneNetRRPred = []
    diffPhysiology_label_geneNetRRPred = []
    for code1, code2 in samePhysiology_diseasePairs:
        samePhysiology_label_geneNetRRPred.append(sigmoid(geneEmbedding_df.loc[code1].dot(geneEmbedding_df.loc[code2])))
    for code1, code2 in diffPhysiology_diseasePairs:
        diffPhysiology_label_geneNetRRPred.append(sigmoid(geneEmbedding_df.loc[code1].dot(geneEmbedding_df.loc[code2])))

    fpr, tpr, threshold = roc_curve(samePhysiology_label, samePhysiology_label_phenotypePred)
    auroc_samePhysiologyPhenotype = auc(fpr, tpr)
    precision, recall, threshold = precision_recall_curve(samePhysiology_label, samePhysiology_label_phenotypePred)
    auprc_samePhysiologyPhenotype = auc(recall, precision)
    print(dataset, 'phenotype, same Physiology:', auroc_samePhysiologyPhenotype, auprc_samePhysiologyPhenotype)

    fpr, tpr, threshold = roc_curve(diffPhysiology_label, diffPhysiology_label_phenotypePred)
    auroc_diffPhysiologyPhenotype = auc(fpr, tpr)
    precision, recall, threshold = precision_recall_curve(diffPhysiology_label, diffPhysiology_label_phenotypePred)
    auprc_diffPhysiologyPhenotype = auc(recall, precision)
    print(dataset, 'phenotype, diff Physiology:', auroc_diffPhysiologyPhenotype, auprc_diffPhysiologyPhenotype)

    fpr, tpr, threshold = roc_curve(samePhysiology_label, samePhysiology_label_geneNetRRPred)
    auroc_samePhysiologyGeneNetRR = auc(fpr, tpr)
    precision, recall, threshold = precision_recall_curve(samePhysiology_label, samePhysiology_label_geneNetRRPred)
    auprc_samePhysiologyGeneNetRR = auc(recall, precision)
    print(dataset, 'geneNetRR, same Physiology:', auroc_samePhysiologyGeneNetRR, auprc_samePhysiologyGeneNetRR)

    fpr, tpr, threshold = roc_curve(diffPhysiology_label, diffPhysiology_label_geneNetRRPred)
    auroc_diffPhysiologyGeneNetRR = auc(fpr, tpr)
    precision, recall, threshold = precision_recall_curve(diffPhysiology_label, diffPhysiology_label_geneNetRRPred)
    auprc_diffPhysiologyGeneNetRR = auc(recall, precision)
    print(dataset, 'geneNetRR, diff Physiology:', auroc_diffPhysiologyGeneNetRR, auprc_diffPhysiologyGeneNetRR)

    return auroc_samePhysiologyPhenotype, auroc_samePhysiologyGeneNetRR, auroc_diffPhysiologyPhenotype, \
           auroc_diffPhysiologyGeneNetRR, auprc_samePhysiologyPhenotype, auprc_samePhysiologyGeneNetRR, \
           auprc_diffPhysiologyPhenotype, auprc_diffPhysiologyGeneNetRR



# ------------------------------------------------------------------------------------------------------------------ #
#                                             human symptom network
# ------------------------------------------------------------------------------------------------------------------ #
def HSDN(dataset):
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

    if dataset == 'UKB':
        multimorbidity_df = UKB_multimorbidity_df.copy()
        selected_phenotypes = ukb_selected_phenotypes
    elif dataset == 'HuDiNe':
        multimorbidity_df = hudine_multimorbidity_df.copy()
        selected_phenotypes = hudine_selected_phenotypes

    disease_intersection = set(phenotype_df.index) & (set(multimorbidity_df['code1']) | set(multimorbidity_df['code2'])) \
                           & (set(diseasePair_SympSim['code1']) | set(diseasePair_SympSim['code2']))

    multimorbidity_df = multimorbidity_df[multimorbidity_df['code1'].isin(disease_intersection)
                                    & multimorbidity_df['code2'].isin(disease_intersection)]

    # multimorbid disease-pairs
    multimorbid_diseasePair = set(zip(multimorbidity_df['code1'], multimorbidity_df['code2']))

    # disease-pairs labels
    disease_order = list(set(multimorbidity_df['code1']) | set(multimorbidity_df['code2']))
    disease_order.sort()
    disease_index = {v: i for i, v in enumerate(disease_order)}

    samePhysiology_diseasePairs = []
    diffPhysiology_diseasePairs = []
    samePhysiology_label = []
    diffPhysiology_label = []
    for code1, code2 in combinations(disease_order, 2):
        if disease_category[code1] == disease_category[code2]:
            samePhysiology_diseasePairs.append((code1, code2))
            if ((code1, code2) in multimorbid_diseasePair) | ((code2, code1) in multimorbid_diseasePair):
                samePhysiology_label.append(1)
            else:
                samePhysiology_label.append(0)
        else:
            diffPhysiology_diseasePairs.append((code1, code2))
            if ((code1, code2) in multimorbid_diseasePair) | ((code2, code1) in multimorbid_diseasePair):
                diffPhysiology_label.append(1)
            else:
                diffPhysiology_label.append(0)

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

    samePhysiology_label_phenotypePred = []
    diffPhysiology_label_phenotypePred = []
    for code1, code2 in samePhysiology_diseasePairs:
        samePhysiology_label_phenotypePred.append(sigmoid(phenotype[disease_index[code1]].dot(phenotype[disease_index[code2]])))
    for code1, code2 in diffPhysiology_diseasePairs:
        diffPhysiology_label_phenotypePred.append(sigmoid(phenotype[disease_index[code1]].dot(phenotype[disease_index[code2]])))

    # SympSim prediction
    diseasePair_SympSim = diseasePair_SympSim[diseasePair_SympSim['code1'].isin(disease_order)
                                                    & diseasePair_SympSim['code2'].isin(disease_order)]
    SympSim_dict = {(code1, code2): sim for code1, code2, sim in diseasePair_SympSim.values.tolist()}

    samePhysiology_label_SympSimPred = []
    diffPhysiology_label_SympSimPred = []
    for code1, code2 in samePhysiology_diseasePairs:
        if (code1, code2) in SympSim_dict:
            samePhysiology_label_SympSimPred.append(SympSim_dict[(code1, code2)])
        elif (code2, code1) in SympSim_dict:
            samePhysiology_label_SympSimPred.append(SympSim_dict[(code2, code1)])
        else:
            samePhysiology_label_SympSimPred.append(0)
            raise Exception

    for code1, code2 in diffPhysiology_diseasePairs:
        if (code1, code2) in SympSim_dict:
            diffPhysiology_label_SympSimPred.append(SympSim_dict[(code1, code2)])
        elif (code2, code1) in SympSim_dict:
            diffPhysiology_label_SympSimPred.append(SympSim_dict[(code2, code1)])
        else:
            diffPhysiology_label_SympSimPred.append(0)
            raise Exception

    fpr, tpr, threshold = roc_curve(samePhysiology_label, samePhysiology_label_phenotypePred)
    auroc_samePhysiologyPhenotype = auc(fpr, tpr)
    precision, recall, threshold = precision_recall_curve(samePhysiology_label, samePhysiology_label_phenotypePred)
    auprc_samePhysiologyPhenotype = auc(recall, precision)
    print(dataset, 'phenotype, same Physiology:', auroc_samePhysiologyPhenotype, auprc_samePhysiologyPhenotype)

    fpr, tpr, threshold = roc_curve(diffPhysiology_label, diffPhysiology_label_phenotypePred)
    auroc_diffPhysiologyPhenotype = auc(fpr, tpr)
    precision, recall, threshold = precision_recall_curve(diffPhysiology_label, diffPhysiology_label_phenotypePred)
    auprc_diffPhysiologyPhenotype = auc(recall, precision)
    print(dataset, 'phenotype, diff Physiology:', auroc_diffPhysiologyPhenotype, auprc_diffPhysiologyPhenotype)

    fpr, tpr, threshold = roc_curve(samePhysiology_label, samePhysiology_label_SympSimPred)
    auroc_samePhysiologySympSim = auc(fpr, tpr)
    precision, recall, threshold = precision_recall_curve(samePhysiology_label, samePhysiology_label_SympSimPred)
    auprc_samePhysiologySympSim = auc(recall, precision)
    print(dataset, 'HSDN, same Physiology:', auroc_samePhysiologySympSim, auprc_samePhysiologySympSim)

    fpr, tpr, threshold = roc_curve(diffPhysiology_label, diffPhysiology_label_SympSimPred)
    auroc_diffPhysiologySympSim = auc(fpr, tpr)
    precision, recall, threshold = precision_recall_curve(diffPhysiology_label, diffPhysiology_label_SympSimPred)
    auprc_diffPhysiologySympSim = auc(recall, precision)
    print(dataset, 'HSDN, diff Physiology:', auroc_diffPhysiologySympSim, auprc_diffPhysiologySympSim)

    return auroc_samePhysiologyPhenotype, auroc_samePhysiologySympSim, auroc_diffPhysiologyPhenotype, auroc_diffPhysiologySympSim, \
           auprc_samePhysiologyPhenotype, auprc_samePhysiologySympSim, auprc_diffPhysiologyPhenotype, auprc_diffPhysiologySympSim



# -------------------------------------------------------------------------------------------------------------------- #
#                   SimilarityFusion, by six different types of biological data (ontological,
#          phenotypic, literature co-occurrence, genetic association, gene expression and drug indication data)
# -------------------------------------------------------------------------------------------------------------------- #
def FusedSim(dataset):
    df = pd.read_csv('baseline_method/FusedSim/diseaseNameMappingToICD.csv', dtype=str)
    mesh_icd9_map = dict(zip(df['Transcriptomic space'].map(str.lower), df['ICD9']))

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

    if dataset == 'UKB':
        multimorbidity_df = UKB_multimorbidity_df.copy()
        selected_phenotypes = ukb_selected_phenotypes
    elif dataset == 'HuDiNe':
        multimorbidity_df = hudine_multimorbidity_df.copy()
        selected_phenotypes = hudine_selected_phenotypes

    disease_intersection = set(phenotype_df.index) & (set(multimorbidity_df['code1']) | set(multimorbidity_df['code2'])) \
                           & (set(diseasePair_fusedSimilarity['code1']) | set(diseasePair_fusedSimilarity['code2']))

    multimorbidity_df = multimorbidity_df[multimorbidity_df['code1'].isin(disease_intersection)
                                    & multimorbidity_df['code2'].isin(disease_intersection)]

    # multimorbid disease-pairs
    multimorbid_diseasePair = set(zip(multimorbidity_df['code1'], multimorbidity_df['code2']))

    # disease-pairs labels
    disease_order = list(set(multimorbidity_df['code1']) | set(multimorbidity_df['code2']))
    disease_order.sort()
    disease_index = {v: i for i, v in enumerate(disease_order)}

    samePhysiology_diseasePairs = []
    diffPhysiology_diseasePairs = []
    samePhysiology_label = []
    diffPhysiology_label = []
    for code1, code2 in combinations(disease_order, 2):
        if disease_category[code1] == disease_category[code2]:
            samePhysiology_diseasePairs.append((code1, code2))
            if ((code1, code2) in multimorbid_diseasePair) | ((code2, code1) in multimorbid_diseasePair):
                samePhysiology_label.append(1)
            else:
                samePhysiology_label.append(0)
        else:
            diffPhysiology_diseasePairs.append((code1, code2))
            if ((code1, code2) in multimorbid_diseasePair) | ((code2, code1) in multimorbid_diseasePair):
                diffPhysiology_label.append(1)
            else:
                diffPhysiology_label.append(0)

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

    samePhysiology_label_phenotypePred = []
    diffPhysiology_label_phenotypePred = []
    for code1, code2 in samePhysiology_diseasePairs:
        samePhysiology_label_phenotypePred.append(sigmoid(phenotype[disease_index[code1]].dot(phenotype[disease_index[code2]])))
    for code1, code2 in diffPhysiology_diseasePairs:
        diffPhysiology_label_phenotypePred.append(sigmoid(phenotype[disease_index[code1]].dot(phenotype[disease_index[code2]])))

    # FusedSim prediction
    diseasePair_fusedSimilarity = diseasePair_fusedSimilarity[diseasePair_fusedSimilarity['code1'].isin(disease_order)
                                      & diseasePair_fusedSimilarity['code2'].isin(disease_order)]
    fusedSim_dict = {(code1, code2): sim for code1, code2, sim in diseasePair_fusedSimilarity.values.tolist()}

    samePhysiology_label_fusedSimPred = []
    diffPhysiology_label_fusedSimPred = []
    for code1, code2 in samePhysiology_diseasePairs:
        if (code1, code2) in fusedSim_dict:
            samePhysiology_label_fusedSimPred.append(fusedSim_dict[(code1, code2)])
        elif (code2, code1) in fusedSim_dict:
            samePhysiology_label_fusedSimPred.append(fusedSim_dict[(code2, code1)])
        else:
            samePhysiology_label_fusedSimPred.append(0)
            print('FusedSim......', code1, code2)

    for code1, code2 in diffPhysiology_diseasePairs:
        if (code1, code2) in fusedSim_dict:
            diffPhysiology_label_fusedSimPred.append(fusedSim_dict[(code1, code2)])
        elif (code2, code1) in fusedSim_dict:
            diffPhysiology_label_fusedSimPred.append(fusedSim_dict[(code2, code1)])
        else:
            diffPhysiology_label_fusedSimPred.append(0)
            print('FusedSim......', code1, code2)

    fpr, tpr, threshold = roc_curve(samePhysiology_label, samePhysiology_label_phenotypePred)
    auroc_samePhysiologyPhenotype = auc(fpr, tpr)
    precision, recall, threshold = precision_recall_curve(samePhysiology_label, samePhysiology_label_phenotypePred)
    auprc_samePhysiologyPhenotype = auc(recall, precision)
    print(dataset, 'phenotype, same Physiology:', auroc_samePhysiologyPhenotype, auprc_samePhysiologyPhenotype)

    fpr, tpr, threshold = roc_curve(diffPhysiology_label, diffPhysiology_label_phenotypePred)
    auroc_diffPhysiologyPhenotype = auc(fpr, tpr)
    precision, recall, threshold = precision_recall_curve(diffPhysiology_label, diffPhysiology_label_phenotypePred)
    auprc_diffPhysiologyPhenotype = auc(recall, precision)
    print(dataset, 'phenotype, diff Physiology:', auroc_diffPhysiologyPhenotype, auprc_diffPhysiologyPhenotype)

    fpr, tpr, threshold = roc_curve(samePhysiology_label, samePhysiology_label_fusedSimPred)
    auroc_samePhysiologyFusedSim = auc(fpr, tpr)
    precision, recall, threshold = precision_recall_curve(samePhysiology_label, samePhysiology_label_fusedSimPred)
    auprc_samePhysiologyFusedSim = auc(recall, precision)
    print(dataset, 'FusedSim, same Physiology:', auroc_samePhysiologyFusedSim, auprc_samePhysiologyFusedSim)

    fpr, tpr, threshold = roc_curve(diffPhysiology_label, diffPhysiology_label_fusedSimPred)
    auroc_diffPhysiologyFusedSim = auc(fpr, tpr)
    precision, recall, threshold = precision_recall_curve(diffPhysiology_label, diffPhysiology_label_fusedSimPred)
    auprc_diffPhysiologyFusedSim = auc(recall, precision)
    print(dataset, 'FusedSim, diff Physiology:', auroc_diffPhysiologyFusedSim, auprc_diffPhysiologyFusedSim)

    return auroc_samePhysiologyPhenotype, auroc_samePhysiologyFusedSim, auroc_diffPhysiologyPhenotype, \
           auroc_diffPhysiologyFusedSim, auprc_samePhysiologyPhenotype, auprc_samePhysiologyFusedSim, \
           auprc_diffPhysiologyPhenotype, auprc_diffPhysiologyFusedSim




def plot_new(ref_order_ls, data_ls, save_name):
    ref_order_ls = ref_order_ls * 2
    fig, axes = plt.subplots(nrows=2, ncols=4, sharey=True, sharex=False, figsize=(6, 4))
    i = 0
    for ax, data in zip(axes.flatten(), data_ls):
        ref = ref_order_ls[i]
        df = pd.DataFrame()
        df['Group'] = [ref, ref, 'Phenotype', 'Phenotype']
        df['Physiology'] = ['Same-physiological', 'Cross-physiological', 'Same-physiological', 'Cross-physiological']
        df['Score'] = data
        sns.barplot(data=df, x='Group', y='Score', hue='Physiology', ax=ax, palette=['#1e90ff', '#ff7f50'])
        ax.get_legend().remove()
        ax.set_xlabel(None)
        ax.tick_params(axis='x', labelrotation=30)
        ax.set_ylim(0.2, 0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if (i != 0) & (i != 4):
            ax.spines['left'].set_visible(False)
            ax.set_ylabel(None)
            ax.get_yaxis().set_visible(False)
        if i == 0:
            ax.set_ylabel('AUROC')
        if i == 4:
            ax.set_ylabel('AUPRC')
        if i < 4:
            ax.set(xticklabels=[])
        i += 1

    fig.savefig(save_name, bbox_inches='tight')
    plt.show()




# ....... Sab
ukb_auc_samePhysiologyPhenotype, ukb_auc_samePhysiologyRef, ukb_auc_diffPhysiologyPhenotype, ukb_auc_diffPhysiologyRef, \
ukb_auprc_samePhysiologyPhenotype, ukb_auprc_samePhysiologyRef, ukb_auprc_diffPhysiologyPhenotype, ukb_auprc_diffPhysiologyRef = SabSim('UKB')

hudine_auc_samePhysiologyPhenotype, hudine_auc_samePhysiologyRef, hudine_auc_diffPhysiologyPhenotype, hudine_auc_diffPhysiologyRef, \
hudine_auprc_samePhysiologyPhenotype, hudine_auprc_samePhysiologyRef, hudine_auprc_diffPhysiologyPhenotype, hudine_auprc_diffPhysiologyRef = SabSim('HuDiNe')

Sab_auroc_ukb = [ukb_auc_samePhysiologyRef, ukb_auc_diffPhysiologyRef, ukb_auc_samePhysiologyPhenotype, ukb_auc_diffPhysiologyPhenotype]
Sab_auprc_ukb = [ukb_auprc_samePhysiologyRef, ukb_auprc_diffPhysiologyRef, ukb_auprc_samePhysiologyPhenotype, ukb_auprc_diffPhysiologyPhenotype]
Sab_auroc_hudine = [hudine_auc_samePhysiologyRef, hudine_auc_diffPhysiologyRef, hudine_auc_samePhysiologyPhenotype, hudine_auc_diffPhysiologyPhenotype]
Sab_auprc_hudine = [hudine_auprc_samePhysiologyRef, hudine_auprc_diffPhysiologyRef, hudine_auprc_samePhysiologyPhenotype, hudine_auprc_diffPhysiologyPhenotype]


# ....... GeneNetRR
ukb_auc_samePhysiologyPhenotype, ukb_auc_samePhysiologyRef, ukb_auc_diffPhysiologyPhenotype, ukb_auc_diffPhysiologyRef, \
ukb_auprc_samePhysiologyPhenotype, ukb_auprc_samePhysiologyRef, ukb_auprc_diffPhysiologyPhenotype, ukb_auprc_diffPhysiologyRef = GeneNetRR('UKB')

hudine_auc_samePhysiologyPhenotype, hudine_auc_samePhysiologyRef, hudine_auc_diffPhysiologyPhenotype, hudine_auc_diffPhysiologyRef, \
hudine_auprc_samePhysiologyPhenotype, hudine_auprc_samePhysiologyRef, hudine_auprc_diffPhysiologyPhenotype, hudine_auprc_diffPhysiologyRef = GeneNetRR('HuDiNe')

GeneNetRR_auroc_ukb = [ukb_auc_samePhysiologyRef, ukb_auc_diffPhysiologyRef, ukb_auc_samePhysiologyPhenotype, ukb_auc_diffPhysiologyPhenotype]
GeneNetRR_auprc_ukb = [ukb_auprc_samePhysiologyRef, ukb_auprc_diffPhysiologyRef, ukb_auprc_samePhysiologyPhenotype, ukb_auprc_diffPhysiologyPhenotype]
GeneNetRR_auroc_hudine = [hudine_auc_samePhysiologyRef, hudine_auc_diffPhysiologyRef, hudine_auc_samePhysiologyPhenotype, hudine_auc_diffPhysiologyPhenotype]
GeneNetRR_auprc_hudine = [hudine_auprc_samePhysiologyRef, hudine_auprc_diffPhysiologyRef, hudine_auprc_samePhysiologyPhenotype, hudine_auprc_diffPhysiologyPhenotype]


# ....... HSDN
ukb_auc_samePhysiologyPhenotype, ukb_auc_samePhysiologyRef, ukb_auc_diffPhysiologyPhenotype, ukb_auc_diffPhysiologyRef, \
ukb_auprc_samePhysiologyPhenotype, ukb_auprc_samePhysiologyRef, ukb_auprc_diffPhysiologyPhenotype, ukb_auprc_diffPhysiologyRef = HSDN('UKB')

hudine_auc_samePhysiologyPhenotype, hudine_auc_samePhysiologyRef, hudine_auc_diffPhysiologyPhenotype, hudine_auc_diffPhysiologyRef, \
hudine_auprc_samePhysiologyPhenotype, hudine_auprc_samePhysiologyRef, hudine_auprc_diffPhysiologyPhenotype, hudine_auprc_diffPhysiologyRef = HSDN('HuDiNe')


HSDN_auroc_ukb = [ukb_auc_samePhysiologyRef, ukb_auc_diffPhysiologyRef, ukb_auc_samePhysiologyPhenotype, ukb_auc_diffPhysiologyPhenotype]
HSDN_auprc_ukb = [ukb_auprc_samePhysiologyRef, ukb_auprc_diffPhysiologyRef, ukb_auprc_samePhysiologyPhenotype, ukb_auprc_diffPhysiologyPhenotype]
HSDN_auroc_hudine = [hudine_auc_samePhysiologyRef, hudine_auc_diffPhysiologyRef, hudine_auc_samePhysiologyPhenotype, hudine_auc_diffPhysiologyPhenotype]
HSDN_auprc_hudine = [hudine_auprc_samePhysiologyRef, hudine_auprc_diffPhysiologyRef, hudine_auprc_samePhysiologyPhenotype, hudine_auprc_diffPhysiologyPhenotype]


# ....... FusedSim
ukb_auc_samePhysiologyPhenotype, ukb_auc_samePhysiologyRef, ukb_auc_diffPhysiologyPhenotype, ukb_auc_diffPhysiologyRef, \
ukb_auprc_samePhysiologyPhenotype, ukb_auprc_samePhysiologyRef, ukb_auprc_diffPhysiologyPhenotype, ukb_auprc_diffPhysiologyRef = FusedSim('UKB')

hudine_auc_samePhysiologyPhenotype, hudine_auc_samePhysiologyRef, hudine_auc_diffPhysiologyPhenotype, hudine_auc_diffPhysiologyRef, \
hudine_auprc_samePhysiologyPhenotype, hudine_auprc_samePhysiologyRef, hudine_auprc_diffPhysiologyPhenotype, hudine_auprc_diffPhysiologyRef = FusedSim('HuDiNe')

FusedSim_auroc_ukb = [ukb_auc_samePhysiologyRef, ukb_auc_diffPhysiologyRef, ukb_auc_samePhysiologyPhenotype, ukb_auc_diffPhysiologyPhenotype]
FusedSim_auprc_ukb = [ukb_auprc_samePhysiologyRef, ukb_auprc_diffPhysiologyRef, ukb_auprc_samePhysiologyPhenotype, ukb_auprc_diffPhysiologyPhenotype]
FusedSim_auroc_hudine = [hudine_auc_samePhysiologyRef, hudine_auc_diffPhysiologyRef, hudine_auc_samePhysiologyPhenotype, hudine_auc_diffPhysiologyPhenotype]
FusedSim_auprc_hudine = [hudine_auprc_samePhysiologyRef, hudine_auprc_diffPhysiologyRef, hudine_auprc_samePhysiologyPhenotype, hudine_auprc_diffPhysiologyPhenotype]


ukb_data_ls = [Sab_auroc_ukb, GeneNetRR_auroc_ukb, HSDN_auroc_ukb, FusedSim_auroc_ukb,
               Sab_auprc_ukb, GeneNetRR_auprc_ukb, HSDN_auprc_ukb, FusedSim_auprc_ukb]
hudine_data_ls = [Sab_auroc_hudine, GeneNetRR_auroc_hudine, HSDN_auroc_hudine, FusedSim_auroc_hudine,
                   Sab_auprc_hudine, GeneNetRR_auprc_hudine, HSDN_auprc_hudine, FusedSim_auprc_hudine]
ref_order_ls = ['Sab', 'GeneNetRR', 'HSDN', 'FusedSim']

plot_new(ref_order_ls, ukb_data_ls, 'fig3_e.pdf')
plot_new(ref_order_ls, hudine_data_ls, 'fig3_f.pdf')



print('ok')