import pandas as pd
import numpy as np
import sys
sys.path.append('/home1/donggy/gene_network/code/graph_network/')
sys.path.append('graph_network')
from pygcn.utils import get_selected_phenotypes
from collections import defaultdict
from itertools import combinations
from scipy.stats import ttest_ind, mannwhitneyu
from statsmodels.stats.proportion import proportions_ztest
import argparse
from rpy2 import robjects
import re



phenotype_df = pd.read_csv('data/disease_phenotype_score_data_processed.csv', index_col=0)
phenotype_df.index = [each[:3] for each in phenotype_df.index]

multimorbidity_df = pd.read_csv('data/ukb_multimorbidity.csv')
multimorbidity_df = multimorbidity_df[multimorbidity_df['code1'].isin(phenotype_df.index) & multimorbidity_df['code2'].isin(phenotype_df.index)]
disease_intersection = set(multimorbidity_df['code1']) | set(multimorbidity_df['code2'])

disease_patients = dict()
with open('../../behavior_data/disease_patient_firstOccur.txt', 'r') as infile:
    for i, line in enumerate(infile):
        if i < 1:
            continue
        str1 = line.strip('\r\n').split('\t')
        code = str1[0][:3]
        if code not in disease_intersection:
            continue
        disease_patients[code] = set(str1[1].split(';'))
    infile.close()

multimorbid_diseasePairs = set()
for code1, code2 in multimorbidity_df.values.tolist():
    if code1 < code2:
        multimorbid_diseasePairs.add((code1, code2))
    else:
        multimorbid_diseasePairs.add((code2, code1))

all_diseasePairs = set()
for code1, code2 in combinations(disease_intersection, 2):
    set1 = disease_patients[code1]
    set2 = disease_patients[code2]
    if len(set1 & set2) <= 30:
        continue
    if code1 < code2:
        all_diseasePairs.add((code1, code2))
    else:
        all_diseasePairs.add((code2, code1))

df = pd.read_csv('UKB_predict_multimorbidity.csv')
df = df[(df['pred_label'] == 1) & (df['label'] == 0)][['code1', 'code2']]
novel_multimorbidity = set()
for code1, code2 in zip(df['code1'], df['code2']):
    if code1 < code2:
        novel_multimorbidity.add((code1, code2))
    else:
        novel_multimorbidity.add((code2, code1))

_, selected_phenotypes = get_selected_phenotypes(path='feature_selection_result/UKB/', lmd=0.0001, threshold=0.001)
continuous_phenotypes = set()
integer_phenotypes = set()
multiple_category_phenotypes = set()
single_category_ordered_phenotypes = set()
single_category_unordered_phenotypes = set()
single_category_binary_phenotypes = set()

for each in selected_phenotypes:
    each1 = each.replace('*pos', '').replace('*neg', '')
    if each1.split('~')[0] == 'continuous':
        continuous_phenotypes.add(each1)
    elif each1.split('~')[0] == 'integer':
        integer_phenotypes.add(each1)
    elif each1.split('~')[0] == 'multiple_category':
        multiple_category_phenotypes.add(each1)
    elif each1.split('~')[0] == 'single_category_ordered':
        single_category_ordered_phenotypes.add(each1)
    elif each1.split('~')[0] == 'single_category_unordered':
        single_category_unordered_phenotypes.add(each1)
    elif each1.split('~')[0] == 'single_category_binary':
        single_category_binary_phenotypes.add(each1)

continuous_phenotypes = list(continuous_phenotypes)
integer_phenotypes = list(integer_phenotypes)
multiple_category_phenotypes = list(multiple_category_phenotypes)
single_category_ordered_phenotypes = list(single_category_ordered_phenotypes)
single_category_unordered_phenotypes = list(single_category_unordered_phenotypes)
single_category_binary_phenotypes = list(single_category_binary_phenotypes)

continuous_phenotypes.sort()
integer_phenotypes.sort()
multiple_category_phenotypes.sort()
single_category_ordered_phenotypes.sort()
single_category_unordered_phenotypes.sort()
single_category_binary_phenotypes.sort()


# field dataCoding
field_dataCoding = dict()
df = pd.read_excel('data/selected_phenotype.xlsx', dtype=str)
for field, coding in df[['FieldID', 'Data_Coding']].values.tolist():
    if coding != '/':
        field_dataCoding[field] = coding


# dataCoding - missing
dataCoding_missing = defaultdict(set)
df = pd.read_excel('../../behavior_data/dataCoding_missing.xlsx', dtype=str)
for each in df.values.tolist():
    set1 = set()
    for each1 in each[4].strip('*').split('|'):
        set1.add(each1.split(':')[0])
    dataCoding_missing[each[1]] = set1


# dataCoding - validCodesOrder
dataCoding_validCodesOrder = dict()
df = df[(df['ValueType'] == 'Categorical (single)') & (df['CodesEncoding'] == 'Ordered')]
for each in df.values.tolist():
    list1 = []
    for each1 in each[6].strip('*').split('|'):
        list1.append(each1.split(':')[0])
    dataCoding_validCodesOrder[each[1]] = list1


# ....... for continuous and integer phenotypes
our_continuous_phenotypes = continuous_phenotypes + integer_phenotypes
def process_continuous_phenotypes(index):
    result = []
    for phenotype in our_continuous_phenotypes[index:index + 1]:
        field = phenotype.split('~')[-1].split('*')[0]
        missing_set = set()
        if field in field_dataCoding:
            dataCoding = field_dataCoding[field]
            if dataCoding in dataCoding_missing:
                missing_set = dataCoding_missing[dataCoding]

        df = pd.read_csv('../../behavior_data/all_split_field_20210331/' + field + '.csv', index_col=0, dtype=str)
        df.index = df.index.map(str)
        df = df[~df.isin(missing_set)]
        df = df.dropna(how='all', axis=0)
        df = df.astype(float)
        df = df.fillna(method='bfill', axis=1)
        S = df.iloc[:, 0]

        for i, (code1, code2) in enumerate(all_diseasePairs):
            disease_phenotype = phenotype_df[phenotype].to_dict()
            code1_coeff = disease_phenotype[code1]
            code2_coeff = disease_phenotype[code2]

            code1_patients = disease_patients[code1]
            code2_patients = disease_patients[code2]
            bothDis_patients = code1_patients & code2_patients

            S1 = S.loc[S.index.isin(code1_patients - bothDis_patients)]
            S2 = S.loc[S.index.isin(code2_patients - bothDis_patients)]
            S3 = S.loc[S.index.isin(bothDis_patients)]
            if (len(S1) <= 30) | (len(S2) <= 30) | (len(S3) <= 30):
                continue

            code1_pat_phe = np.mean(S1)
            code2_pat_phe = np.mean(S2)
            bothDis_pat_phe = np.mean(S3)

            t1, p1 = ttest_ind(S3, S1)
            t2, p2 = ttest_ind(S3, S2)
            t3, p3 = ttest_ind(S2, S1)
            t4, p4 = ttest_ind(S3, pd.concat([S1, S2], axis=0))

            if (code1, code2) in multimorbid_diseasePairs:
                result.append(
                    [code1, code2, 'multimorbidity', phenotype, code1_coeff, code2_coeff, code1_coeff * code2_coeff,
                     code1_pat_phe, code2_pat_phe, bothDis_pat_phe, t1, p1, t2, p2, t3, p3, t4, p4])
            elif (code1, code2) in novel_multimorbidity:
                result.append([code1, code2, 'multimorbidity_novel', phenotype, code1_coeff, code2_coeff,
                               code1_coeff * code2_coeff,
                               code1_pat_phe, code2_pat_phe, bothDis_pat_phe, t1, p1, t2, p2, t3, p3, t4, p4])
            else:
                result.append(
                    [code1, code2, 'nonmultimorbidity', phenotype, code1_coeff, code2_coeff, code1_coeff * code2_coeff,
                     code1_pat_phe, code2_pat_phe, bothDis_pat_phe, t1, p1, t2, p2, t3, p3, t4, p4])

        df = pd.DataFrame(result,
                          columns=['code1', 'code2', 'multimorbidity_flag', 'phenotype', 'code1_coeff', 'code2_coeff',
                                   'code1_coeff*code2_coeff', 'code1_patient_phenotype', 'code2_patient_phenotype',
                                   'bothDisease_patient_phenotype', 't1', 'p1', 't2', 'p2', 't3', 'p3', 't4', 'p4'])

        df.to_csv('patient_phenotype_test/continuous' + str(index) + '.csv', index=False)





# ....... for multiple category / single category binary / single category unordered phenotypes
our_binary_phenotypes = single_category_binary_phenotypes + multiple_category_phenotypes + single_category_unordered_phenotypes
def process_binary_phenotypes(index):
    result = []
    for phenotype in our_binary_phenotypes[index:index + 1]:
        field, value = phenotype.split('~')[-1].split('*')[0].split('_')
        missing_set = set()
        if field in field_dataCoding:
            dataCoding = field_dataCoding[field]
            if dataCoding in dataCoding_missing:
                missing_set = dataCoding_missing[dataCoding]

        df = pd.read_csv('../../behavior_data/all_split_field_20210331/' + field + '.csv', index_col=0, dtype=str)
        df.index = df.index.map(str)
        df = df[~df.isin(missing_set)]
        df = df.dropna(how='all', axis=0)
        participants = set(df.index)

        df1 = df[df == value]
        positive_participants = set(df1.dropna(how='all', axis=0).index)

        for i, (code1, code2) in enumerate(all_diseasePairs):
            disease_phenotype = phenotype_df[phenotype].to_dict()
            code1_coeff = disease_phenotype[code1]
            code2_coeff = disease_phenotype[code2]

            code1_patients = disease_patients[code1]
            code2_patients = disease_patients[code2]
            bothDis_patients = code1_patients & code2_patients

            code1_pat_part = (code1_patients - bothDis_patients) & participants
            code2_pat_part = (code2_patients - bothDis_patients) & participants
            bothDis_pat_part = bothDis_patients & participants

            if (len(code1_pat_part) <= 30) | (len(code2_pat_part) <= 30) | (len(bothDis_pat_part) <= 30):
                continue

            code1_pat_pos = (code1_patients - bothDis_patients) & positive_participants
            code2_pat_pos = (code2_patients - bothDis_patients) & positive_participants
            bothDise_pat_pos = bothDis_patients & positive_participants

            proportion1 = len(code1_pat_pos) / len(code1_pat_part)
            proportion2 = len(code2_pat_pos) / len(code2_pat_part)
            proportion3 = len(bothDise_pat_pos) / len(bothDis_pat_part)

            z1, p1 = proportions_ztest([len(bothDise_pat_pos), len(code1_pat_pos)],
                                       [len(bothDis_pat_part), len(code1_pat_part)])
            z2, p2 = proportions_ztest([len(bothDise_pat_pos), len(code2_pat_pos)],
                                       [len(bothDis_pat_part), len(code2_pat_part)])
            z3, p3 = proportions_ztest([len(code2_pat_pos), len(code1_pat_pos)],
                                       [len(code2_pat_part), len(code1_pat_part)])
            z4, p4 = proportions_ztest([len(bothDise_pat_pos), len(code1_pat_pos | code2_pat_pos)],
                                       [len(bothDis_pat_part), len(code1_pat_part | code2_pat_part)])

            if (code1, code2) in multimorbid_diseasePairs:
                result.append(
                    [code1, code2, 'multimorbidity', phenotype, code1_coeff, code2_coeff, code1_coeff * code2_coeff,
                     proportion1, proportion2, proportion3, z1, p1, z2, p2, z3, p3, z4, p4])
            elif (code1, code2) in novel_multimorbidity:
                result.append([code1, code2, 'multimorbidity_novel', phenotype, code1_coeff, code2_coeff,
                               code1_coeff * code2_coeff,
                               proportion1, proportion2, proportion3, z1, p1, z2, p2, z3, p3, z4, p4])
            else:
                result.append(
                    [code1, code2, 'nonmultimorbidity', phenotype, code1_coeff, code2_coeff, code1_coeff * code2_coeff,
                     proportion1, proportion2, proportion3, z1, p1, z2, p2, z3, p3, z4, p4])

        df = pd.DataFrame(result,
                          columns=['code1', 'code2', 'multimorbidity_flag', 'phenotype', 'code1_coeff', 'code2_coeff',
                                   'code1_coeff*code2_coeff', 'code1_patient_phenotype', 'code2_patient_phenotype',
                                   'bothDisease_patient_phenotype', 't1', 'p1', 't2', 'p2', 't3', 'p3', 't4', 'p4'])

        df.to_csv('patient_phenotype_test/binary' + str(index) + '.csv', index=False)




# ....... for single category ordered phenotypes
def process_ordered_phenotypes(index):
    result = []
    for phenotype in single_category_ordered_phenotypes[index:index + 1]:
        field = phenotype.split('~')[-1].split('*')[0]
        validCodesOrder = []
        if field in field_dataCoding:
            dataCoding = field_dataCoding[field]
            if dataCoding in dataCoding_validCodesOrder:
                validCodesOrder = dataCoding_validCodesOrder[dataCoding]
        if len(validCodesOrder) == 0:
            raise Exception('no valid codes')

        df = pd.read_csv('../../behavior_data/all_split_field_20210331/' + field + '.csv', index_col=0, dtype=str)
        df.index = df.index.map(str)
        validCodes_dict = {v: i for i, v in enumerate(validCodesOrder)}
        df = df[df.isin(validCodesOrder)]
        df = df.dropna(how='all', axis=0)
        df = df.replace(validCodes_dict)
        df = df.fillna(method='bfill', axis=1)
        S = df.iloc[:, 0]

        for i, (code1, code2) in enumerate(all_diseasePairs):
            disease_phenotype = phenotype_df[phenotype].to_dict()
            code1_coeff = disease_phenotype[code1]
            code2_coeff = disease_phenotype[code2]

            code1_patients = disease_patients[code1]
            code2_patients = disease_patients[code2]
            bothDis_patients = code1_patients & code2_patients

            S1 = S[S.index.isin(code1_patients - bothDis_patients)]
            S2 = S[S.index.isin(code2_patients - bothDis_patients)]
            S3 = S[S.index.isin(bothDis_patients)]
            S4 = S[S.index.isin((code1_patients | code2_patients) - bothDis_patients)]

            if (len(S1) <= 30) | (len(S2) <= 30) | (len(S3) <= 30):
                continue

            if len(S) >= 5:
                u1, p1 = mannwhitneyu(S3, S1)
                u2, p2 = mannwhitneyu(S3, S2)
                u3, p3 = mannwhitneyu(S2, S1)
                u4, p4 = mannwhitneyu(S3, S4)
            else:
                list1 = []
                list2 = []
                list3 = []
                list4 = []
                for v in validCodesOrder:
                    list1.append(len(S1[S1 == v]))
                    list2.append(len(S2[S2 == v]))
                    list3.append(len(S3[S3 == v]))
                    list4.append(len(S4[S4 == v]))

                temp1 = pd.DataFrame([list3, list1])
                temp2 = pd.DataFrame([list3, list2])
                temp3 = pd.DataFrame([list2, list1])
                temp4 = pd.DataFrame([list3, list4])

                temp1.to_csv('a1.csv', index=False, header=False)
                temp2.to_csv('a2.csv', index=False, header=False)
                temp3.to_csv('a3.csv', index=False, header=False)
                temp4.to_csv('a4.csv', index=False, header=False)

                robjects.r('library(DescTools)')
                robjects.r('a1=read.csv(file="a1.csv", header=F)')
                robjects.r('a1=data.matrix(a1)')
                x = robjects.r('MHChisqTest(a1)')
                x = str(x)
                u1 = float(re.findall(r'X-squared = (.*?),', x)[0])
                p1 = float(x.split('p-value = ')[-1].strip())

                robjects.r('a2=read.csv(file="a2.csv", header=F)')
                robjects.r('a2=data.matrix(a2)')
                x = robjects.r('MHChisqTest(a2)')
                x = str(x)
                u2 = float(re.findall(r'X-squared = (.*?),', x)[0])
                p2 = float(x.split('p-value = ')[-1].strip())

                robjects.r('a3=read.csv(file="a3.csv", header=F)')
                robjects.r('a3=data.matrix(a3)')
                x = robjects.r('MHChisqTest(a3)')
                x = str(x)
                u3 = float(re.findall(r'X-squared = (.*?),', x)[0])
                p3 = float(x.split('p-value = ')[-1].strip())

                robjects.r('a4=read.csv(file="a4.csv", header=F)')
                robjects.r('a4=data.matrix(a4)')
                x = robjects.r('MHChisqTest(a4)')
                x = str(x)
                u4 = float(re.findall(r'X-squared = (.*?),', x)[0])
                p4 = float(x.split('p-value = ')[-1].strip())

            if (code1, code2) in multimorbid_diseasePairs:
                result.append(
                    [code1, code2, 'multimorbidity', phenotype, code1_coeff, code2_coeff, code1_coeff * code2_coeff,
                     np.mean(S1), np.mean(S2), np.mean(S3), u1, p1, u2, p2, u3, p3, u4, p4])
            elif (code1, code2) in novel_multimorbidity:
                result.append([code1, code2, 'multimorbidity_novel', phenotype, code1_coeff, code2_coeff,
                               code1_coeff * code2_coeff,
                               np.mean(S1), np.mean(S2), np.mean(S3), u1, p1, u2, p2, u3, p3, u4, p4])
            else:
                result.append(
                    [code1, code2, 'nonmultimorbidity', phenotype, code1_coeff, code2_coeff, code1_coeff * code2_coeff,
                     np.mean(S1), np.mean(S2), np.mean(S3), u1, p1, u2, p2, u3, p3, u4, p4])

        df = pd.DataFrame(result,
                          columns=['code1', 'code2', 'multimorbidity_flag', 'phenotype', 'code1_coeff', 'code2_coeff',
                                   'code1_coeff*code2_coeff', 'code1_patient_phenotype', 'code2_patient_phenotype',
                                   'bothDisease_patient_phenotype', 't1', 'p1', 't2', 'p2', 't3', 'p3', 't4', 'p4'])

        df.to_csv('patient_phenotype_test/single_category_ordered' + str(index) + '.csv', index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='input index')
    parser.add_argument('--index', type=int, default=None)
    parser.add_argument('--type', type=str, default=None)
    args = parser.parse_args()

    if args.type == 'continuous':
        process_continuous_phenotypes(args.index)
    if args.type == 'binary':
        process_binary_phenotypes(args.index)
    if args.type == 'ordered':
        process_ordered_phenotypes(args.index)



print('ok')