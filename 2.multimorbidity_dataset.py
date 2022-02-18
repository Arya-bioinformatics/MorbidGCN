import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats import multitest
from itertools import combinations, product, islice
from collections import defaultdict




def RR(Ii, Ij, Cij, N):
    temp1 = Cij * N
    temp2 = Ii * Ij
    return float(temp1)/float(temp2)



def significance(Ii, Ij, Cij, N):
    CijStar = float(Ii * Ij)/float(N)
    p = stats.poisson.pmf(np.arange(0, N), CijStar)
    p_value = 1 - sum(p[:Cij])
    if p_value < 0:
        p_value = 0
    return p_value



def get_ukb_multimorbidity(outpath='data/ukb_multimorbidity.csv'):
    disease_patient = dict()
    infile = open('../../behavior_data/disease_patient_firstOccur.txt', 'r')
    for line in islice(infile, 1, None):
        str1 = line.strip('\r\n')
        str2 = str1.split('\t')
        disease_patient[str2[0]] = set(str2[1].split(';'))
    infile.close()

    N = 502486
    disease_pair = list(combinations(disease_patient.keys(), 2))

    result = list()
    for disease1, disease2 in disease_pair:
        set1 = disease_patient[disease1]
        set2 = disease_patient[disease2]
        Ii = len(set1)
        Ij = len(set2)
        Cij = len(set1 & set2)
        rr = RR(Ii, Ij, Cij, N)
        p = significance(Ii, Ij, Cij, N)

        if disease1 < disease2:
            result.append([disease1[:3], disease2[:3], disease1, disease2, str(Ii), str(Ij), str(Cij), str(rr), str(p)])
        else:
            result.append([disease2[:3], disease1[:3], disease2, disease1, str(Ij), str(Ii), str(Cij), str(rr), str(p)])

    df = pd.DataFrame(result, columns=['code1', 'code2', 'Description1', 'Description2', 'Ii', 'Ij', 'Cij', 'RR',
                             'P_value'])

    # ---------------------- multimorbidity filter
    df = df[df['P_value'] < (0.05/df.shape[0])]
    df = df[df['RR'] > 1]
    df.to_csv(outpath, index=False)




def HuDiNe_pVal(inpath):
    # --------------- calculate hidalgo multimorbidity p values
    df = pd.read_table(inpath, dtype=str)

    N = 13039018
    nn = np.arange(0, N)

    list1 = []
    for each in df.values.tolist():
        code1, code2, n1, n2, n12, RR, RR_low, RR_high, pval, pval_adj = each
        my_p = significance(int(n1), int(n2), int(n12), N, nn=nn)
        list1.append([code1, code2, n1, n2, n12, RR, RR_low, RR_high, pval, pval_adj, my_p])

    df = pd.DataFrame(list1,
                      columns=['code1', 'code2', 'n1', 'n2', 'n12', 'RR', 'RR_low', 'RR_high', 'pval', 'pval_adj', 'my_p'])
    return df








if __name__ == '__main__':

    # ukb multimorbidity
    get_ukb_multimorbidity(outpath='data/ukb_multimorbidity.txt')


    # hidalgo multimorbidity
    df_3digit = HuDiNe_pVal('../source_download/multimorbidity/hidalgo/AllNet3.tsv')
    s = df_3digit['my_p'].astype(float)
    fdr_3digit = multitest.fdrcorrection(s)[1]

    df_5digit = HuDiNe_pVal('../source_download/multimorbidity/hidalgo/AllNet5.tsv')
    s = df_5digit['my_p'].astype(float)
    fdr_5digit = multitest.fdrcorrection(s)[1]

    icd9_icd10_map = defaultdict(set)
    with open('/home1/donggy/UMLS/result/icd9cm_icd10_map.txt', 'r') as infile:
        for i, line in enumerate(infile):
            if i < 1:
                continue
            str1 = line.strip('\r\n').split('\t')
            set1 = set(str1[1].split('|'))
            set2 = set([each for each in str1[4].split('|') if '-' not in each])
            for each in set1:
                if len(each.split('.')[0]) != 3:
                    print(each)
                    raise Exception
                icd9_icd10_map[each] = set2
        infile.close()

    hidalgo_multimorbidity_set = set()

    for i, each in enumerate(df_3digit.values.tolist()):
        if each[5] <= 1:
            continue
        padj = fdr_3digit[i]
        if padj >= 0.05:
            continue
        code1, code2 = '000' + str(each[0]), '000' + str(each[1])
        code1, code2 = code1[-3:], code2[-3:]
        if (code1 in icd9_icd10_map) & (code2 in icd9_icd10_map):
            set1 = icd9_icd10_map[code1]
            set2 = icd9_icd10_map[code2]
            set1 = set1 - (set1 & set2)
            set2 = set2 - (set1 & set2)
            for code1, code2 in product(set1, set2):
                if code1 < code2:
                    hidalgo_multimorbidity_set.add((code1, code2))
                else:
                    hidalgo_multimorbidity_set.add((code2, code1))

    for i, each in enumerate(df_5digit.values.tolist()):
        if each[5] <= 1:
            continue
        padj = fdr_5digit[i]
        if padj >= 0.05:
            continue
        code1, code2 = '000' + str1[0], '000' + str1[1]
        if '.' in code1:
            code1 = code1.split('.')[0][-3:] + '.' + code1.split('.')[1]
        else:
            code1 = code1[-3:]
        if '.' in code2:
            code2 = code2.split('.')[0][-3:] + '.' + code2.split('.')[1]
        else:
            code2 = code2[-3:]

        if (code1 in icd9_icd10_map) & (code2 in icd9_icd10_map):
            set1 = icd9_icd10_map[code1]
            set2 = icd9_icd10_map[code2]
            set1 = set1 - (set1 & set2)
            set2 = set2 - (set1 & set2)
            for code1, code2 in product(set1, set2):
                if code1 < code2:
                    hidalgo_multimorbidity_set.add((code1, code2))
                else:
                    hidalgo_multimorbidity_set.add((code2, code1))

    hidalgo_multimorbidity_df = pd.DataFrame(hidalgo_multimorbidity_set, columns=['code1', 'code2'])
    hidalgo_multimorbidity_df.to_csv('data/hudine_multimorbidity.csv', index=False)


    print('ok')