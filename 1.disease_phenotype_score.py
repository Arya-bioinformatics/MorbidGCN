import pandas as pd
import numpy as np
from scipy import sparse
from scipy.stats import kendalltau, pointbiserialr
from collections import defaultdict
from itertools import islice, product
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer




patient_diseases = defaultdict(set)
disease_patients = defaultdict(set)
all_patient = set()
infile = open('../../behavior_data/disease_patient_firstOccur.txt', 'r')
for line in islice(infile, 1, None):
    str1 = line.strip('\r\n')
    str2 = str1.split('\t')
    set1 = set(str2[1].split(';'))
    disease_patients[str2[0]] = set1
    all_patient |= set1
    for each in set1:
        patient_diseases[each].add(str2[0])
infile.close()




def quantify_continuous_phenotypes(outpath='data/disease_phenotype_score_continuous.txt'):
    # field description and dataCoding
    field_description = dict()
    field_dataCoding = dict()
    df = pd.read_excel('data/selected_phenotype.xlsx', dtype=str)
    df = df[df['ValueType'] == 'Continuous']
    for each in df[['Group', 'Path', 'FieldID', 'Field', 'Data_Coding']].values.tolist():
        field_description[each[2]] = each[0] + '~' + each[1] + '~' + each[2] + '*' + each[3]
        field_dataCoding[each[2]] = each[4]

    # dataCoding - missing
    dataCoding_missing = defaultdict(set)
    df = pd.read_excel('../../behavior_data/dataCoding_missing.xlsx', dtype=str)
    for each in df.values.tolist():
        set1 = set()
        for each1 in each[4].strip('*').split('|'):
            set1.add(each1.split(':')[0])
        dataCoding_missing[each[1]] = set1

    result = []
    outfile = open(outpath, 'w+')
    outfile.write('field\tfield_description\tdisease\tcoef\tp\n')
    outfile.close()

    scaler = StandardScaler()
    for each in field_description:
        description = field_description[each]

        df = pd.read_csv('../../behavior_data/all_split_field_20210331/' + each + '.csv', index_col=0, dtype=str)
        df.index = df.index.map(str)

        dataCoding = field_dataCoding[each]
        missing_set = dataCoding_missing[dataCoding]
        df = df[~df.isin(missing_set)]
        df = df.dropna(how='all', axis=0)
        df = df.astype(float)
        df = df.fillna(method='bfill', axis=1)
        S = df.iloc[:, 0]

        people_order = S.index
        phenotype_ordered_array = np.array(S)

        list1 = []
        for each1 in people_order:
            set1 = patient_diseases[each1]
            list1 += [i for i in product([each1], set1)]

        df = pd.DataFrame(list1, columns=['patient', 'disease'])

        people_index = {v: i for i, v in enumerate(people_order)}
        all_disease = set(df['disease'])
        disease_order = list(all_disease)
        disease_order.sort()
        disease_index = {v: i for i, v in enumerate(disease_order)}
        df['patient'] = df['patient'].map(people_index)
        df['disease'] = df['disease'].map(disease_index)
        data = sparse.coo_matrix(([1] * df.shape[0], (df['patient'], df['disease'])),
                                 shape=(len(people_index), len(disease_index)), dtype=float).toarray()
        df = pd.DataFrame(data, columns=disease_order, index=people_order)
        df = df[df.columns[df.sum() > 10]]
        if df.shape[1] == 0:
            continue

        X = phenotype_ordered_array.reshape(-1, 1)
        X = scaler.fit_transform(X).flatten()
        for disease in df.columns:
            y = np.array(df[disease])
            [r, p] = pointbiserialr(y, X)
            result.append([each, description, disease, str(r), str(p)])

        if len(result) > 10000:
            outfile = open(outpath, 'a')
            for each1 in result:
                outfile.write('\t'.join(each1) + '\n')
            outfile.close()
            result = []

    outfile = open(outpath, 'a')
    for each1 in result:
        outfile.write('\t'.join(each1) + '\n')
    outfile.close()





def quantify_integer_phenotypes(outpath='data/disease_phenotype_score_integer.txt'):
    # field description and dataCoding
    field_description = dict()
    field_dataCoding = dict()
    df = pd.read_excel('data/selected_phenotype.xlsx', dtype=str)
    df = df[df['ValueType'] == 'Integer']
    for each in df[['Group', 'Path', 'FieldID', 'Field', 'Data_Coding']].values.tolist():
        field_description[each[2]] = each[0] + '~' + each[1] + '~' + each[2] + '*' + each[3]
        field_dataCoding[each[2]] = each[4]

    # dataCoding - missing
    dataCoding_missing = defaultdict(set)
    df = pd.read_excel('../../behavior_data/dataCoding_missing.xlsx', dtype=str)
    for each in df.values.tolist():
        set1 = set()
        for each1 in each[4].strip('*').split('|'):
            set1.add(each1.split(':')[0])
        dataCoding_missing[each[1]] = set1

    result = []
    outfile = open(outpath, 'w+')
    outfile.write('field\tfield_description\tdisease\tcoef\tp\n')
    outfile.close()

    scaler = StandardScaler()
    for each in field_description:
        description = field_description[each]

        df = pd.read_csv('../../behavior_data/all_split_field_20210331/' + each + '.csv', index_col=0, dtype=str)
        df.index = df.index.map(str)

        dataCoding = field_dataCoding[each]
        missing_set = dataCoding_missing[dataCoding]
        df = df[~df.isin(missing_set)]
        df = df.dropna(how='all', axis=0)
        df = df.fillna(method='bfill', axis=1)
        S = df.iloc[:, 0]

        people_order = S.index
        phenotype_ordered_array = np.array(S)

        list1 = []
        for each1 in people_order:
            set1 = patient_diseases[each1]
            list1 += [i for i in product([each1], set1)]

        df = pd.DataFrame(list1, columns=['patient', 'disease'])

        people_index = {v: i for i, v in enumerate(people_order)}
        all_disease = set(df['disease'])
        disease_order = list(all_disease)
        disease_order.sort()
        disease_index = {v: i for i, v in enumerate(disease_order)}
        df['patient'] = df['patient'].map(people_index)
        df['disease'] = df['disease'].map(disease_index)
        data = sparse.coo_matrix(([1] * df.shape[0], (df['patient'], df['disease'])),
                                 shape=(len(people_index), len(disease_index)), dtype=float).toarray()
        df = pd.DataFrame(data, columns=disease_order, index=people_order)
        df = df[df.columns[df.sum() > 10]]
        if df.shape[1] == 0:
            continue

        X = phenotype_ordered_array.reshape(-1, 1)
        X = scaler.fit_transform(X).flatten()
        for disease in df.columns:
            y = np.array(df[disease])
            [r, p] = pointbiserialr(y, X)
            result.append([each, description, disease, str(r), str(p)])

        if len(result) > 10000:
            outfile = open(outpath, 'a')
            for each1 in result:
                outfile.write('\t'.join(each1) + '\n')
            outfile.close()
            result = []

    outfile = open(outpath, 'a')
    for each1 in result:
        outfile.write('\t'.join(each1) + '\n')
    outfile.close()





def quantify_multiple_category_phenotypes(outpath='data/disease_phenotype_score_multiple_category.txt'):
    # field description and dataCoding
    field_description = dict()
    field_dataCoding = dict()
    df = pd.read_excel('data/selected_phenotype.xlsx', dtype=str)
    df = df[df['ValueType'] == 'Categorical (multiple)']
    for each in df[['Group', 'Path', 'FieldID', 'Field', 'Data_Coding']].values.tolist():
        field_description[each[2]] = each[0] + '~' + each[1] + '~' + each[2] + '*' + each[3]
        field_dataCoding[each[2]] = each[4]

    # dataCoding - missing
    dataCoding_missing = defaultdict(set)
    df = pd.read_excel('../../behavior_data/dataCoding_missing.xlsx', dtype=str)
    for each in df.values.tolist():
        set1 = set()
        for each1 in each[4].strip('*').split('|'):
            set1.add(each1.split(':')[0])
        dataCoding_missing[each[1]] = set1

    outfile = open(outpath, 'w+')
    outfile.write('field\tfield_description\tdisease\tscore\n')
    outfile.close()

    result = []
    for each in field_description:
        description = field_description[each]

        df = pd.read_csv('../../behavior_data/all_split_field_20210331/' + each + '.csv', index_col=0, dtype=str)
        df.index = df.index.map(str)
        dataCoding = field_dataCoding[each]
        missing_set = dataCoding_missing[dataCoding]
        df = df[~df.isin(missing_set)]
        df = df.dropna(how='all', axis=0)

        participants = set(df.index)
        Pi = len(participants)

        values = set()
        for col in df.columns:
            values |= set(df[col].dropna().unique())

        for v in values:
            df1 = df[df == v]
            positive_participant = set(df1.dropna(how='all', axis=0).index)
            Pi_1 = len(positive_participant)
            if (Pi_1 / Pi < 0.01) | (Pi_1 / Pi > 0.99):
                continue
            description1 = ('_' + v + '*').join(description.split('*'))
            for disease in disease_patients:
                patients = disease_patients[disease]
                if len(patients & participants) <= 10:
                    continue
                n11 = len(patients & positive_participant)
                n10 = len(patients & (participants - positive_participant))
                n01 = len(positive_participant) - n11
                n00 = len((participants - positive_participant) - patients)
                phi = (n11 * n00 - n10 * n01) / (np.power((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00), 0.5))
                result.append([each, description1, disease, str(phi)])

        if len(result) > 10000:
            outfile = open(outpath, 'a')
            for each1 in result:
                outfile.write('\t'.join(each1) + '\n')
            outfile.close()
            result = []

    outfile = open(outpath, 'a')
    for each1 in result:
        outfile.write('\t'.join(each1) + '\n')
    outfile.close()




def quantify_single_category_binary_phenotypes(outpath='data/disease_phenotype_score_single_category_binary.txt'):
    # field description and dataCoding
    field_description = dict()
    field_dataCoding = dict()
    df = pd.read_excel('data/selected_phenotype.xlsx', dtype=str)
    df = df[df['ValueType'] == 'Categorical (single)']
    for each in df[['Group', 'Path', 'FieldID', 'Field', 'Data_Coding']].values.tolist():
        field_description[each[2]] = each[0] + '~' + each[1] + '~' + each[2] + '*' + each[3]
        field_dataCoding[each[2]] = each[4]

    # dataCoding - missing
    dataCoding_missing = dict()
    df = pd.read_excel('../../behavior_data/dataCoding_missing.xlsx', dtype=str)
    df = df[(df['ValueType'] == 'Categorical (single)') & (df['CodesEncoding'] == 'Binary')]
    for each in df.values.tolist():
        set1 = set()
        for each1 in each[4].strip('*').split('|'):
            set1.add(each1.split(':')[0])
        dataCoding_missing[each[1]] = set1

    outfile = open(outpath, 'w+')
    outfile.write('field\tfield_description\tdisease\tscore\n')
    outfile.close()

    result = []
    for each in field_description:
        dataCoding = field_dataCoding[each]
        if dataCoding not in dataCoding_missing:
            continue

        description = field_description[each]

        df = pd.read_csv('../../behavior_data/all_split_field_20210331/' + each + '.csv', index_col=0, dtype=str)
        df.index = df.index.map(str)
        missing_set = dataCoding_missing[dataCoding]
        df = df[~df.isin(missing_set)]
        df = df.dropna(how='all', axis=0)
        df = df.fillna(method='bfill', axis=1)
        S = df.iloc[:, 0]
        S1 = S.value_counts()

        participants = set(S.index)
        Pi = len(participants)

        S2 = S[S == S1.index[0]]
        positive_participant = set(S2.dropna().index)
        Pi_1 = len(positive_participant)
        if (Pi_1 / Pi < 0.01) | (Pi_1 / Pi > 0.99):
            continue
        description1 = ('_' + S1.index[0] + '*').join(description.split('*'))
        for disease in disease_patients:
            patients = disease_patients[disease]
            if len(patients & participants) <= 10:
                continue
            n11 = len(patients & positive_participant)
            n10 = len(patients & (participants - positive_participant))
            n01 = len(positive_participant) - n11
            n00 = len((participants - positive_participant) - patients)
            phi = (n11 * n00 - n10 * n01) / (np.power((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00), 0.5))
            result.append([each, description1, disease, str(phi)])

        if len(result) > 10000:
            outfile = open(outpath, 'a')
            for each1 in result:
                outfile.write('\t'.join(each1) + '\n')
            outfile.close()
            result = []

    outfile = open(outpath, 'a')
    for each1 in result:
        outfile.write('\t'.join(each1) + '\n')
    outfile.close()




def quantify_single_category_ordered_phenotypes(outpath='data/disease_phenotype_score_single_category_ordered.txt'):
    # field description and dataCoding
    field_description = dict()
    field_dataCoding = dict()
    df = pd.read_excel('data/selected_phenotype.xlsx', dtype=str)
    df = df[df['ValueType'] == 'Categorical (single)']
    for each in df[['Group', 'Path', 'FieldID', 'Field', 'Data_Coding']].values.tolist():
        field_description[each[2]] = each[0] + '~' + each[1] + '~' + each[2] + '*' + each[3]
        field_dataCoding[each[2]] = each[4]

    # dataCoding - validCodesOrder
    dataCoding_validCodesOrder = dict()
    df = pd.read_excel('../../behavior_data/dataCoding_missing.xlsx', dtype=str)
    df = df[(df['ValueType'] == 'Categorical (single)') & (df['CodesEncoding'] == 'Ordered')]
    for each in df.values.tolist():
        list1 = []
        for each1 in each[6].strip('*').split('|'):
            list1.append(each1.split(':')[0])
        dataCoding_validCodesOrder[each[1]] = list1

    outfile = open(outpath, 'w+')
    outfile.write('field\tfield_description\tdisease\tcorrelation\tp\n')
    outfile.close()

    j = 0
    result = []
    for each in field_description:
        dataCoding = field_dataCoding[each]
        if dataCoding not in dataCoding_validCodesOrder:
            continue
        validCodesOrder = dataCoding_validCodesOrder[dataCoding]
        if len(validCodesOrder) == 0:
            raise Exception('no valid codes')

        df = pd.read_csv('../../behavior_data/all_split_field_20210331/' + each + '.csv', index_col=0, dtype=str)
        df.index = df.index.map(str)
        validCodes_dict = {v: i for i, v in enumerate(validCodesOrder)}
        df = df[df.isin(validCodesOrder)]
        df = df.dropna(how='all', axis=0)
        df = df.replace(validCodes_dict)
        df = df.fillna(method='bfill', axis=1)
        S = df.iloc[:, 0]
        S1 = S.value_counts()
        if S1.iloc[1] < S1.iloc[0] * 0.05:
            continue

        description = field_description[each]
        print('single_category_ordered', j, description)
        j += 1

        for disease in disease_patients:
            patients = disease_patients[disease]
            labels = [1. if people in patients else 0. for people in S.index]
            if sum(labels) <= 10.:
                continue
            tau, p_value = kendalltau(S, labels)
            result.append([each, description, disease, str(tau), str(p_value)])

        if len(result) > 10000:
            outfile = open(outpath, 'a')
            for each1 in result:
                outfile.write('\t'.join(each1) + '\n')
            outfile.close()
            result = []

    outfile = open(outpath, 'a')
    for each1 in result:
        outfile.write('\t'.join(each1) + '\n')
    outfile.close()






def quantify_single_category_unordered_phenotypes(outpath='data/disease_phenotype_score_single_category_unordered.txt'):
    # field description and dataCoding
    field_description = dict()
    field_dataCoding = dict()
    df = pd.read_excel('data/selected_phenotype.xlsx', dtype=str)
    df = df[df['ValueType'] == 'Categorical (single)']
    for each in df[['Group', 'Path', 'FieldID', 'Field', 'Data_Coding']].values.tolist():
        field_description[each[2]] = each[0] + '~' + each[1] + '~' + each[2] + '*' + each[3]
        field_dataCoding[each[2]] = each[4]

    # dataCoding - missing
    dataCoding_missing = dict()
    df = pd.read_excel('../../behavior_data/dataCoding_missing.xlsx', dtype=str)
    df = df[(df['ValueType'] == 'Categorical (single)') & (df['CodesEncoding'] == 'Unordered')]
    for each in df.values.tolist():
        set1 = set()
        for each1 in each[4].strip('*').split('|'):
            set1.add(each1.split(':')[0])
        dataCoding_missing[each[1]] = set1

    outfile = open(outpath, 'w+')
    outfile.write('field\tfield_description\tdisease\tscore\n')
    outfile.close()

    result = []
    for each in field_description:
        dataCoding = field_dataCoding[each]
        if dataCoding not in dataCoding_missing:
            continue

        description = field_description[each]

        df = pd.read_csv('../../behavior_data/all_split_field_20210331/' + each + '.csv', index_col=0, dtype=str)
        df.index = df.index.map(str)
        missing_set = dataCoding_missing[dataCoding]
        df = df[~df.isin(missing_set)]
        df = df.dropna(how='all', axis=0)
        df = df.fillna(method='bfill', axis=1)
        S = df.iloc[:, 0]
        S1 = S.value_counts()

        participants = set(S.index)
        Pi = len(participants)

        for v in S1.index:
            S2 = S[S == v]
            positive_participant = set(S2.dropna().index)
            Pi_1 = len(positive_participant)
            if (Pi_1 / Pi < 0.01) | (Pi_1 / Pi > 0.99):
                continue
            description1 = ('_' + v + '*').join(description.split('*'))
            for disease in disease_patients:
                patients = disease_patients[disease]
                if len(patients & participants) <= 10:
                    continue
                n11 = len(patients & positive_participant)
                n10 = len(patients & (participants - positive_participant))
                n01 = len(positive_participant) - n11
                n00 = len((participants - positive_participant) - patients)
                phi = (n11 * n00 - n10 * n01) / (np.power((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00), 0.5))
                result.append([each, description1, disease, str(phi)])

        if len(result) > 10000:
            outfile = open(outpath, 'a')
            for each1 in result:
                outfile.write('\t'.join(each1) + '\n')
            outfile.close()
            result = []

    outfile = open(outpath, 'a')
    for each1 in result:
        outfile.write('\t'.join(each1) + '\n')
    outfile.close()




def merge_phenotype(outpath='data/disease_phenotype_score_data.csv'):
    disease_phenotype_score = dict()

    # continuous
    infile = open('data/disease_phenotype_score_continuous.txt', 'r')
    for line in islice(infile, 1, None):
        str1 = line.strip('\r\n')
        str2 = str1.split('\t')
        phenotype, disease, score = str2[1:4]
        disease_phenotype_score[('continuous~' + phenotype, disease)] = score
    infile.close()

    # integer
    infile = open('data/disease_phenotype_score_integer.txt', 'r')
    for line in islice(infile, 1, None):
        str1 = line.strip('\r\n')
        str2 = str1.split('\t')
        phenotype, disease, score = str2[1:4]
        disease_phenotype_score[('integer~' + phenotype, disease)] = score
    infile.close()

    # multiple category
    infile = open('data/disease_phenotype_score_multiple_category.txt', 'r')
    for line in islice(infile, 1, None):
        str1 = line.strip('\r\n')
        str2 = str1.split('\t')
        phenotype, disease, score = str2[1:4]
        disease_phenotype_score[('multiple_category~' + phenotype, disease)] = score
    infile.close()

    # single category (binary)
    infile = open('data/disease_phenotype_score_single_category_binary.txt', 'r')
    for line in islice(infile, 1, None):
        str1 = line.strip('\r\n')
        str2 = str1.split('\t')
        phenotype, disease, score = str2[1:4]
        disease_phenotype_score[('single_category_binary~' + phenotype, disease)] = score
    infile.close()

    # single category (ordered)
    infile = open('data/disease_phenotype_score_single_category_ordered.txt', 'r')
    for line in islice(infile, 1, None):
        str1 = line.strip('\r\n')
        str2 = str1.split('\t')
        phenotype, disease, score = str2[1:4]
        disease_phenotype_score[('single_category_ordered~' + phenotype, disease)] = score
    infile.close()

    # single category (unordered)
    infile = open('data/disease_phenotype_score_single_category_unordered.txt', 'r')
    for line in islice(infile, 1, None):
        str1 = line.strip('\r\n')
        str2 = str1.split('\t')
        phenotype, disease, score = str2[1:4]
        disease_phenotype_score[('single_category_unordered~' + phenotype, disease)] = score
    infile.close()

    phenotype_tuple, disease_tuple = zip(*disease_phenotype_score.keys())
    all_phenotype = set(phenotype_tuple)
    all_disease = set(disease_tuple)
    phenotype_order = list(all_phenotype)
    phenotype_order.sort()
    disease_order = list(all_disease)
    disease_order.sort()
    disease_index = {v: i for i, v in enumerate(disease_order)}
    phenotype_index = {v: i for i, v in enumerate(phenotype_order)}

    data = np.empty((len(disease_order), len(phenotype_order)))
    data[:] = np.nan
    for each in disease_phenotype_score:
        phenotype, disease = each
        score = disease_phenotype_score[each]
        if score == 'nan':
            print(each, 'nan')
            continue
        i = disease_index[disease]
        j = phenotype_index[phenotype]
        data[i, j] = float(score)

    df = pd.DataFrame(data, columns=phenotype_order, index=disease_order)
    df.to_csv(outpath)



def missing_imputation(outpath='data/disease_phenotype_score_data_processed.csv'):
    df = pd.read_csv('data/disease_phenotype_score_data.csv', index_col=0)
    # remove diseases and phenotypes with missing rate >= 10%
    df = df.T
    df = df.loc[:, df.isnull().sum() / df.shape[0] < 0.1]
    df = df.T
    df = df.loc[:, df.isnull().sum() / df.shape[0] < 0.1]
    print('shape after removing missing rate >= 10% diseases and phenotype', df.shape)

    # imputation by knn, for each phenotype, using the neighbors calculated on the fields belonging to the same group
    col_group = []
    for col in df.columns:
        group = col.split('~')[1]
        if group not in col_group:
            col_group.append(group)

    imputer = KNNImputer()
    imputed_df = pd.DataFrame()
    for each_group in col_group:
        df1 = df[[col for col in df.columns if ('~' + each_group + '~') in col]]
        X = imputer.fit_transform(df1)
        X = pd.DataFrame(X, columns=df1.columns, index=df1.index)
        imputed_df = pd.concat([imputed_df, X], axis=1)

    imputed_df.to_csv(outpath)
    print(imputed_df.shape)








if __name__ == '__main__':

    quantify_continuous_phenotypes()
    quantify_integer_phenotypes()
    quantify_multiple_category_phenotypes()
    quantify_single_category_binary_phenotypes()
    quantify_single_category_ordered_phenotypes()
    quantify_single_category_unordered_phenotypes()
    merge_phenotype()
    missing_imputation()



    print('ok')