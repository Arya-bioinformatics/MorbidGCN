import pandas as pd
import os
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
from matplotlib import pyplot as plt
plt.rc('font', family='Helvetica')
import re
from lifelines import CoxPHFitter





# ......................... survival analysis
# ....... sex, age
sample_sex = pd.read_csv('../../behavior_data/all_split_field_20210331/31.csv', index_col=0)
sample_sex.index = sample_sex.index.map(str)
sample_sex = sample_sex.iloc[:, 0]
sample_age = pd.read_csv('../../behavior_data/all_split_field_20210331/21022.csv', index_col=0)
sample_age.index = sample_age.index.map(str)
sample_age = sample_age.iloc[:, 0]


# blood sample collected date
df = pd.read_csv('../../behavior_data/all_split_field_20210331/3166.csv', index_col=0)
df.index = df.index.map(str)
df = df.dropna(axis=0, how='all')
df = df.fillna(method='bfill', axis=1)
S = df.iloc[:, 0]
blood_collected_date = S.str.replace(r'T.*', '').astype('datetime64')
blood_sample = set(blood_collected_date.index)


# disease patients
disease_patients = dict()
with open('../../behavior_data/disease_patient_firstOccur.txt', 'r') as infile:
    for i, line in enumerate(infile):
        if i < 1:
            continue
        str1 = line.strip('\r\n')
        str2 = str1.split('\t')
        disease_patients[str2[0][:3]] = set(str2[1].split(';')) & blood_sample
    infile.close()


# disease diagnosis date field
disease_diagnosisDate_field = dict()
df = pd.read_csv('../../behavior_data/Data_Dictionary_Showcase_supplement.csv')
df = df[df['Path'].str.contains('First occurrences >') & df['Field'].str.contains('Date ')]
for fieldID, field in df[['FieldID', 'Field']].values.tolist():
    code = re.findall(r'Date (.*?) first reported', field)[0]
    disease_diagnosisDate_field[code] = str(fieldID)


df = pd.read_csv('phenotype_significant_multimorbidity.csv')
df = df[df['phenotype'].str.contains('Blood~')]
multimorbidity_set = set(zip(df['code1'], df['code2']))

phenotype_sample_score = dict()
for each in df['phenotype']:
    if each in phenotype_sample_score:
        continue
    field = each.split('~')[-1].split('*')[0]
    temp = pd.read_csv('../../behavior_data/all_split_field_20210331/' + field + '.csv', index_col=0)
    temp.index = temp.index.map(str)
    temp = temp.dropna(axis=0, how='all')
    temp = temp.fillna(method='bfill', axis=1)
    phenotype_sample_score[each] = temp.iloc[:, 0]


# disease diagnosis date
disease_diagnosis_date = dict()
disease_set = set(df['code1']) | set(df['code2'])
for each in disease_set:
    temp = pd.read_csv('../../behavior_data/all_split_field_20210331/' + disease_diagnosisDate_field[each] + '.csv', index_col=0)
    temp.index = temp.index.map(str)
    temp = temp.dropna(axis=0, how='all')
    disease_diagnosis_date[each] = temp.iloc[:, 0].astype('datetime64')

print('data load end.......')

multimorbidity_ls = list(multimorbidity_set)
multimorbidity_ls.sort()
for i, (code1, code2) in enumerate(multimorbidity_ls):
    if code1 >= code2:
        raise Exception
    print(i, code1, code2)
    code1_patients = list(disease_patients[code1])
    code2_patients = list(disease_patients[code2])
    code1_patients.sort()
    code2_patients.sort()

    code1_blood_date = blood_collected_date.reindex(code1_patients)
    code2_blood_date = blood_collected_date.reindex(code2_patients)

    code1_diagnosis_date = disease_diagnosis_date[code1].reindex(code1_patients)
    code2_diagnosis_date = disease_diagnosis_date[code2].reindex(code2_patients)

    code1_before_sample = set(code1_blood_date[code1_blood_date > code1_diagnosis_date].index)
    code1_after_sample = set(code1_blood_date[code1_blood_date < code1_diagnosis_date].index)
    code2_before_sample = set(code2_blood_date[code2_blood_date > code2_diagnosis_date].index)
    code2_after_sample = set(code2_blood_date[code2_blood_date < code2_diagnosis_date].index)

    if len(code1_before_sample & code2_after_sample) > 30:
        data = pd.DataFrame()
        sample_order = list(code1_before_sample - code2_before_sample)
        sample_order.sort()
        temp = disease_diagnosis_date[code2].reindex(sample_order).fillna(pd.to_datetime('2021-03-31')) \
               - blood_collected_date.reindex(sample_order)
        data['day_delta'] = [each.days for each in list(temp)]
        data['sex'] = sample_sex.reindex(sample_order).tolist()
        data['age'] = sample_age.reindex(sample_order).tolist()
        data['event'] = [1 if each in code2_after_sample else 0 for each in sample_order]

        df1 = df[(df['code1'] == code1) & (df['code2'] == code2)]
        for phenotype in df1['phenotype']:
            data[phenotype] = phenotype_sample_score[phenotype].reindex(sample_order).tolist()

        data = data.dropna(axis=0, how='any')

        cph = CoxPHFitter()
        try:
            cph.fit(data, duration_col='day_delta', event_col='event')
        except:
            continue
        cph.print_summary()
        temp = cph.summary
        temp.to_csv('survival_analysis_result/' + '_'.join([code1, code2, str(cph.concordance_index_)]) + '.csv')

    if len(code2_before_sample & code1_after_sample) > 30:
        data = pd.DataFrame()
        sample_order = list(code2_before_sample - code1_before_sample)
        sample_order.sort()
        temp = disease_diagnosis_date[code1].reindex(sample_order).fillna(pd.to_datetime('2021-03-31')) \
               - blood_collected_date.reindex(sample_order)
        data['day_delta'] = [each.days for each in list(temp)]
        data['sex'] = sample_sex.reindex(sample_order).tolist()
        data['age'] = sample_age.reindex(sample_order).tolist()
        data['event'] = [1 if each in code1_after_sample else 0 for each in sample_order]

        df1 = df[(df['code1'] == code1) & (df['code2'] == code2)]
        for phenotype in df1['phenotype']:
            data[phenotype] = phenotype_sample_score[phenotype].reindex(sample_order).tolist()
        data = data.dropna(axis=0, how='any')

        cph = CoxPHFitter()
        try:
            cph.fit(data, duration_col='day_delta', event_col='event')
        except:
            continue
        cph.print_summary()
        temp = cph.summary
        temp.to_csv('survival_analysis_result/' + '_'.join([code2, code1, str(cph.concordance_index_)]) + '.csv')



# ............. survival analysis summary
file_list = os.listdir('survival_analysis_result/')
total = 0
for each in file_list:
    df = pd.read_csv('survival_analysis_result/' + each)
    df = df[df['covariate'].str.contains('Blood~')]
    total += df.shape[0]
bf_threshold = 0.05 / total

significant_file = []
result = []
for each in file_list:
    print(each)
    df = pd.read_csv('survival_analysis_result/' + each)
    df = df[df['covariate'].str.contains('Blood~')]
    df = df[df['p'] < bf_threshold]
    if df.shape[0] != 0:
        code1, code2 = each.split('_')[0:2]
        significant_file.append(each)
        for each1 in df.values.tolist():
            result.append([code1, code2] + each1)

print('total significant file:', len(significant_file))

df1 = pd.DataFrame(result, columns=['code1', 'code2'] + list(df.columns))
df1.to_csv('multimorbidity_phenotype_prognosed.csv', index=False)