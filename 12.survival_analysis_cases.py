import pandas as pd
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

list1 = []
list2 = []
for index, each in zip(df.index, df.values.tolist()):
    for each1 in each:
        if isinstance(each1, str):
            list1.append(index)
            list2.append(each1)
            break
S = pd.Series(list2, index=list1)
blood_collected_date = S.str.replace(r'T.*', '').astype('datetime64')
blood_sample = set(blood_collected_date.index)


# disease patients
disease_patients = dict()
diabetes_disease_sample = set()
with open('../../behavior_data/disease_patient_firstOccur.txt', 'r') as infile:
    for i, line in enumerate(infile):
        if i < 1:
            continue
        str1 = line.strip('\r\n')
        str2 = str1.split('\t')
        if str2[0][:3] not in ['G30', 'E11', 'E10', 'I10']:
            continue
        disease_patients[str2[0][:3]] = set(str2[1].split(';')) & blood_sample
        if (str2[0][:3] == 'E10') | (str2[0][:3] == 'E11'):
            diabetes_disease_sample = diabetes_disease_sample | (set(str2[1].split(';')) & blood_sample)
    infile.close()


# disease diagnosis date field
disease_diagnosisDate_field = dict()
df = pd.read_csv('../../behavior_data/Data_Dictionary_Showcase_supplement.csv')
df = df[df['Path'].str.contains('First occurrences >') & df['Field'].str.contains('Date ')]
for fieldID, field in df[['FieldID', 'Field']].values.tolist():
    code = re.findall(r'Date (.*?) first reported', field)[0]
    disease_diagnosisDate_field[code] = str(fieldID)


multimorbidity_set = set([( 'G30', 'I10')])
phenotype_ls = ['continuous~Blood~Biological samples > Blood assays > Blood biochemistry~30750*Glycated haemoglobin (HbA1c)',
                'continuous~Blood~Biological samples > Blood assays > Blood biochemistry~30720*Cystatin C']
# 'continuous~Blood~Biological samples > Blood assays > Blood biochemistry~30720*Cystatin C'

phenotype_sample_score = dict()
for each in phenotype_ls:
    if each in phenotype_sample_score:
        continue
    field = each.split('~')[-1].split('*')[0]
    temp = pd.read_csv(field + '.csv', index_col=0)
    temp.index = temp.index.map(str)
    temp = temp.dropna(axis=0, how='all')
    temp = temp.fillna(method='bfill', axis=1)
    phenotype_sample_score[each] = temp.iloc[:, 0]


# disease diagnosis date
disease_diagnosis_date = dict()
disease_set = set()
for each in multimorbidity_set:
    disease_set.add(each[0])
    disease_set.add(each[1])

for each in disease_set:
    temp = pd.read_csv(disease_diagnosisDate_field[each] + '.csv', index_col=0)
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
        print('*******')
        data = pd.DataFrame()
        sample_order = list(code1_before_sample - code2_before_sample)
        sample_order.sort()
        temp = disease_diagnosis_date[code2].reindex(sample_order).fillna(pd.to_datetime('2021-03-31')) \
               - blood_collected_date.reindex(sample_order)
        data['day_delta'] = [each.days for each in list(temp)]
        data['Sex'] = sample_sex.reindex(sample_order).tolist()
        data['Age'] = sample_age.reindex(sample_order).tolist()
        data['event'] = [1 if each in code2_after_sample else 0 for each in sample_order]
        # data['Diabetes'] = [1 if each in diabetes_disease_sample else 0 for each in sample_order]

        for phenotype in phenotype_ls:
            if '30720' in phenotype:
                key = 'Cystatin C'
            if '30750' in phenotype:
                key = 'HbA1c'
            data[key] = phenotype_sample_score[phenotype].reindex(sample_order).tolist()

        data = data.dropna(axis=0, how='any')

        cph = CoxPHFitter()
        try:
            cph.fit(data, duration_col='day_delta', event_col='event')
        except:
            continue
        cph.print_summary()
        temp = cph.summary
        temp.to_csv('b1.csv')

        plt.figure(figsize=(6, 2))
        cph.plot()
        plt.savefig('a1.pdf', bbox_inches='tight')
        plt.show()


    if len(code2_before_sample & code1_after_sample) > 30:
        print('############')
        data = pd.DataFrame()
        sample_order = list(code2_before_sample - code1_before_sample)
        sample_order.sort()
        temp = disease_diagnosis_date[code1].reindex(sample_order).fillna(pd.to_datetime('2021-03-31')) \
               - blood_collected_date.reindex(sample_order)
        data['day_delta'] = [each.days for each in list(temp)]
        data['Sex'] = sample_sex.reindex(sample_order).tolist()
        data['Age'] = sample_age.reindex(sample_order).tolist()
        data['event'] = [1 if each in code1_after_sample else 0 for each in sample_order]
        # data['Diabetes'] = [1 if each in diabetes_disease_sample else 0 for each in sample_order]

        for phenotype in phenotype_ls:
            if '30720' in phenotype:
                key = 'Cystatin C'
            if '30750' in phenotype:
                key = 'HbA1c'
            data[key] = phenotype_sample_score[phenotype].reindex(sample_order).tolist()
        data = data.dropna(axis=0, how='any')

        cph = CoxPHFitter()
        try:
            cph.fit(data, duration_col='day_delta', event_col='event')
        except:
            continue
        cph.print_summary()
        temp = cph.summary
        temp.to_csv('c.csv')

        plt.figure(figsize=(6, 2))
        cph.plot(hazard_ratios=True, columns=['HbA1c', 'Cystatin C', 'Age', 'Sex'], elinewidth=0.5, capsize=0, capthick=0.1,
                      markeredgewidth=0.5, markersize=5)

        plt.savefig('fig5_c1.pdf', bbox_inches='tight')
        plt.show()


print('ok')