import os
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from itertools import combinations

from sklearn.metrics import precision_recall_curve, f1_score
import matplotlib.pyplot as plt
plt.rc('font', family='Helvetica')
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42





def get_PredLabel(dataset):
    file_list = os.listdir('multimorbidity_prediction_result/' + dataset)

    diseasePair_prob_sum = defaultdict(int)
    diseasePair_label = defaultdict(int)
    for i, each in enumerate(file_list):
        df = pd.read_csv('multimorbidity_prediction_result/' + dataset + '/' + each)
        for code1, code2, label, prob in df.values.tolist():
            if code1 < code2:
                key = code1 + '-' + code2
            else:
                key = code2 + '-' + code1
            diseasePair_prob_sum[key] += prob
            diseasePair_label[key] = label

    list1 = []
    for each in diseasePair_prob_sum:
        code1, code2 = each.split('-')
        mean_prob = diseasePair_prob_sum[each] / 1000
        list1.append([code1, code2, mean_prob, diseasePair_label[each]])
    df = pd.DataFrame(list1, columns=['code1', 'code2', 'prob', 'label'])

    precision, recall, thresholds = precision_recall_curve(df['label'], df['prob'])
    f1_scores = 2*recall*precision/(recall+precision)
    best_f1 = np.max(f1_scores)
    best_threshold = thresholds[np.argmax(f1_scores)]
    print('Best F1-Score: ', best_f1)
    print('Best threshold: ', best_threshold)

    diseasePair_predLabel = []
    for code1, code2, prob, label in df.values.tolist():
        if prob >= best_threshold:
            diseasePair_predLabel.append([code1, code2, prob, 1, label])
        else:
            diseasePair_predLabel.append([code1, code2, prob, 0, label])

    df = pd.DataFrame(diseasePair_predLabel, columns=['code1', 'code2', 'pred_probability', 'pred_label', 'label'])
    df.to_csv(dataset + '_' + 'predict_multimorbidity.csv', index=False)



def get_novel_multimorbidity(dataset):
    df = pd.read_csv(dataset + '_' + 'predict_multimorbidity.csv')
    # novel
    df1 = df[(df['pred_label'] == 1) & (df['label'] == 0)][['code1', 'code2']]
    novel_multimorbidty = set()
    for code1, code2 in df1.values.tolist():
        if code1 < code2:
            novel_multimorbidty.add((code1, code2))
        else:
            novel_multimorbidty.add((code2, code1))
    # known
    df1 = df[df['label'] == 1][['code1', 'code2']]
    known_multimorbidty = set()
    for code1, code2 in df1.values.tolist():
        if code1 < code2:
            known_multimorbidty.add((code1, code2))
        else:
            known_multimorbidty.add((code2, code1))
    # disease
    all_diseases = set(df['code1']) | set(df['code2'])

    return novel_multimorbidty, known_multimorbidty, all_diseases





get_PredLabel('UKB')
get_PredLabel('HuDiNe')
ukb_novel_multimorbidty, ukb_known_multimorbidty, ukb_disease = get_novel_multimorbidity('UKB')
hudine_novel_multimorbidty, hudine_known_multimorbidty, hudine_disease = get_novel_multimorbidity('HuDiNe')

# random select same number of disease-pairs and enrich (ukb)
ukb_diseasePair = set()
for code1, code2 in combinations(ukb_disease, 2):
    if code1 < code2:
        key = (code1, code2)
    else:
        key = (code2, code1)
    if key not in ukb_known_multimorbidty:
        ukb_diseasePair.add(key)

repeated = len(ukb_novel_multimorbidty & hudine_known_multimorbidty)
c = 0
for i in range(10000):
    set1 = set(random.sample(ukb_diseasePair, len(ukb_novel_multimorbidty)))
    if len(set1 & hudine_known_multimorbidty) >= repeated:
        c += 1
print('ukb permutation result:', c/10000)


# random select same number of disease-pairs and enrich (hudine)
hudine_diseasePair = set()
for code1, code2 in combinations(hudine_disease, 2):
    if code1 < code2:
        key = (code1, code2)
    else:
        key = (code2, code1)
    if key not in hudine_known_multimorbidty:
        hudine_diseasePair.add(key)

repeated = len(hudine_novel_multimorbidty & ukb_known_multimorbidty)
c = 0
for i in range(10000):
    set1 = set(random.sample(hudine_diseasePair, len(hudine_novel_multimorbidty)))
    if len(set1 & ukb_known_multimorbidty) >= repeated:
        c += 1
print('hudine permutation result:', c/10000)


# overlap significance of the two novel discovered multimorbidity set
c = 0
overlapped = len(ukb_novel_multimorbidty & hudine_novel_multimorbidty)
for i in range(10000):
    set1 = set(random.sample(ukb_diseasePair, len(ukb_novel_multimorbidty)))
    set2 = set(random.sample(hudine_diseasePair, len(hudine_novel_multimorbidty)))
    if len(set1 & set2) >= overlapped:
        c += 1
print('novel discovered multimorbidity overlap significance:', c/10000)


# save results for venn plot using r
list1 = []
for code1, code2 in ukb_known_multimorbidty:
    list1.append([code1 + '*' + code2])
df = pd.DataFrame(list1, columns=['code'])
df.to_csv('ukb_known_multimorbidity.csv', index=False)

list1 = []
for code1, code2 in ukb_novel_multimorbidty:
    list1.append([code1 + '*' + code2])
df = pd.DataFrame(list1, columns=['code'])
df.to_csv('ukb_novel_multimorbidity.csv', index=False)

list1 = []
for code1, code2 in hudine_known_multimorbidty:
    list1.append([code1 + '*' + code2])
df = pd.DataFrame(list1, columns=['code'])
df.to_csv('hudine_known_multimorbidity.csv', index=False)

list1 = []
for code1, code2 in hudine_novel_multimorbidty:
    list1.append([code1 + '*' + code2])
df = pd.DataFrame(list1, columns=['code'])
df.to_csv('hudine_novel_multimorbidity.csv', index=False)





# ------- test the significance of the cross-physiological multimorbidity proportion
disease_category = dict()
df = pd.read_excel('data/disease_category.xlsx')
for code, category in df.values.tolist():
    disease_category[code[:3]] = category

ukb_novel_crossPhy_num = 0
for code1, code2 in ukb_novel_multimorbidty:
    if disease_category[code1] != disease_category[code2]:
        ukb_novel_crossPhy_num += 1

hudine_novel_crossPhy_num = 0
for code1, code2 in hudine_novel_multimorbidty:
    if disease_category[code1] != disease_category[code2]:
        hudine_novel_crossPhy_num += 1

total = 0
for i in range(10000):
    list1 = random.sample(ukb_diseasePair, len(ukb_novel_multimorbidty))
    count = 0
    for code1, code2 in list1:
        if disease_category[code1] != disease_category[code2]:
            count += 1
    if count >= ukb_novel_crossPhy_num:
        total += 1
print('ukb novel cross-physiological multimorbidity, ', 'true: ', ukb_novel_crossPhy_num, 'random: ', total)


total = 0
for i in range(10000):
    list1 = random.sample(hudine_diseasePair, len(hudine_novel_multimorbidty))
    count = 0
    for code1, code2 in list1:
        if disease_category[code1] != disease_category[code2]:
            count += 1
    if count >= hudine_novel_crossPhy_num:
        total += 1
print('hudine novel cross-physiological multimorbidity, ', 'true: ', hudine_novel_crossPhy_num, 'random: ', total)


# -- plot ukb novel multimorbidity
set1 = set()
set2 = set()
for code1, code2 in ukb_novel_multimorbidty:
    if disease_category[code1] == disease_category[code2]:
        set1.add((code1, code2))
    else:
        set2.add((code1, code2))

ratio = len(set1)/len(ukb_novel_multimorbidty)
sizes = [ratio, 1-ratio]
colors = ['#ffa500', '#6495ed']
labels = [f'Same-physiological ({len(set1)})', f'Cross-physiological ({len(set2)})']
explode = [0.05, 0.]
patches, text1, text2 = plt.pie(sizes, explode, labels, colors, autopct='%3.2f%%', shadow=False, startangle=0,
                                pctdistance=0.8)
plt.title('Novel multimorbidities in the UKB dataset')
plt.savefig('fig4_d.pdf', bbox_inches='tight')
plt.show()


# -- plot hudine novel multimorbidity
set1 = set()
set2 = set()
for code1, code2 in hudine_novel_multimorbidty:
    if disease_category[code1] == disease_category[code2]:
        set1.add((code1, code2))
    else:
        set2.add((code1, code2))

ratio = len(set1)/len(hudine_novel_multimorbidty)
sizes = [ratio, 1-ratio]
colors = ['#ffa500', '#6495ed']
labels = [f'Same-physiological ({len(set1)})', f'Cross-physiological ({len(set2)})']
explode = [0.05, 0.]
patches, text1, text2 = plt.pie(sizes, explode, labels, colors, autopct='%3.2f%%', shadow=False, startangle=0,
                                pctdistance=0.8)
plt.title('Novel multimorbidities in the HuDiNe dataset')
plt.savefig('fig4_e.pdf', bbox_inches='tight')
plt.show()


print('ok')