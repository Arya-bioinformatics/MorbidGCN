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





def plot_novelMultimorbidity_topologySimilarity(dataset='UKB', outfile='fig_s3_1.pdf'):
    # phenotype data
    phenotype_df = pd.read_csv('data/disease_phenotype_score_data_processed.csv', index_col=0)
    phenotype_df.index = [index[:3] for index in phenotype_df.index]

    if dataset == 'UKB':
        # UKB comorbidity data
        comorbidity_df = pd.read_table('data/comorbidity_filter_firstOccur_timeWindow.txt')[
            ['ICD10 code1', 'ICD10 code2']]
        comorbidity_df.columns = ['code1', 'code2']
    else:
        # Hidalgo comorbidity data
        comorbidity_df = pd.read_csv('data/hidalgo_multimorbidity_fdr.csv')


    disease_intersection = set(phenotype_df.index) & (set(comorbidity_df['code1']) | set(comorbidity_df['code2']))
    comorbidity_df = comorbidity_df[
        comorbidity_df['code1'].isin(disease_intersection) & comorbidity_df['code2'].isin(disease_intersection)]
    disease_order = list(set(comorbidity_df['code1']) | set(comorbidity_df['code2']))
    phenotype_df = phenotype_df.reindex(disease_order)

    edge_list = list(zip(comorbidity_df['code1'], comorbidity_df['code2']))

    G = nx.Graph()
    G.add_edges_from(edge_list)

    novel_multimorbidity = set()
    if dataset == 'UKB':
        df = pd.read_csv('ukb_novel_multimorbidity.csv')
    else:
        df = pd.read_csv('hudine_novel_multimorbidity.csv')
    for each in df['code']:
        code1, code2 = each.split('*')
        novel_multimorbidity.add((code1, code2))

    non_multimorbidity = set()
    for code1, code2 in combinations(disease_order, 2):
        if ((code1, code2) in set(edge_list)) | ((code2, code1) in set(edge_list)):
            continue
        elif ((code1, code2) in novel_multimorbidity) | ((code2, code1) in novel_multimorbidity):
            continue
        else:
            non_multimorbidity.add((code1, code2))

    list1 = []
    for code1, code2 in novel_multimorbidity:
        set1 = set(nx.single_source_shortest_path(G, code1, 1).keys())
        set2 = set(nx.single_source_shortest_path(G, code2, 1).keys())
        list1.append(len(set1 & set2) / len(set1 | set2))

    list2 = []
    for code1, code2 in non_multimorbidity:
        set1 = set(nx.single_source_shortest_path(G, code1, 1).keys())
        set2 = set(nx.single_source_shortest_path(G, code2, 1).keys())
        list2.append(len(set1 & set2) / len(set1 | set2))

    print(ttest_ind(list1, list2))

    print(np.mean(list1), np.mean(list2))

    plt.figure(figsize=(5, 4))
    sns.distplot(list1, kde=True, label='novel_multimorbidity')
    sns.distplot(list2, kde=True, label='non-multimorbidity')
    plt.legend()
    if dataset == 'UKB':
        plt.title('UKB dataset')
    else:
        plt.title('HuDiNe dataset')
    plt.ylabel('Density of topology similarity scores')
    plt.savefig(outfile, bbox_inches='tight')
    plt.show()




def plot_novelMultimorbidity_phenotypeSimilarity(dataset='UKB', outfile='fig_s3_2.pdf'):
    # phenotype data
    phenotype_df = pd.read_csv('data/disease_phenotype_score_data_processed.csv', index_col=0)
    phenotype_df.index = [index[:3] for index in phenotype_df.index]

    # UKB comorbidity data
    if dataset == 'UKB':
        comorbidity_df = pd.read_table('data/comorbidity_filter_firstOccur_timeWindow.txt')[
            ['ICD10 code1', 'ICD10 code2']]
        comorbidity_df.columns = ['code1', 'code2']
        df = pd.read_csv('a1.csv')
        df.columns = ['Phenotype', 'score']
        selected_phenotype = set(df['Phenotype'])
    else:
        # Hidalgo comorbidity data
        comorbidity_df = pd.read_csv('data/hidalgo_multimorbidity_fdr.csv')
        df = pd.read_csv('a2.csv')
        df.columns = ['Phenotype', 'score']
        selected_phenotype = set(df['Phenotype'])

    disease_intersection = set(phenotype_df.index) & (set(comorbidity_df['code1']) | set(comorbidity_df['code2']))
    comorbidity_df = comorbidity_df[
        comorbidity_df['code1'].isin(disease_intersection) & comorbidity_df['code2'].isin(disease_intersection)]
    disease_order = list(set(comorbidity_df['code1']) | set(comorbidity_df['code2']))
    disease_index = {v: i for i, v in enumerate(disease_order)}
    phenotype_df = phenotype_df.reindex(disease_order)

    features = np.concatenate(
        [phenotype_df[phenotype_df > 0].fillna(0).values, -phenotype_df[phenotype_df < 0].fillna(0).values], axis=-1)
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    features = pd.DataFrame(features)
    features.columns = [col + '*pos' for col in phenotype_df.columns] + [col + '*neg' for col in phenotype_df.columns]
    features.index = phenotype_df.index
    features = features.loc[:, features.columns.isin(selected_phenotype)]

    data = features.values
    adj = data.dot(data.T)

    edge_list = list(zip(comorbidity_df['code1'], comorbidity_df['code2']))
    novel_multimorbidity = set()
    if dataset == 'UKB':
        df = pd.read_csv('ukb_novel_multimorbidity.csv')
    else:
        df = pd.read_csv('hudine_novel_multimorbidity.csv')
    for each in df['code']:
        code1, code2 = each.split('*')
        novel_multimorbidity.add((code1, code2))

    non_multimorbidity = set()
    for code1, code2 in combinations(disease_order, 2):
        if ((code1, code2) in set(edge_list)) | ((code2, code1) in set(edge_list)):
            continue
        elif ((code1, code2) in novel_multimorbidity) | ((code2, code1) in novel_multimorbidity):
            continue
        else:
            non_multimorbidity.add((code1, code2))

    # similarity for novel multimorbidity
    list1 = []
    for i, (code1, code2) in enumerate(novel_multimorbidity):
        print(i, 'novel_multimorbidity')
        id1 = disease_index[code1]
        id2 = disease_index[code2]
        list1.append(adj[id1, id2])

    # similarity for non-multimorbidity
    list2 = []
    for i, (code1, code2) in enumerate(non_multimorbidity):
        id1, id2 = disease_index[code1], disease_index[code2]
        list2.append(adj[id1, id2])

    print(ttest_ind(list1, list2))
    print(np.mean(list1), np.mean(list2))

    plt.figure(figsize=(5, 4))
    sns.distplot(list1, kde=True, label='novel_multimorbidity')
    sns.distplot(list2, kde=True, label='non-multimorbidity')
    plt.legend()
    if dataset == 'UKB':
        plt.title('UKB dataset')
    else:
        plt.title('HuDiNe dataset')
    plt.ylabel('Density of phenotype similarity scores')
    plt.savefig(outfile, bbox_inches='tight')
    plt.show()



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



# topological and phenotypic similarity of novel multimorbidity
plot_novelMultimorbidity_topologySimilarity(dataset='UKB', outfile='fig_s3_1.pdf')
plot_novelMultimorbidity_phenotypeSimilarity(dataset='UKB', outfile='fig_s3_1.pdf')





print('ok')