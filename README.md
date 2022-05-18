# MorbidGCN integrates population phenotypes and disease network for multimorbidity prediction based on graph convolutional network
In this manuscript, we develop a model named MorbidGCN based on the graph convolutional network (GCN), to predict missing multimorbidity relationships among diseases by utilizing two types of information, i.e., the population phenotypes from the UK Biobank (UKB) and the disease network constructed by the known multimorbidities.

The MorbidGCN framework mainly include four steps:
1. population phenotype processing
2. disease-phenotype scores quantification
3. feature selection
4. multimorbidity prediction based on GCN layers


## Contents
* data: the source data and the generated data by our analysis
* baseline_method: other biological data in our compared methods
* data_split4graphNet: dataset partitions generated for graph network models
* graph network: source codes for graph noetwork models, such as DeepWalk, Node2Vec, and GCN
* 1.disease_phenotype_score.py: source codes for phenotype processing and disease-phenotype score quantification
* 2.multimorbidity_dataset.py: source codes for generating the UKB and the HuDiNe datasets
* 3.feature_selection_cv.py: source codes for feature selection based on 10-fold cross validation
* 3.feature_selection_plot.py: source codes for generating figures 2 and S1
* 4.phenotype_similarity.py: source codes for multimorbidity prediction based on disease similarities in population phenotypes and other biological data, and also for generating figure 3a-d
* 5.physiology_performance.py: source codes for same-multimorbidity and cross-multimorbidity prediction and figure 3e, f
* 6.plot_otherStatistics_correCoef.py:source codes for multimorbidity prediciton based on other disease-phenotype scores and figure s2
* 6.score_by_other_statistics.py: source codes for generating disease-phenotype scores by other statistics
* 7.data_for_graphNetwork.py: source codes for generating dataset splits 
* 8.deepwalk_main.py: sources codes for training and evulation the deepwalk model for mulrimorbidity prediction
* 9.node2vec_main.py: sources codes for training and evulation the node2vec model for mulrimorbidity prediction
* 10.gcn_main.py: sources codes for training and evulation the MorbidGCN model for mulrimorbidity prediction
* 10.gcn_main_summary.py: sources codes for figure 4a, b
* 11.gcn_predict_novel.py: source codes for predicting multimorbidity probabilities between disease-apirs
* 11.gcn_predict_novel_summary.py: source codes for identifying novel multimrobidities, evulating the novel identified multimorbidities and figure 4d
* 11.venn.R: source codes for generating figure 4c
* 12.feature_contribution.py: source codes for phenotype differential analysis
* 12.feature_contribution_summary.py: source codes for generating figures 5a, b and s3, s4
* 12.survival_analysis.py: source codes for prognosis analysis of multimorbidity occurrence
* 12.survival_analysis_cases.py: source codes for figure 5c and s5
* The file "Supplementary Tables.xlsx" contains Supplementary Tables S1-S11 referred in the original paper.


## Cite
Please cite our paper if you use this code in your own work:

Guiying Dong, Zi-Chao Zhang, Jianfeng Feng, Fengzhu Sun, and Xing-Ming Zhao. MorbidGCN integrates population phenotypes and disease network for multimorbidity prediction based on graph convolutional network.


## Contact
Guiying Dong: dongguiying2017@163.com
