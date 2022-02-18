import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['pdf.fonttype'] = 42
plt.rc('font', family='Helvetica')




def plot_UKB_performance(outpath):

    file_list = ['deepwalk.csv', 'node2vec.csv', 'no_features.csv', 'all_features.csv',
                 'selected_features.csv', 'only_selected_features.csv']

    list1 = []
    for each in file_list:
        key = each.replace('.csv', '')
        df = pd.read_csv('ukb_plot_data/' + each)
        df = df.iloc[:, -2:]
        for each1 in df.values.tolist():
            list1.append([key, 'AUROC', each1[0]])
            list1.append([key, 'AUPRC', each1[1]])

    df1 = pd.DataFrame(list1, columns=['Group', 'Measure', 'Score'])

    plt.figure(figsize=(6, 6))
    ax = sns.barplot(x='Group', y='Score', hue='Measure', data=df1, palette=['#1e90ff', '#7cb5ec'], errwidth=1, capsize=0.1)
    ax.tick_params(axis='x', rotation=30)
    ax.set_ylim(0.84, 0.95)
    ax.set_xlabel(None)
    ax.legend_.set_title(None)
    plt.savefig(outpath, bbox_inches='tight')
    plt.show()



def plot_HuDiNe_performance(outpath):
    file_list = ['deepwalk.csv', 'node2vec.csv', 'no_features.csv', 'all_features.csv',
                 'selected_features.csv', 'only_selected_features.csv']

    list1 = []
    for each in file_list:
        key = each.replace('.csv', '')
        df = pd.read_csv('hudine_plot_data/' + each)
        df = df.iloc[:, -2:]
        for each1 in df.values.tolist():
            list1.append([key, 'AUROC', each1[0]])
            list1.append([key, 'AUPRC', each1[1]])

    df1 = pd.DataFrame(list1, columns=['Group', 'Measure', 'Score'])

    # plot hudine
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6), gridspec_kw={'height_ratios': [7, 2]})

    sns.barplot(x='Group', y='Score', hue='Measure', data=df1, palette=['#1e90ff', '#7cb5ec'], errwidth=1, capsize=0.1,
                ax=ax1)
    sns.barplot(x='Group', y='Score', hue='Measure', data=df1, palette=['#1e90ff', '#7cb5ec'], errwidth=1, capsize=0.1,
                ax=ax2)

    ax1.set_ylim(0.82, 0.885)  # outliers only
    ax2.set_ylim(0.72, .74)  # most of the data

    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.get_legend().remove()
    ax1.set_xlabel(None)
    ax2.set_xlabel(None)
    ax2.set_ylabel(None)
    ax1.legend_.set_title(None)
    ax1.set_xticks([])

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=10,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
    plt.xticks(range(6),
               ['DeepWalk', 'Node2Vec', 'No_features', 'All_features', 'Selected_features', 'Only_selected_features'],
               rotation=30)
    plt.savefig(outpath, bbox_inches='tight')
    plt.show()


plot_UKB_performance('fig4_a.pdf')
plot_HuDiNe_performance('fig4_b.pdf')



print('ok')