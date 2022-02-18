import pandas as pd
import numpy as np
from scipy import sparse as sp


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges(adj, test_percent=0.1, val_percent=0.05):
    """ Randomly removes some edges from original graph to create
    test and validation sets for link prediction task
    :param adj: complete sparse adjacency matrix of the graph
    :param test_percent: percentage of edges in test set
    :param val_percent: percentage of edges in validation set
    :return: train incomplete adjacency matrix, validation and test sets
    """
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[None, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert adj.diagonal().sum() == 0

    edges_positive, _, _ = sparse_to_tuple(adj)
    # Filtering out edges from lower triangle of adjacency matrix
    edges_positive = edges_positive[edges_positive[:, 1] > edges_positive[:, 0], :]
    # val_edges, val_edges_false, test_edges, test_edges_false = None, None, None, None

    # number of positive (and negative) edges in test and val sets:
    num_test = int(np.floor(edges_positive.shape[0] * test_percent))
    num_val = int(np.floor(edges_positive.shape[0] * val_percent))

    # sample positive edges for test and val sets:
    edges_positive_idx = np.arange(edges_positive.shape[0])
    while True:
        np.random.shuffle(edges_positive_idx)
        val_edge_idx = edges_positive_idx[:num_val]
        test_edge_idx = edges_positive_idx[num_val:(num_val + num_test)]
        test_edges = edges_positive[test_edge_idx] # positive test edges
        val_edges = edges_positive[val_edge_idx] # positive val edges
        train_edges = np.delete(edges_positive, np.hstack([test_edge_idx, val_edge_idx]), axis=0) # positive train edges
        if len(np.unique(train_edges)) == adj.shape[0]:
            break


    # the above strategy for sampling without replacement will not work for
    # sampling negative edges on large graphs, because the pool of negative
    # edges is much much larger due to sparsity, therefore we'll use
    # the following strategy:
    # 1. sample random linear indices from adjacency matrix WITH REPLACEMENT
    # (without replacement is super slow). sample more than we need so we'll
    # probably have enough after all the filtering steps.
    # 2. remove any edges that have already been added to the other edge lists
    # 3. convert to (i,j) coordinates
    # 4. only keep i < j, to ensure they're upper triangle elements
    # 5. remove any duplicate elements if there are any
    # 6. remove any diagonal elements
    # 7. if we don't have enough edges, repeat this process until we get enough
    positive_idx, _, _ = sparse_to_tuple(adj) # [i,j] coord pairs for all true edges
    positive_idx = positive_idx[:, 0] * adj.shape[0] + positive_idx[:, 1] # linear indices
    test_edges_false = np.empty((0, 2), dtype='int64')
    idx_test_edges_false = np.empty((0,), dtype='int64')

    while len(test_edges_false) < len(test_edges):
        # step 1:
        idx = np.random.choice(adj.shape[0]**2, 2*(num_test - len(test_edges_false)), replace=True)
        # step 2:
        idx = idx[~np.in1d(idx, positive_idx, assume_unique=True)]
        idx = idx[~np.in1d(idx, idx_test_edges_false, assume_unique=True)]
        # step 3:
        rowidx = idx // adj.shape[0]
        colidx = idx % adj.shape[0]
        coords = np.vstack((rowidx, colidx)).transpose()
        # step 4:
        uppertrimask = coords[:, 0] < coords[:, 1]
        coords = coords[uppertrimask]
        # step 5:
        coords = np.unique(coords, axis=0) # note: coords are now sorted lexicographically
        np.random.shuffle(coords) # not anymore
        # step 6:
        coords = coords[coords[:, 0] != coords[:, 1]]
        # step 7:
        coords = coords[:min(num_test, coords.shape[0])]
        test_edges_false = np.append(test_edges_false, coords, axis=0)
        idx = coords[:, 0] * adj.shape[0] + coords[:, 1]
        idx_test_edges_false = np.append(idx_test_edges_false, idx)

    val_edges_false = np.empty((0, 2), dtype='int64')
    idx_val_edges_false = np.empty((0,), dtype='int64')
    while len(val_edges_false) < len(val_edges):
        # step 1:
        idx = np.random.choice(adj.shape[0]**2, 2*(num_val - len(val_edges_false)), replace=True)
        # step 2:
        idx = idx[~np.in1d(idx, positive_idx, assume_unique=True)]
        idx = idx[~np.in1d(idx, idx_test_edges_false, assume_unique=True)]
        idx = idx[~np.in1d(idx, idx_val_edges_false, assume_unique=True)]
        # step 3:
        rowidx = idx // adj.shape[0]
        colidx = idx % adj.shape[0]
        coords = np.vstack((rowidx, colidx)).transpose()
        # step 4:
        uppertrimask = coords[:, 0] < coords[:, 1]
        coords = coords[uppertrimask]
        # step 5:
        coords = np.unique(coords, axis=0) # note: coords are now sorted lexicographically
        np.random.shuffle(coords) # not any more
        # step 6:
        coords = coords[coords[:, 0] != coords[:, 1]]
        # step 7:
        coords = coords[:min(num_val, coords.shape[0])]
        val_edges_false = np.append(val_edges_false, coords, axis=0)
        idx = coords[:, 0] * adj.shape[0] + coords[:, 1]
        idx_val_edges_false = np.append(idx_val_edges_false, idx)

    # sanity checks:
    train_edges_linear = train_edges[:, 0] * adj.shape[0] + train_edges[:, 1]
    test_edges_linear = test_edges[:, 0] * adj.shape[0] + test_edges[:, 1]
    val_edges_linear = val_edges[:, 0] * adj.shape[0] + val_edges[:, 1]
    assert not np.any(np.in1d(idx_test_edges_false, positive_idx))
    assert not np.any(np.in1d(idx_val_edges_false, positive_idx))
    assert not np.any(np.in1d(idx_val_edges_false, idx_test_edges_false))
    assert not np.any(np.in1d(test_edges_linear, train_edges_linear))
    assert not np.any(np.in1d(val_edges_linear, train_edges_linear))
    assert not np.any(np.in1d(val_edges_linear, test_edges_linear))

    return train_edges, val_edges, val_edges_false, test_edges, test_edges_false




def get_adj(multimorbidity_df, phenotype_df):
    disease_intersection = set(phenotype_df.index) & (set(multimorbidity_df['code1']) | set(multimorbidity_df['code2']))

    multimorbidity_df = multimorbidity_df[multimorbidity_df['code1'].isin(disease_intersection) & multimorbidity_df['code2'].isin(disease_intersection)]
    disease_order = list(set(multimorbidity_df['code1']) | set(multimorbidity_df['code2']))
    disease_order.sort()

    # ................................................................. #
    disease_multimorbidity_count = []
    for each in disease_order:
        disease_multimorbidity_count.append([each, multimorbidity_df[(multimorbidity_df['code1'] == each) | (multimorbidity_df['code2'] == each)].shape[0]])
    temp = pd.DataFrame(disease_multimorbidity_count, columns=['disease', 'count'])
    temp = temp.sort_values(by='disease').sort_values(by='count', ascending=False)
    disease_order = list(temp['disease'])

    # ................................................................. #
    disease_index = {v: i for i, v in enumerate(disease_order)}
    multimorbidity_df['code1'] = multimorbidity_df['code1'].replace(disease_index)
    multimorbidity_df['code2'] = multimorbidity_df['code2'].replace(disease_index)

    adj = sp.coo_matrix((np.ones(multimorbidity_df.shape[0]), (multimorbidity_df['code1'], multimorbidity_df['code2'])),
                        shape=(len(disease_index), len(disease_index)), dtype=float).toarray()
    adj = adj + adj.T
    assert adj.diagonal().sum() == 0

    return adj, disease_index



# read data
for dataset in ['UKB', 'DuDiNe']:
    if dataset == 'UKB':
        multimorbidity_df = pd.read_csv('data/ukb_multimorbidity.csv')
    elif dataset == 'DuDiNe':
        multimorbidity_df = pd.read_csv('data/hudine_multimorbidity.csv')

    phenotype_df = pd.read_table('data/disease_phenotype_score_data_processed.csv', index_col=0)
    phenotype_df.index = [index[:3] for index in phenotype_df.index]

    adj, disease_index = get_adj(multimorbidity_df, phenotype_df)

    i = 0
    while i < 100:
        train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(sp.coo_matrix(adj))
        with open('data_split4graphNet/' + dataset + '/data' + str(i) + '.npy', 'wb') as f:
            np.save(f, train_edges)
            np.save(f, val_edges)
            np.save(f, val_edges_false)
            np.save(f, test_edges)
            np.save(f, test_edges_false)
        f.close()
        i += 1

    with open('data_split4graphNet/' + dataset + '/disease_index.txt', 'w+') as f:
        for each in disease_index:
            f.write('\t'.join([each, str(disease_index[each])]) + '\n')
        f.close()



print('ok')