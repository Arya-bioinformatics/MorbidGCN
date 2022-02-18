import sys
sys.path.append('./graph_network')
import numpy as np
import pandas as pd
from ge import Node2Vec
import networkx as nx
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import argparse



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def evaluate_embeddings(embeddings, pos_edges, neg_edges):
    preds = []
    for edge in pos_edges:
        preds.append(sigmoid(embeddings[edge[0]].dot(embeddings[edge[1]])))
    for edge in neg_edges:
        preds.append(sigmoid(embeddings[edge[0]].dot(embeddings[edge[1]])))
    labels = [1] * pos_edges.shape[0] + [0] * neg_edges.shape[0]
    auroc = roc_auc_score(labels, preds)
    precision, recall, thresholds = precision_recall_curve(labels, preds)
    auprc = auc(recall, precision)
    return auroc, auprc



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='input ......')
    parser.add_argument('--dataset', type=str, default='UKB', help='multimorbidity dataset')
    parser.add_argument('--walk_length', type=int, default=2, help='walk length')
    parser.add_argument('--num_walks', type=int, default=120, help='number of walks')
    parser.add_argument('--window_size', type=int, default=2, help='window size')
    parser.add_argument('--epochs', type=int, default=1, help='epochs')
    parser.add_argument('--embed_size', type=int, default=64, help='embedding size')
    parser.add_argument('--p', type=float, default=1., help='p')
    parser.add_argument('--q', type=float, default=1., help='q')
    parser.add_argument('--outfile', type=str, default=None, help='file that are used to save the result')

    args = parser.parse_args()
    walk_length, num_walks, window_size, epochs, embed_size, p, q = args.walk_length, args.num_walks, args.window_size, \
                                                                    args.epochs, args.embed_size, args.p, args.q

    result = []
    for i in range(0, 100):
        with open('data_split4graphNet/UKB/data' + str(i) + '.npy', 'rb') as f:
            train_edge = np.load(f)
            val_edge = np.load(f)
            val_edge_false = np.load(f)
            test_edge = np.load(f)
            test_edge_false = np.load(f)
        f.close()

        G = nx.Graph()
        G.add_edges_from(train_edge)

        model = Node2Vec(G, walk_length=walk_length, num_walks=num_walks, p=p, q=q, workers=1, use_rejection_sampling=0)
        model.train(embed_size=embed_size, window_size=window_size, iter=epochs)
        embeddings = model.get_embeddings()

        val_auroc, val_auprc = evaluate_embeddings(embeddings, val_edge, val_edge_false)
        test_auroc, test_auprc = evaluate_embeddings(embeddings, test_edge, test_edge_false)

        print('val_auroc:', val_auroc, 'val_auprc:', val_auprc, 'test_auroc', test_auroc, 'test_auprc', test_auprc)

        result.append([walk_length, num_walks, window_size, epochs, embed_size, p, q, i, val_auroc, val_auprc, test_auroc, test_auprc])

    df = pd.DataFrame(result, columns=['walk_length', 'num_walks', 'window_size', 'epochs', 'embed_size', 'p', 'q', 'i',
                                       'val_auroc', 'val_auprc', 'test_auroc', 'test_auprc'])
    df.to_csv('gn_result/' + args.dataset + '/' + args.outfile + '.csv', index=False)

    print('ok')