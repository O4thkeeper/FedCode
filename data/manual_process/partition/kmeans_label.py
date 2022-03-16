import argparse
import pickle
import json

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from tqdm import tqdm


def get_embedding_Kmeans(embedding_exist, corpus, n_clusters, batch_size=32):
    # if not embedding_exist:
    #     encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    #     corpus_embeddings = encoder.encode(corpus, show_progress_bar=True, batch_size=batch_size)
    # else:
    #     corpus_embeddings = corpus
    # print("start Kmeans")
    # clustering_model = KMeans(n_clusters=n_clusters)
    # clustering_model.fit(corpus_embeddings)
    # cluster_assignment = clustering_model.labels_
    # print("end Kmeans")
    cluster_assignment = []
    for i in range(len(corpus)):
        cluster_assignment.append(i % n_clusters)

    return np.array(cluster_assignment)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cluster_number', type=int, default='20')

    parser.add_argument('--batch_size', type=int, default='64')

    parser.add_argument('--data_file', type=str, default='')

    parser.add_argument('--label_file', type=str, default='')

    parser.add_argument('--dataset', type=str, default="")

    args = parser.parse_args()

    corpus = []
    if args.dataset == 'codesearch':
        with open(args.data_file, "r", encoding='utf-8') as f:
            for idx, line in tqdm(enumerate(f)):
                line = line.strip().split('<CODESPLIT>')
                if len(line) != 5:
                    continue
                nl = line[3]
                corpus.append(nl)
    elif args.dataset == 'codedoc':
        with open(args.data_file, encoding="utf-8") as f:
            for line in tqdm(f):
                line = line.strip()
                js = json.loads(line)
                nl = ' '.join(js['docstring_tokens']).replace('\n', '')
                nl = ' '.join(nl.strip().split())
                corpus.append(nl)
    else:
        raise RuntimeError('dataset not support')

    n_clusters = args.cluster_number

    cluster_assignment = get_embedding_Kmeans(False, corpus, n_clusters, args.batch_size)
    with open(args.label_file, 'wb') as f:
        pickle.dump(cluster_assignment, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
