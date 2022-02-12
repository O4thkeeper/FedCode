import pickle
import subprocess
import argparse
import numpy as np


def partition_class_samples_with_dirichlet_distribution(N, alpha, client_num, idx_batch, idx_k):
    np.random.shuffle(idx_k)
    proportions = np.random.dirichlet(np.repeat(alpha, client_num))

    proportions = np.array([
        p * (len(idx_j) < N / client_num)
        for p, idx_j in zip(proportions, idx_batch)
    ])
    proportions = proportions / proportions.sum()
    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

    idx_batch = [
        idx_j + idx.tolist()
        for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
    ]

    return idx_batch


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--client_number", type=int, default="100")

    parser.add_argument("--data_file", type=str, default="")

    parser.add_argument("--partition_file", type=str, default="")

    parser.add_argument("--kmeans_num", type=int)

    parser.add_argument("--beta", type=float)

    parser.add_argument("--min_size", type=int, default=100)

    args = parser.parse_args()

    client_num = args.client_number
    beta = args.beta

    file_length = int(subprocess.getoutput("wc -l %s" % args.data_file).split()[0])
    index_list = [i for i in range(file_length)]
    print("file length:%s" % file_length)

    print("start dirichlet distribution")
    min_size = 0
    partition_result = None
    while min_size < args.min_size:
        partition_result = [[] for _ in range(client_num)]
        partition_result = partition_class_samples_with_dirichlet_distribution(
            file_length, beta, client_num, partition_result, index_list)
        min_size = min([len(i) for i in partition_result])
        print("sample min size:%s" % min_size)


    result_dict = {str(j): i for i in range(len(partition_result)) for j in partition_result[i]}

    result_dict.update({"n_client": client_num, "beta": beta,
                        "partition_method": "niid_quantity_clients=%d_beta=%.1f" % (args.client_number, args.beta)})
    print("result size: ", len(result_dict))

    with open(args.partition_file, "wb") as f:
        pickle.dump(result_dict, f)
    print("partition finished")


if __name__ == '__main__':
    main()
