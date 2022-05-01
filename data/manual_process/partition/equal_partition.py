import argparse
import pickle
import random
import subprocess


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--client_number", type=int, default="100")

    parser.add_argument("--data_file", type=str, default="")

    parser.add_argument("--partition_file", type=str, default="")

    args = parser.parse_args()

    client_num = args.client_number

    file_length = int(subprocess.getoutput("wc -l %s" % args.data_file).split()[0])
    index_list = [i for i in range((file_length // client_num + 1) * client_num)]
    print("file length:%s" % file_length)
    partition_result = []
    random.shuffle(index_list)
    size_per_client = len(index_list) // client_num
    print("size_per_client:%s" % size_per_client)
    for i in range(client_num):
        partition_result.append(index_list[i * size_per_client:i * size_per_client + size_per_client])

    result_dict = {str(j): i for i in range(len(partition_result)) for j in partition_result[i]}

    result_dict.update({"n_client": client_num, "beta": 0,
                        "partition_method": "equal_quantity_clients=%d" % args.client_number})

    with open(args.partition_file, "wb") as f:
        pickle.dump(result_dict, f)
    print("partition finished")


if __name__ == '__main__':
    main()
