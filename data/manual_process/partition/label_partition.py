import array
import pickle
import argparse
import numpy as np
import math
import random


def dynamic_batch_fill(label_index_tracker, label_index_matrix, remaining_length, current_label_id):
    """
    params
    ------------------------------------------------------------------------
    label_index_tracker : 1d numpy array track how many data each label has used
    label_index_matrix : 2d array list of indexs of each label
    remaining_length : int remaining empty space in current partition client list
    current_label_id : int current round label id
    ------------------------------------------------------------------------

    return
    ---------------------------------------------------------
    label_index_offset: dict  dictionary key is label id
    and value is the offset associated with this key
    ----------------------------------------------------------
    """
    remaining_unfiled = remaining_length
    label_index_offset = {}
    label_remain_length_dict = {}
    total_label_remain_length = 0
    # calculate total number of all the remaing labels and each label's remaining length
    for label_id, label_list in enumerate(label_index_matrix):
        if label_id == current_label_id:
            label_remain_length_dict[label_id] = 0
            continue
        label_remaining_count = len(label_list) - label_index_tracker[label_id]
        if label_remaining_count > 0:
            total_label_remain_length = (total_label_remain_length +
                                         label_remaining_count)
        else:
            label_remaining_count = 0
        label_remain_length_dict[label_id] = label_remaining_count
    length_pointer = remaining_unfiled

    if total_label_remain_length > 0:
        label_sorted_by_length = {
            k: v
            for k, v in sorted(label_remain_length_dict.items(),
                               key=lambda item: item[1])
        }
    else:
        label_index_offset = label_remain_length_dict
        return label_index_offset
    # for each label calculate the offset move forward by distribution of remaining labels
    for label_id in label_sorted_by_length.keys():
        fill_count = math.ceil(label_remain_length_dict[label_id] /
                               total_label_remain_length * remaining_length)
        fill_count = min(fill_count, label_remain_length_dict[label_id])
        offset_forward = fill_count
        # if left room not enough for all offset set it to 0
        if length_pointer - offset_forward <= 0 and length_pointer > 0:
            label_index_offset[label_id] = length_pointer
            length_pointer = 0
            break
        else:
            length_pointer -= offset_forward
            label_remain_length_dict[label_id] -= offset_forward
        label_index_offset[label_id] = offset_forward

    # still has some room unfilled
    if length_pointer > 0:
        for label_id in label_sorted_by_length.keys():
            # make sure no infinite loop happens
            fill_count = math.ceil(label_sorted_by_length[label_id] /
                                   total_label_remain_length * length_pointer)
            fill_count = min(fill_count, label_remain_length_dict[label_id])
            offset_forward = fill_count
            if length_pointer - offset_forward <= 0 and length_pointer > 0:
                label_index_offset[label_id] += length_pointer
                length_pointer = 0
                break
            else:
                length_pointer -= offset_forward
                label_remain_length_dict[label_id] -= offset_forward
            label_index_offset[label_id] += offset_forward

    return label_index_offset


def label_skew_process(label_vocab, label_assignment, client_num, alpha, data_length):
    """
    params
    -------------------------------------------------------------------
    label_vocab : dict label vocabulary of the dataset
    label_assignment : 1d list a list of label, the index of list is the index associated to label
    client_num : int number of clients
    alpha : float similarity of each client, the larger the alpha the similar data for each client
    -------------------------------------------------------------------
    return
    ------------------------------------------------------------------
    partition_result : 2d array list of partition index of each client
    ------------------------------------------------------------------
    """
    label_index_matrix = [[] for _ in label_vocab]
    label_proportion = []
    partition_result = [[] for _ in range(client_num)]
    client_length = 0
    print("client_num", client_num)
    # shuffle indexs and calculate each label proportion of the dataset
    for index, value in enumerate(label_vocab):
        label_location = np.where(label_assignment == value)[0]
        label_proportion.append(len(label_location) / data_length)
        np.random.shuffle(label_location)
        label_index_matrix[index].extend(label_location[:])
    print(label_proportion)
    # calculate size for each partition client
    label_index_tracker = np.zeros(len(label_vocab), dtype=int)
    total_index = data_length
    each_client_index_length = int(total_index / client_num)
    print("each index length", each_client_index_length)
    client_dir_dis = np.array([alpha * l for l in label_proportion])
    print("alpha", alpha)
    print("client dir dis", client_dir_dis)
    proportions = np.random.dirichlet(client_dir_dis)
    print("dir distribution", proportions)
    # add all the unused data to the client
    for client_id in range(len(partition_result)):
        each_client_partition_result = partition_result[client_id]
        proportions = np.random.dirichlet(client_dir_dis)
        client_length = min(each_client_index_length, total_index)
        if total_index < client_length * 2:
            client_length = total_index
        total_index -= client_length
        client_length_pointer = client_length
        # for each label calculate the offset length assigned to by Dir distribution and then extend assignment
        for label_id, _ in enumerate(label_vocab):
            offset = round(proportions[label_id] * client_length)
            if offset >= client_length_pointer:
                offset = client_length_pointer
                client_length_pointer = 0
            else:
                if label_id == (len(label_vocab) - 1):
                    offset = client_length_pointer
                client_length_pointer -= offset

            start = int(label_index_tracker[label_id])
            end = int(label_index_tracker[label_id] + offset)
            label_data_length = len(label_index_matrix[label_id])
            # if the the label is assigned to a offset length that is more than what its remaining length
            if end > label_data_length:
                each_client_partition_result.extend(
                    label_index_matrix[label_id][start:])
                label_index_tracker[label_id] = label_data_length
                label_index_offset = dynamic_batch_fill(
                    label_index_tracker, label_index_matrix,
                    end - label_data_length, label_id)
                for fill_label_id in label_index_offset.keys():
                    start = label_index_tracker[fill_label_id]
                    end = (label_index_tracker[fill_label_id] +
                           label_index_offset[fill_label_id])
                    each_client_partition_result.extend(
                        label_index_matrix[fill_label_id][start:end])
                    label_index_tracker[fill_label_id] = (
                            label_index_tracker[fill_label_id] +
                            label_index_offset[fill_label_id])
            else:
                each_client_partition_result.extend(
                    label_index_matrix[label_id][start:end])
                label_index_tracker[
                    label_id] = label_index_tracker[label_id] + offset

        # if last client still has empty rooms, fill empty rooms with the rest of the unused data
        if client_id == len(partition_result) - 1:
            print("last id length", len(each_client_partition_result))
            print("Last client fill the rest of the unfilled lables.")
            for not_fillall_label_id in range(len(label_vocab)):
                if label_index_tracker[not_fillall_label_id] < len(
                        label_index_matrix[not_fillall_label_id]):
                    print("fill more id", not_fillall_label_id)
                    start = label_index_tracker[not_fillall_label_id]
                    each_client_partition_result.extend(
                        label_index_matrix[not_fillall_label_id][start:])
                    label_index_tracker[not_fillall_label_id] = len(
                        label_index_matrix[not_fillall_label_id])
        partition_result[client_id] = each_client_partition_result

    return partition_result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--client_num", type=int, default=64)

    parser.add_argument("--label_file", type=str, default="", )

    parser.add_argument("--partition_file", type=str, default="")

    parser.add_argument("--label_weight_file", type=str, default="")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--cluster_num", type=int)

    parser.add_argument("--alpha", type=float, default=1.0)

    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    with open(args.label_file, 'rb') as f:
        label_assignment, train_len = pickle.load(f)
    label_assignment = label_assignment[:train_len]

    print("start data processing")
    label_vocab = [i for i in range(args.cluster_num)]
    data_per_client = label_skew_process(label_vocab, label_assignment, args.client_num, args.alpha,
                                         len(label_assignment))

    owner_of_data = array.array('H', [0 for _ in range(len(label_assignment))])
    label_weight_list = []
    result_dict = {}
    result_dict.update({"n_client": args.client_num, "beta": args.alpha,
                        "partition_method": "niid_label_clients=%d_beta=%.1f" % (args.client_num, args.alpha)})

    for client_id, client_data in enumerate(data_per_client):
        label_weight = [0 for _ in range(args.cluster_num)]
        for data_id in client_data:
            owner_of_data[data_id] = client_id
            label_weight[label_assignment[data_id]] += 1
        label_weight_list.append(label_weight)
    result_dict['label_weight'] = label_weight_list
    result_dict['owner_of_data'] = owner_of_data
    with open(args.partition_file, 'wb') as f:
        pickle.dump(result_dict, f)


if __name__ == '__main__':
    main()
