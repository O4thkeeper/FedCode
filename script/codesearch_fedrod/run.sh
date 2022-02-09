#py=/Users/fenghao/Documents/pythonWork/venv/bin/python
py=python

${py} -m main.codesearch_fedrod \
  --client_num_in_total 15 \
  --client_num_per_round 5 \
  --comm_round 9 \
  --dataset "codesearch" \
  --data_file "data/store/codesearch/python_train.h5" \
  --data_type "train" \
  --partition_file "data/store/codesearch/python_train_partition.h5" \
  --partition_method niid_quantity_clients=15_beta=1.0 \
  --eval_data_file "data/store/codesearch/train_valid/python/valid.txt" \
  --fl_algorithm "FedRod" \
  --model_type "roberta-base" \
  --model_name "microsoft/codebert-base" \
  --do_lower_case True \
  --train_batch_size 64 \
  --eval_batch_size 32 \
  --max_seq_length 200 \
  --learning_rate 5e-5 \
  --epochs 1 \
  --output_dir "tmp/fedrod_codesearch_output/" \
  --do_train
#  --do_eval