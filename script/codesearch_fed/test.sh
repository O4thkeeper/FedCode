#py=/Users/fenghao/Documents/pythonWork/venv/bin/python
py=python

${py} -m main.codesearch_fedavg \
  --client_num_in_total 15 \
  --client_num_per_round 5 \
  --comm_round 9 \
  --dataset "codesearch" \
  --data_file "data/store/codesearch/test/python/acc_test_data.txt" \
  --data_type "test" \
  --partition_file "data/store/codesearch/python_train_partition.h5" \
  --partition_method niid_quantity_clients=15_beta=1.0 \
  --fl_algorithm FedAvg \
  --model_type "roberta-base" \
  --model_name "cache/model" \
  --do_lower_case True \
  --train_batch_size 64 \
  --eval_batch_size 32 \
  --max_seq_length 200 \
  --learning_rate 5e-5 \
  --server_lr 0.1 \
  --epochs 1 \
  --output_dir "tmp/fedavg_codesearch_output/" \
  --do_test \
  --test_mode acc