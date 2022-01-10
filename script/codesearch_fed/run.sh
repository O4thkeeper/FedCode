#py=/Users/fenghao/Documents/pythonWork/venv/bin/python
py=python

${py} -m main.codesearch_fedavg \
  --client_num_in_total 3 \
  --client_num_per_round 1 \
  --comm_round 2 \
  --dataset "codesearch" \
  --data_file "data/store/codesearch/python_train.h5" \
  --data_type "train" \
  --partition_file "data/store/codesearch/python_train_partition.h5" \
  --partition_method niid_quantity_clients=3_beta=1.0 \
  --fl_algorithm FedAvg \
  --model_type distilbert \
  --model_name distilbert-base-uncased \
  --do_lower_case True \
  --train_batch_size 64 \
  --eval_batch_size 32 \
  --max_seq_length 200 \
  --learning_rate 5e-5 \
  --server_lr 0.1 \
  --epochs 3 \
  --output_dir "tmp/fedavg_codesearch_output/"