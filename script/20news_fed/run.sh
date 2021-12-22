#py=/Users/fenghao/Documents/pythonWork/venv/bin/python
py=python

${py} -m main.main_20news_serial \
  --client_num_in_total 50 \
  --client_num_per_round 10 \
  --comm_round 30 \
  --dataset "20news" \
  --data_file "data/store/20news/20news_data.h5" \
  --partition_file "data/store/20news/20news_partition.h5" \
  --partition_method niid_quantity_clients=5_beta=5.0 \
  --fl_algorithm FedAvg \
  --model_type distilbert \
  --model_name distilbert-base-uncased \
  --do_lower_case True \
  --train_batch_size 32 \
  --eval_batch_size 8 \
  --max_seq_length 256 \
  --lr 5e-5 \
  --server_lr 0.1 \
  --epochs 1 \
  --output_dir "tmp/fedavg_20news_output/"