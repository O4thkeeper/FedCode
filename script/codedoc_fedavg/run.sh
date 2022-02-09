#py=/Users/fenghao/Documents/pythonWork/venv/bin/python
py=python

${py} -m main.codedoc_fedavg \
  --client_num_in_total 15 \
  --client_num_per_round 5 \
  --comm_round 9 \
  --dataset "codedoc" \
  --train_data_file "data/store/codedoc/python/train.jsonl" \
  --train_partition_file "data/store/codedoc/python/train_partition.pk" \
  --eval_data_file "data/store/codedoc/python/eval.jsonl" \
  --eval_partition_file "data/store/codedoc/python/eval_partition.pk" \
  --partition_method niid_quantity_clients=15_beta=1.0 \
  --fl_algorithm FedAvg \
  --model_type 'roberta-base' \
  --model_name 'microsoft/codebert-base' \
  --do_lower_case True \
  --train_batch_size 64 \
  --eval_batch_size 32 \
  --max_seq_length 256 \
  --max_target_length 128 \
  --learning_rate 5e-5 \
  --beam_size 1000 \
  --epochs 1 \
  --output_dir "tmp/fedavg_codedoc_output/" \
  --do_train \
  --do_eval