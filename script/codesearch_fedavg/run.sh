#py=/Users/fenghao/Documents/pythonWork/venv/bin/python
py=python

${py} -m main.codesearch_fedavg \
  --client_num_in_total 15 \
  --client_num_per_round 5 \
  --comm_round 9 \
  --dataset "codesearch" \
  --language "python" \
  --train_data_file "data/store/codesearch/train_valid/python/train.txt" \
  --train_partition_file "data/store/codesearch/train_valid/python/train_partition.pk" \
  --partition_method niid_quantity_clients=15_beta=1.0 \
  --fl_algorithm FedAvg \
  --model_type 'roberta-base' \
  --model_name 'microsoft/codebert-base' \
  --do_lower_case True \
  --train_batch_size 64 \
  --eval_batch_size 32 \
  --max_seq_length 200 \
  --learning_rate 5e-5 \
  --epochs 1 \
  --output_dir "tmp/fedavg_codesearch_output/" \
  --cache_dir "cache/codesearch/python" \
  --do_train


python -m main.codesearch_fedavg \
  --client_num_in_total 15 \
  --client_num_per_round 5 \
  --comm_round 9 \
  --dataset "codesearch" \
  --language "java" \
  --train_data_file "data/store/codesearch/train_valid/java/train.txt" \
  --train_partition_file "data/store/codesearch/train_valid/java/train_partition.pk" \
  --partition_method niid_quantity_clients=15_beta=1.0 \
  --fl_algorithm FedAvg \
  --model_type 'roberta-base' \
  --model_name 'microsoft/codebert-base' \
  --do_lower_case True \
  --train_batch_size 64 \
  --eval_batch_size 32 \
  --max_seq_length 200 \
  --learning_rate 5e-5 \
  --epochs 1 \
  --output_dir "tmp/fedavg_codesearch_output/" \
  --cache_dir "cache/codesearch/java" \
  --do_train

python -m main.codesearch_fedavg \
   --client_num_in_total 15 \
   --client_num_per_round 5 \
   --comm_round 9 \
   --dataset "codesearch" \
   --language "go" \
  --train_data_file "data/store/codesearch/train_valid/go/train.txt" \
   --train_partition_file "data/store/codesearch/train_valid/go/train_partition.pk" \
   --partition_method niid_quantity_clients=15_beta=1.0 \
   --fl_algorithm FedAvg \
   --model_type 'roberta-base' \
   --model_name 'microsoft/codebert-base' \
   --do_lower_case True \
   --train_batch_size 64 \
   --eval_batch_size 32 \
   --max_seq_length 200 \
   --learning_rate 5e-5 \
   --epochs 1 \
   --output_dir "tmp/fedavg_codesearch_output/" \
   --cache_dir "cache/codesearch/go" \
   --do_train


python -m main.codesearch_fedavg \
   --client_num_in_total 64 \
   --client_num_per_round 8 \
   --comm_round 12 \
   --dataset "codesearch" \
   --language "javascript" \
   --train_data_file "data/store/codesearch/train_valid/javascript/train.txt" \
   --train_partition_file "data/store/codesearch/train_valid/javascript/train_partition_64.pk" \
   --partition_method niid_quantity_clients=64_beta=1.0 \
   --fl_algorithm FedAvg \
   --model_type 'roberta-base' \
   --model_name 'microsoft/codebert-base' \
   --do_lower_case True \
   --train_batch_size 64 \
   --eval_batch_size 32 \
   --max_seq_length 200 \
   --learning_rate 1e-5 \
   --epochs 1 \
   --output_dir "tmp/fedavg_codesearch_output/" \
   --cache_dir "cache/codesearch/javascript_64" \
   --do_train

python -m main.codesearch_fedavg \
   --client_num_in_total 64 \
   --client_num_per_round 8 \
   --comm_round 12 \
   --dataset "codesearch" \
   --language "ruby" \
   --train_data_file "data/store/codesearch/train_valid/ruby/train.txt" \
   --train_partition_file "data/store/codesearch/train_valid/ruby/train_partition_64.pk" \
   --partition_method niid_quantity_clients=64_beta=1.0 \
   --fl_algorithm FedAvg \
   --model_type '../../models/roberta-base' \
   --model_name '../../models/codebert-base' \
   --do_lower_case True \
   --train_batch_size 64 \
   --eval_batch_size 32 \
   --max_seq_length 200 \
   --learning_rate 1e-5 \
   --epochs 1 \
   --output_dir "tmp/fedavg_codesearch_output/" \
   --cache_dir "cache/codesearch/ruby_64" \
   --do_train