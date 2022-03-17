python -m main.codesearch_fedrod \
  --client_num_in_total 64 \
  --client_num_per_round 8 \
  --comm_round 12 \
  --dataset "codesearch" \
  --language "php" \
  --train_data_file "data/store/codesearch/train_valid/php/train.txt" \
  --train_partition_file "data/store/codesearch/train_valid/php/train_partition_64.pk" \
  --partition_method niid_quantity_clients=64_beta=1.0 \
  --eval_data_file "data/store/codesearch/train_valid/php/valid.txt" \
  --fl_algorithm "FedRod" \
  --model_type '../../models/roberta-base' \
  --model_name '../../models/codebert-base' \
  --do_lower_case True \
  --train_batch_size 64 \
  --eval_batch_size 32 \
  --max_seq_length 200 \
  --learning_rate 1e-5 \
  --epochs 1 \
  --output_dir "tmp/fedrod_codesearch_output/" \
  --cache_dir "cache/codesearch/php_64" \
  --do_train

python -m main.codesearch_fedrod \
  --client_num_in_total 64 \
  --client_num_per_round 8 \
  --comm_round 12 \
  --dataset "codesearch" \
  --language "python" \
  --train_data_file "data/store/codesearch/train_valid/python/train.txt" \
  --train_partition_file "data/store/codesearch/train_valid/python/label_partition.pk" \
  --partition_method niid_label_clients=64_beta=1.0 \
  --label_count 10 \
  --fl_algorithm "FedRod" \
  --model_type '../../models/roberta-base' \
  --model_name '../../models/codebert-base' \
  --do_lower_case True \
  --train_batch_size 64 \
  --eval_batch_size 32 \
  --max_seq_length 200 \
  --learning_rate 1e-5 \
  --epochs 1 \
  --output_dir "tmp/fedrod_codesearch_output/" \
  --cache_dir "cache/codesearch/python_64_label" \
  --do_train

python -m main.codesearch_fedrod \
  --client_num_in_total 5 \
  --client_num_per_round 3 \
  --comm_round 2 \
  --dataset "codesearch" \
  --language "test" \
  --train_data_file "data/store/codesearch/train_valid/test/train.txt" \
  --train_partition_file "data/store/codesearch/train_valid/test/label_partition.pk" \
  --partition_method niid_label_clients=5_beta=1.0 \
  --label_count 10 \
  --fl_algorithm "FedRod" \
  --model_type '../../models/roberta-base' \
  --model_name '../../models/codebert-base' \
  --do_lower_case True \
  --train_batch_size 64 \
  --eval_batch_size 32 \
  --max_seq_length 200 \
  --learning_rate 1e-5 \
  --epochs 1 \
  --output_dir "tmp/fedrod_codesearch_output/" \
  --cache_dir "cache/codesearch/test_64_label" \
  --do_train