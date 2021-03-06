python -m main.codedoc_fedrod \
  --client_num_in_total 64 \
  --client_num_per_round 8 \
  --comm_round 16 \
  --dataset "codedoc" \
  --language "go" \
  --train_data_file "data/store/codedoc/go/train.jsonl" \
  --train_partition_file "data/store/codedoc/go/train_partition_64.pk" \
  --eval_data_file "data/store/codedoc/go/valid.jsonl" \
  --partition_method niid_quantity_clients=64_beta=1.0 \
  --fl_algorithm FedRod \
  --model_type "roberta-base" \
  --model_name "microsoft/codebert-base" \
  --do_lower_case True \
  --train_batch_size 32 \
  --eval_batch_size 32 \
  --max_seq_length 256 \
  --max_target_length 128 \
  --learning_rate 5e-5 \
  --beam_size 10 \
  --epochs 1 \
  --output_dir "tmp/fedrod_codedoc_output/" \
  --cache_dir "cache/codedoc/go_64" \
  --do_train \
  --do_eval