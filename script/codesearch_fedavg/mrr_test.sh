python -m main.mrr_test \
  --data_file "data/store/codesearch/go_test_3.jsonl" \
  --data_type "mrr_test" \
  --model_type "../../models/roberta-base" \
  --model_name "cache/codesearch/go_64/model/FedAvg/" \
  --batch_size 128 \
  --test_batch_size 1000 \
  --output_dir "tmp/fedavg_codesearch_output/go" \
  --manual_seed 42 \
  --max_seq_length 200