
python -m main.mrr_test \
  --data_file "data/store/codesearch/python_test_1.jsonl" \
  --data_type "mrr_test" \
  --model_type "roberta-base" \
  --model_name "cache/codesearch/python/model/FedAvg/" \
  --batch_size 64 \
  --test_batch_size 1000 \
  --output_dir "tmp/fedavg_codesearch_output/python" \
  --manual_seed 42 \
  --max_seq_length 200


python -m main.mrr_test \
  --data_file "data/store/codesearch/go_test_1.jsonl" \
  --data_type "mrr_test" \
  --model_type "roberta-base" \
  --model_name "cache/codesearch/go/model/FedAvg/" \
  --batch_size 64 \
  --test_batch_size 1000 \
  --output_dir "tmp/fedavg_codesearch_output/go" \
  --manual_seed 42 \
  --max_seq_length 200

python -m main.mrr_test \
  --data_file "data/store/codesearch/java_test_1.jsonl" \
  --data_type "mrr_test" \
  --model_type "roberta-base" \
  --model_name "cache/codesearch/java/model/FedAvg/" \
  --batch_size 64 \
  --test_batch_size 1000 \
  --output_dir "tmp/fedavg_codesearch_output/java" \
  --manual_seed 42 \
  --max_seq_length 200

python -m main.mrr_test \
  --data_file "data/store/codesearch/javascript_test_0.jsonl" \
  --data_type "mrr_test" \
  --model_type "../../models/roberta-base" \
  --model_name "cache/codesearch/javascript_64/model/FedAvg/" \
  --batch_size 64 \
  --test_batch_size 1000 \
  --output_dir "tmp/fedavg_codesearch_output/javascript" \
  --manual_seed 42 \
  --max_seq_length 200