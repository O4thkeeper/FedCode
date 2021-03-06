python -m main.mrr_test_fedrod \
  --data_file "data/store/codesearch/python_test_3.jsonl" \
  --data_type "mrr_test" \
  --model_type '../../models/roberta-base' \
  --model_name "cache/codesearch/python_64/model/FedRod/" \
  --batch_size 64 \
  --test_batch_size 1000 \
  --output_dir "tmp/fedrod_codesearch_output/python" \
  --manual_seed 42 \
  --max_seq_length 200




python -m main.mrr_test_fedrod \
  --data_file "data/store/codesearch/python_test_3.jsonl" \
  --data_type "mrr_test" \
  --model_type '../../models/roberta-base' \
  --model_name "cache/codesearch/python_64_label/model/FedRod/" \
  --label_file data/store/codesearch/train_valid/python/data_label.pk \
  --batch_size 64 \
  --test_batch_size 1000 \
  --output_dir "tmp/fedrod_codesearch_output/python" \
  --manual_seed 42 \
  --max_seq_length 200

CUDA_VISIBLE_DEVICES=1 python -m main.mrr_test_fedrod \
  --data_file "data/store/codesearch/test_test_3.jsonl" \
  --data_type "mrr_test" \
  --model_type '../../models/roberta-base' \
  --model_name "cache/codesearch/test_64_label/model/FedRod/" \
  --label_file data/store/codesearch/train_valid/test/data_label.pk \
  --label_count 10 \
  --batch_size 64 \
  --test_batch_size 10 \
  --output_dir "tmp/fedrod_codesearch_output/test" \
  --manual_seed 42 \
  --max_seq_length 200

python -m main.mrr_test_fedrod \
  --data_file "data/store/codesearch/go_test_0.jsonl" \
  --data_type "mrr_test" \
  --model_type '../../models/roberta-base' \
  --model_name "cache/codesearch/go_64_label/model/FedRod/" \
  --label_file data/store/codesearch/train_valid/go/data_label.pk \
  --label_count 100 \
  --batch_size 64 \
  --test_batch_size 1000 \
  --output_dir "tmp/fedrod_codesearch_output/go" \
  --manual_seed 42 \
  --max_seq_length 200
