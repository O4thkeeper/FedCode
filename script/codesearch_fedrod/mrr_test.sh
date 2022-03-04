#py=/Users/fenghao/Documents/pythonWork/venv/bin/python
py=python

${py} -m main.mrr_test_fedrod \
  --data_file "data/store/codesearch/python_test_1.jsonl" \
  --data_type "mrr_test" \
  --model_type "roberta-base" \
  --model_name "cache/codesearch/python/model/FedRod/" \
  --batch_size 64 \
  --test_batch_size 1000 \
  --output_dir "tmp/fedrod_codesearch_output/python" \
  --manual_seed 42 \
  --max_seq_length 200
