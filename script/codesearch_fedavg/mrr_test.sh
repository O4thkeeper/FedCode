#py=/Users/fenghao/Documents/pythonWork/venv/bin/python
py=python

${py} -m main.mrr_test \
  --data_file "data/store/codesearch/python_test_1.jsonl" \
  --data_type "mrr_test" \
  --model_type "roberta-base" \
  --model_name "cache/model" \
  --batch_size 64 \
  --test_batch_size 1000 \
  --output_dir "tmp/fedavg_codesearch_output/" \
  --manual_seed 42 \
  --max_seq_length 200