#py=/Users/fenghao/Documents/pythonWork/venv/bin/python
py=python

${py} -m main.mrr_test \
  --data_file "data/store/codesearch/python_test_0.jsonl" \
  --data_type "mrr_test" \
  --model_type "roberta-base" \
  --model_name "cache/model" \
  --batch_size 64 \
  --output_dir "tmp/fedavg_codesearch_output/"