python -m main.codedoc_fedrod \
  --dataset "codedoc" \
  --language "test" \
  --test_data_file "data/store/codedoc/test/test.jsonl" \
  --fl_algorithm FedRod \
  --model_type "roberta-base" \
  --model_name "microsoft/codebert-base" \
  --load_model "data/store/codedoc/test/model/FedRod" \
  --do_lower_case True \
  --eval_batch_size 64 \
  --max_seq_length 256 \
  --max_target_length 128 \
  --beam_size 10 \
  --output_dir "tmp/fedavg_codedoc_output/" \
  --cache_dir "data/store/codedoc/test/" \
  --do_test

python -m main.codedoc_fedrod \
  --dataset "codedoc" \
  --language "python" \
  --test_data_file "data/store/codedoc/python/test_1.jsonl" \
  --fl_algorithm FedRod \
  --model_type "roberta-base" \
  --model_name "microsoft/codebert-base" \
  --load_model "cache/codedoc/python_64/model/FedRod/" \
  --do_lower_case True \
  --eval_batch_size 64 \
  --max_seq_length 256 \
  --max_target_length 128 \
  --beam_size 10 \
  --output_dir "tmp/fedavg_codedoc_output/" \
  --cache_dir "cache/codedoc/python_64" \
  --do_test

python -m main.codedoc_fedrod \
  --dataset "codedoc" \
  --language "java" \
  --test_data_file "data/store/codedoc/java/test_1.jsonl" \
  --fl_algorithm FedRod \
  --model_type "roberta-base" \
  --model_name "microsoft/codebert-base" \
  --load_model "cache/codedoc/java_64/model/FedRod/" \
  --do_lower_case True \
  --eval_batch_size 64 \
  --max_seq_length 256 \
  --max_target_length 128 \
  --beam_size 10 \
  --output_dir "tmp/fedavg_codedoc_output/" \
  --cache_dir "cache/codedoc/java_64" \
  --do_test