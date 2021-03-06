python -m main.codedoc_fedavg \
  --dataset "codedoc" \
  --language "java" \
  --test_data_file "data/store/codedoc/java/test.jsonl" \
  --fl_algorithm FedAvg \
  --model_type "roberta-base" \
  --model_name "microsoft/codebert-base" \
  --load_model "cache/codedoc/java/model/FedAvg/model.pt" \
  --do_lower_case True \
  --eval_batch_size 32 \
  --max_seq_length 256 \
  --max_target_length 128 \
  --beam_size 10 \
  --output_dir "tmp/fedavg_codedoc_output/" \
  --cache_dir "cache/codedoc/java" \
  --do_test

python -m main.codedoc_fedavg \
  --dataset "codedoc" \
  --language "go" \
  --test_data_file "data/store/codedoc/go/test.jsonl" \
  --fl_algorithm FedAvg \
  --model_type "roberta-base" \
  --model_name "microsoft/codebert-base" \
  --load_model "cache/codedoc/go/model/FedAvg/model.pt" \
  --do_lower_case True \
  --eval_batch_size 32 \
  --max_seq_length 256 \
  --max_target_length 128 \
  --beam_size 10 \
  --output_dir "tmp/fedavg_codedoc_output/" \
  --cache_dir "cache/codedoc/go" \
  --do_test

python -m main.codedoc_fedavg \
  --dataset "codedoc" \
  --language "python" \
  --test_data_file "data/store/codedoc/python/test.jsonl" \
  --fl_algorithm FedAvg \
  --model_type "roberta-base" \
  --model_name "microsoft/codebert-base" \
  --load_model "cache/codedoc/python_64/model/FedAvg/model.pt" \
  --do_lower_case True \
  --eval_batch_size 64 \
  --max_seq_length 256 \
  --max_target_length 128 \
  --beam_size 10 \
  --output_dir "tmp/fedavg_codedoc_output/" \
  --cache_dir "cache/codedoc/python_64" \
  --do_test