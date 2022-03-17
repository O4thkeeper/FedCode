python -m data.manual_process.partition.quantity_partition \
  --client_number 64 \
  --data_file data/store/codedoc/javascript/train.jsonl \
  --partition_file data/store/codedoc/javascript/train_partition_64.pk \
  --kmeans_num 0 \
  --beta 1 \
  --min_size 50
