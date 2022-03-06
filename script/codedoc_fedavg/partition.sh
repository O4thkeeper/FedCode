python -m data.manual_process.partition.quantity_partition \
  --client_number 15 \
  --data_file data/store/codedoc/python/train.jsonl \
  --partition_file data/store/codedoc/python/train_partition.pk \
  --kmeans_num 0 \
  --beta 1 \
  --min_size 5000

#${py} -m data.manual_process.partition.quantity_partition \
#  --client_number 15 \
#  --data_file data/store/codedoc/python/eval.jsonl \
#  --partition_file data/store/codedoc/python/eval_partition.pk \
#  --kmeans_num 0 \
#  --beta 1 \
#  --min_size 200

#${py} -m data.manual_process.partition.quantity_partition \
#  --client_number 15 \
#  --data_file data/store/codedoc/python_test.txt \
#  --partition_file data/store/codedoc/python_test_partition.pk \
#  --kmeans_num 0 \
#  --beta 1 \
#  --min_size 1000

python -m data.manual_process.partition.quantity_partition \
  --client_number 15 \
  --data_file data/store/codedoc/java/train.jsonl \
  --partition_file data/store/codedoc/java/train_partition.pk \
  --kmeans_num 0 \
  --beta 1 \
  --min_size 2000


python -m data.manual_process.partition.quantity_partition \
  --client_number 15 \
  --data_file data/store/codedoc/go/train.jsonl \
  --partition_file data/store/codedoc/go/train_partition.pk \
  --kmeans_num 0 \
  --beta 1 \
  --min_size 1000