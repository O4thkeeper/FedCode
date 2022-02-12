#py=/Users/fenghao/Documents/pythonWork/venv/bin/python
py=python

${py} -m data.manual_process.partition.quantity_partition \
  --client_number 15 \
  --data_file data/store/codedoc/python/train.jsonl \
  --partition_file data/store/codedoc/python/train_partition.pk \
  --kmeans_num 0 \
  --beta 1 \
  --min_size 5000

${py} -m data.manual_process.partition.quantity_partition \
  --client_number 15 \
  --data_file data/store/codedoc/python/eval.jsonl \
  --partition_file data/store/codedoc/python/eval_partition.pk \
  --kmeans_num 0 \
  --beta 1 \
  --min_size 200

#${py} -m data.manual_process.partition.quantity_partition \
#  --client_number 15 \
#  --data_file data/store/codedoc/python_test.txt \
#  --partition_file data/store/codedoc/python_test_partition.pk \
#  --kmeans_num 0 \
#  --beta 1 \
#  --min_size 1000
