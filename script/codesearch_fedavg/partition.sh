#py=/Users/fenghao/Documents/pythonWork/venv/bin/python
py=python

${py} -m data.manual_process.partition.quantity_partition \
  --client_number 15 \
  --data_file data/store/codesearch/train_valid/python/train.txt \
  --partition_file data/store/codesearch/train_valid/python/train_partition.pk \
  --kmeans_num 0 \
  --beta 1 \
  --min_size 8000