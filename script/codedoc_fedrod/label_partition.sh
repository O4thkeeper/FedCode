python -m data.manual_process.partition.kmeans_label \
  --cluster_number 10 \
  --batch_size 64 \
  --train_data_file data/store/codesearch/train_valid/python/train.txt \
  --test_data_file data/store/codesearch/python_test_3.jsonl \
  --label_file data/store/codesearch/train_valid/python/data_label.pk \
  --dataset codesearch

python -m data.manual_process.partition.label_partition \
  --client_num 64 \
  --label_file data/store/codesearch/train_valid/python/data_label.pk \
  --partition_file data/store/codesearch/train_valid/python/label_partition.pk \
  --seed 42 \
  --cluster_num 10 \
  --alpha 1.0

python -m data.manual_process.partition.kmeans_label \
  --cluster_number 10 \
  --batch_size 64 \
  --train_data_file data/store/codesearch/train_valid/test/train.txt \
  --test_data_file data/store/codesearch/test_test_3.jsonl \
  --label_file data/store/codesearch/train_valid/test/data_label.pk \
  --dataset codesearch

python -m data.manual_process.partition.label_partition \
  --client_num 5 \
  --label_file data/store/codesearch/train_valid/test/data_label.pk \
  --partition_file data/store/codesearch/train_valid/test/label_partition.pk \
  --seed 42 \
  --cluster_num 10 \
  --alpha 1.0

#todo fill