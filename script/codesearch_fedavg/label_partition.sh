python -m data.manual_process.partition.kmeans_label \
  --cluster_number 5 \
  --batch_size 16 \
  --data_file data/store/codesearch/train_valid/python/train.txt \
  --label_file data/store/codesearch/train_valid/python/train_label.pk \
  --dataset codesearch

python -m data.manual_process.partition.label_partition \
  --client_num 6 \
  --label_file data/store/codesearch/train_valid/python/train_label.pk \
  --partition_file data/store/codesearch/train_valid/python/train_pariton_kl.pk \
  --seed 42 \
  --cluster_num 5 \
  --alpha 1.0
