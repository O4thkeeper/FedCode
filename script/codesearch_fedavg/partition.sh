
python -m data.manual_process.partition.quantity_partition \
  --client_number 15 \
  --data_file data/store/codesearch/train_valid/python/train.txt \
  --partition_file data/store/codesearch/train_valid/python/train_partition.pk \
  --kmeans_num 0 \
  --beta 1 \
  --min_size 10000


#python -m data.manual_process.partition.quantity_partition \
#  --client_number 15 \
#  --data_file data/store/codesearch/train_valid/go/train.txt \
#  --partition_file data/store/codesearch/train_valid/go/train_partition.pk \
#  --kmeans_num 0 \
#  --beta 1 \
#  --min_size 10000


#python -m data.manual_process.partition.quantity_partition \
#  --client_number 15 \
#  --data_file data/store/codesearch/train_valid/java/train.txt \
#  --partition_file data/store/codesearch/train_valid/java/train_partition.pk \
#  --kmeans_num 0 \
#  --beta 1 \
#  --min_size 10000

python -m data.manual_process.partition.quantity_partition \
  --client_number 15 \
  --data_file data/store/codesearch/train_valid/javascript/train.txt \
  --partition_file data/store/codesearch/train_valid/javascript/train_partition.pk \
  --kmeans_num 0 \
  --beta 1 \
  --min_size 2000

python -m data.manual_process.partition.quantity_partition \
  --client_number 15 \
  --data_file data/store/codesearch/train_valid/ruby/train.txt \
  --partition_file data/store/codesearch/train_valid/ruby/train_partition.pk \
  --kmeans_num 0 \
  --beta 1 \
  --min_size 1000

python -m data.manual_process.partition.quantity_partition \
  --client_number 64 \
  --data_file data/store/codesearch/train_valid/python/train.txt \
  --partition_file data/store/codesearch/train_valid/python/train_partition_64.pk \
  --kmeans_num 0 \
  --beta 1 \
  --min_size 1000