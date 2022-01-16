#py=/Users/fenghao/Documents/pythonWork/venv/bin/python
py=python

${py} -m data.manual_process.partition.niid_quantity \
--client_number 15  \
--data_file data/store/codesearch/python_train.h5  \
--partition_file data/store/codesearch/python_train_partition.h5 \
--task_type text_classification \
--kmeans_num 0 \
--beta 1 \
--min_size 10000 \
--mode 1