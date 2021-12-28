#py=/Users/fenghao/Documents/pythonWork/venv/bin/python
py=python

#sh data/manual_process/download/codesearch.sh

${py} -m data.manual_process.load.codesearch \
  --data_dir data/store/codesearch \
  --h5_file_path data/store/codesearch/ \
  --data_type train

# todo change params
#${py} -m data.manual_process.partition.niid_quantity \
#--client_number 50  \
#--data_file data/store/20news/20news_data.h5  \
#--partition_file data/store/20news/20news_partition.h5 \
#--task_type text_classification \
#--kmeans_num 0 \
#--beta 1
