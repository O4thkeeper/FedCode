#py=/Users/fenghao/Documents/pythonWork/venv/bin/python
py=python

sh data/manual_process/download/codesearch.sh

${py} -m data.manual_process.load.codesearch \
  --data_dir data/store/codesearch \
  --h5_file_path data/store/codesearch/ \
  --data_type train