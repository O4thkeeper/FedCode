#py=/Users/fenghao/Documents/pythonWork/venv/bin/python
py=python

${py} -m data.manual_process.test_data.codesearch_test_data \
  --language python \
  --type acc \
  --data_dir 'data/store/codesearch'
#  --batch_size
