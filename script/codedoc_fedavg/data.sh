#py=/Users/fenghao/Documents/pythonWork/venv/bin/python
py=python

cd data/store/codedoc || exit
#unzip dataset.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip
#wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip
#wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/ruby.zip
#wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/javascript.zip
#wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/go.zip
#wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/php.zip

unzip python.zip
#unzip java.zip
#unzip ruby.zip
#unzip javascript.zip
#unzip go.zip
#unzip php.zip
#rm *.zip
rm *.pkl

${py} ../../manual_process/load/codedoc.py
rm -r */final
