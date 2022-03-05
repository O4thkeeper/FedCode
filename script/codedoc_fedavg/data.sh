cd data/store/codedoc || exit
#unzip dataset.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip
#wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/ruby.zip
#wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/javascript.zip
#wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/go.zip
#wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/php.zip

unzip python.zip
unzip java.zip
#unzip ruby.zip
#unzip javascript.zip
#unzip go.zip
#unzip php.zip

python ../../manual_process/load/codedoc.py

rm *.zip
rm *.pkl
rm -r */final
