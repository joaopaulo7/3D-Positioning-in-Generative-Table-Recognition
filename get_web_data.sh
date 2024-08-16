#! /usr/bin/bash

cd aux

wget https://dax-cdn.cdn.appdomain.cloud/dax-pubtabnet/2.0.0/pubtabnet.tar.gz -P data/

wget https://github.com/ajjimeno/icdar-task-b/raw/master/final_eval.zip -P data/
wget https://github.com/ajjimeno/icdar-task-b/raw/master/final_eval.json -P data/anns/test/

python3 Get_Models.py
