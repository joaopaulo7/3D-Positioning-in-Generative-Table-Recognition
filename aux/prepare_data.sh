#! /usr/bin/bash

# wget https://dax-cdn.cdn.appdomain.cloud/dax-pubtabnet/2.0.0/pubtabnet.tar.gz -P data/

tar -C data/ -xf data/pubtabnet.tar.gz  pubtabnet/

mv data/pubtabnet data/imgs

mv data/imgs/PubTabNet_2.0.0.jsonl data/

mkdir data/anns data/anns/train data/anns/test data/anns/val


