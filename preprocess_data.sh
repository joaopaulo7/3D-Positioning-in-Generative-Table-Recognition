#! /usr/bin/bash

cd aux

tar -C data/ -xf data/pubtabnet.tar.gz  pubtabnet/

mv data/pubtabnet data/imgs

mv data/imgs/PubTabNet_2.0.0.jsonl data/

mkdir data/anns data/anns/train data/anns/test data/anns/val

python3 Prep_Donut.py

python3 Train_Processor.py
