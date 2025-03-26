#! /usr/bin/bash

cd aux

echo "decompressing data"

tar -C data/ -xf data/pubtabnet.tar.gz  pubtabnet/

mv data/pubtabnet data/imgs

mv data/imgs/PubTabNet_2.0.0.jsonl data/

unzip data/final_eval.zip  -d data/imgs/final_eval

echo "decompression done. Prepping data."

mkdir data/anns data/anns/train data/anns/test data/anns/val

python3 Prep_Donut.py

python3 Prep_Val.py

echo "Prepping done. Training processors."

python3 Train_Processor.py

echo "Creating auxiliary directories"

mkdir models/by_step/3D_TML models/by_step/3D_HTML models/by_step/Pos_Enc

mkdir outputs/3D_TML outputs/3D_HTML outputs/Pos_Enc

mkdir outputs/3D_TML/evals outputs/3D_HTML/evals outputs/Pos_Enc/evals
